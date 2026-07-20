# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression test: MultiprocessRequestMetricCollector must not silently drop metrics.

Collection stops on a single ``None`` sentinel put by the main process, while
metrics are put() by worker processes. Since ``Queue.put()`` only buffers into a
per-process feeder thread with no cross-producer ordering, the sentinel can
overtake metrics that were put() earlier but not yet flushed: the collector
reads ``None``, stops, and drops them. They surface as neither successes nor
failures (the production symptom: summary reports fewer requests than were sent,
with zero failures).

Mirrors load_generator.py: each worker put()s a metric then bumps a shared
counter, and the main side tears the collector down once the counter is
satisfied. Metrics are large so feeders lag, as a 10k-token stream does live.

FAILS on current main; passes once start() drains the queue (queue.join) before
sending the sentinel.
"""

import asyncio
import multiprocessing as mp
import sys
import time
import unittest
from multiprocessing.sharedctypes import Synchronized
from typing import Optional

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text

# Match the production start method (main.py) so shared objects are inherited.
if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

# ~450 KB per metric (a large streaming response's lifecycle record).
_BODY = "x" * 400_000
_CHUNKS = [f'{{"choices":[{{"text":"{i}"}}]}}' for i in range(2000)]
_TIMES = [float(i) for i in range(2000)]


def _metric() -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="{}",
        response_data=_BODY,
        error=None,
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=_CHUNKS,
                chunk_times=_TIMES,
                output_tokens=len(_CHUNKS),
                output_token_times=_TIMES,
            ),
        ),
    )


def _producer(queue: "mp.JoinableQueue[Optional[RequestLifecycleMetric]]", finished: "Synchronized[int]", count: int) -> None:
    # Same ordering as Worker.schedule_client's finally: record, then count.
    for _ in range(count):
        queue.put(_metric())
        with finished.get_lock():
            finished.value += 1


class TestMultiprocessCollectorNoDrops(unittest.IsolatedAsyncioTestCase):
    async def test_all_recorded_metrics_are_collected(self) -> None:
        workers, per_worker = 4, 60
        total = workers * per_worker
        collector = MultiprocessRequestMetricCollector()
        finished: "Synchronized[int]" = mp.Value("i", 0)
        procs = [mp.Process(target=_producer, args=(collector.queue, finished, per_worker)) for _ in range(workers)]

        try:
            async with collector.start():
                for p in procs:
                    p.start()
                # Wait only on the counter, exactly like run_stage; then fall out
                # of the context, which sends the sentinel while feeders may lag.
                deadline = time.monotonic() + 60.0
                while finished.value < total:
                    self.assertLess(time.monotonic(), deadline, "producers stalled")
                    await asyncio.sleep(0.01)
            collected = len(collector.get_metrics())
        finally:
            # Undelivered items keep buggy-path feeders blocked, so terminate
            # rather than join (a harmless no-op once workers exit cleanly).
            for p in procs:
                p.terminate()
                p.join(timeout=2.0)

        self.assertEqual(collected, total, f"collector dropped {total - collected}/{total} metrics with zero failures")
