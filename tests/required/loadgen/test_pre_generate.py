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
"""Integration tests for pre-generation (load.pre_generate).

These exercise the full multi-worker LoadGenerator with the concurrent load type to verify:
- Requests still complete and produce unique, deterministic prompts.
- The warmup handshake does not deadlock, including when concurrency_level < num_workers
  (some workers are assigned 0 concurrency but must still join the warmup barrier).
- Runs are byte-for-byte reproducible for a given base_seed.
"""

import ast
import multiprocessing as mp
import sys
import unittest
from unittest.mock import patch
from typing import Any, List

from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    ConcurrentLoadStage,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
)
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# Match production (main.py) start method so datagen/tokenizer objects are inherited via fork.
if sys.platform == "darwin":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass


class _DummyHFTokenizer:
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)

    def encode(self, text: str) -> list[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return list(range(10, 10010))


class _DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _DummyHFTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) if text else 0


def _make_concurrent_config(
    num_requests: int,
    concurrency_level: int,
    num_workers: int,
    pre_generate: bool,
    base_seed: int = 42,
) -> LoadConfig:
    stage = ConcurrentLoadStage(num_requests=num_requests, concurrency_level=concurrency_level)
    # Replicate the runtime adjustments main.py applies for the concurrent load type.
    stage.duration = 1
    stage.rate = num_requests
    return LoadConfig(
        type=LoadType.CONCURRENT,
        num_workers=num_workers,
        worker_max_concurrency=0,
        stages=[stage],
        base_seed=base_seed,
        pre_generate=pre_generate,
    )


async def _run_and_collect_prompts(
    num_requests: int,
    concurrency_level: int,
    num_workers: int,
    pre_generate: bool,
    base_seed: int = 42,
) -> List[str]:
    api_config = APIConfig(type=APIType.Completion, streaming=False)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=num_requests + 1),
        output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=num_requests + 1),
    )
    datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer(), seed=base_seed)

    collector = MultiprocessRequestMetricCollector()
    client = MockModelServerClient(collector, api_config, mock_latency=0)

    load_config = _make_concurrent_config(num_requests, concurrency_level, num_workers, pre_generate, base_seed)
    load_gen = LoadGenerator(datagen, load_config)

    async with collector.start():
        await load_gen.mp_run(client)

    metrics = collector.get_metrics()
    return [ast.literal_eval(m.request_data)["prompt"] for m in metrics]


class TestPreGenerateConcurrent(unittest.IsolatedAsyncioTestCase):
    async def test_pre_generate_completes_with_unique_prompts(self) -> None:
        """A concurrent stage with pre_generate on completes and yields unique prompts."""
        num_requests = 8
        prompts = await _run_and_collect_prompts(
            num_requests=num_requests, concurrency_level=4, num_workers=2, pre_generate=True
        )
        self.assertEqual(len(prompts), num_requests)
        self.assertEqual(len(set(prompts)), num_requests, "Expected all prompts to be unique")

    async def test_pre_generate_no_deadlock_when_concurrency_below_workers(self) -> None:
        """concurrency_level < num_workers means some workers get 0 concurrency; they must
        still join the warmup barrier (regression guard against warmup deadlock)."""
        num_requests = 6
        prompts = await _run_and_collect_prompts(
            num_requests=num_requests, concurrency_level=2, num_workers=4, pre_generate=True
        )
        self.assertEqual(len(prompts), num_requests)
        self.assertEqual(len(set(prompts)), num_requests)

    async def test_pre_generate_deterministic_across_runs(self) -> None:
        """Same base_seed must produce byte-identical prompts run-to-run with pre_generate."""
        run1 = await _run_and_collect_prompts(
            num_requests=8, concurrency_level=4, num_workers=2, pre_generate=True, base_seed=1234
        )
        run2 = await _run_and_collect_prompts(
            num_requests=8, concurrency_level=4, num_workers=2, pre_generate=True, base_seed=1234
        )
        self.assertEqual(sorted(run1), sorted(run2))

    async def test_pre_generate_off_still_works(self) -> None:
        """Sanity: the same stage with pre_generate explicitly disabled still completes."""
        num_requests = 8
        prompts = await _run_and_collect_prompts(
            num_requests=num_requests, concurrency_level=4, num_workers=2, pre_generate=False
        )
        self.assertEqual(len(prompts), num_requests)
        self.assertEqual(len(set(prompts)), num_requests)

    async def test_pre_generate_correct_with_many_idle_workers(self) -> None:
        """Low concurrency on many workers: only min(workers, concurrency) channels are
        allocated, and the many idle workers must not break routing or correctness.

        Guards the /dev/shm-pressure fix (fewer queues) alongside routing correctness.
        """
        num_requests = 6
        prompts = await _run_and_collect_prompts(
            num_requests=num_requests, concurrency_level=2, num_workers=6, pre_generate=True
        )
        self.assertEqual(len(prompts), num_requests)
        self.assertEqual(len(set(prompts)), num_requests)


class TestPreGenerateResolution(unittest.TestCase):
    def _make_gen(self, load_type: LoadType, pre_generate: Any) -> LoadGenerator:
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=4),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=4),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer(), seed=1)
        if load_type == LoadType.CONCURRENT:
            stages: Any = [ConcurrentLoadStage(num_requests=4, concurrency_level=2)]
        else:
            from inference_perf.config import StandardLoadStage

            stages = [StandardLoadStage(rate=4, duration=1)]
        cfg = LoadConfig(type=load_type, num_workers=2, stages=stages, pre_generate=pre_generate)
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            return LoadGenerator(datagen, cfg)

    def test_defaults_true_for_concurrent(self) -> None:
        self.assertTrue(self._make_gen(LoadType.CONCURRENT, None).pre_generate)

    def test_defaults_false_for_constant(self) -> None:
        self.assertFalse(self._make_gen(LoadType.CONSTANT, None).pre_generate)

    def test_explicit_opt_in_for_constant(self) -> None:
        self.assertTrue(self._make_gen(LoadType.CONSTANT, True).pre_generate)

    def test_explicit_opt_out_for_concurrent(self) -> None:
        self.assertFalse(self._make_gen(LoadType.CONCURRENT, False).pre_generate)


class TestPreGenerationChannelCount(unittest.TestCase):
    """The number of per-worker queue channels must be bounded by the concurrency level so
    low-concurrency runs on many-core nodes don't allocate one queue (and its semaphores) per
    CPU (regression guard for /dev/shm exhaustion)."""

    def _make_gen(self, num_workers: int, concurrency_levels: List[int]) -> LoadGenerator:
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=64),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=64),
        )
        datagen = RandomDataGenerator(api_config, data_config, _DummyCustomTokenizer(), seed=1)
        stages = [ConcurrentLoadStage(num_requests=8, concurrency_level=c) for c in concurrency_levels]
        cfg = LoadConfig(type=LoadType.CONCURRENT, num_workers=num_workers, stages=stages)
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            return LoadGenerator(datagen, cfg)

    def test_channels_capped_by_concurrency(self) -> None:
        gen = self._make_gen(num_workers=64, concurrency_levels=[4])
        self.assertEqual(gen._pregeneration_channel_count(), 4)

    def test_channels_capped_by_workers(self) -> None:
        gen = self._make_gen(num_workers=2, concurrency_levels=[8])
        self.assertEqual(gen._pregeneration_channel_count(), 2)

    def test_channels_use_max_concurrency_across_stages(self) -> None:
        gen = self._make_gen(num_workers=64, concurrency_levels=[4, 16, 8])
        self.assertEqual(gen._pregeneration_channel_count(), 16)


if __name__ == "__main__":
    unittest.main()
