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
from typing import AsyncGenerator, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType


@pytest.mark.asyncio
async def test_completion_api_data() -> None:
    data = CompletionAPIData(prompt="Hello, world!")
    assert data.get_api_type() == APIType.Completion
    assert data.prompt == "Hello, world!"
    assert await data.to_payload("test-model", 100, False, True) == {
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "ignore_eos": False,
        "skip_special_tokens": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.asyncio
async def test_streaming_completion_uses_returned_token_ids() -> None:
    data = CompletionAPIData(prompt="Hello")
    response = MagicMock()
    response.content = MagicMock()

    async def iter_any() -> AsyncGenerator[bytes, None]:
        yield (
            b'data: {"choices":[{"text":"A","prompt_token_ids":[10,11],"token_ids":[20]}]}\n\n'
            b'data: {"choices":[{"text":"B","token_ids":[21]}]}\n\n'
            b"data: [DONE]\n\n"
        )

    response.content.iter_any = iter_any
    tokenizer = MagicMock()
    tokenizer.count_tokens.return_value = 999

    info = await data.process_response(
        cast(ClientResponse, response),
        APIConfig(type=APIType.Completion, streaming=True, return_token_ids=True),
        tokenizer,
    )

    assert data.prompt_token_ids == [10, 11]
    assert data.model_response_token_ids == [20, 21]
    assert data.model_response == "AB"
    assert info.input_tokens == 2
    assert info.response_info is not None
    assert info.response_info.output_tokens == 2
