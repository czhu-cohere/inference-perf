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
import pytest
import asyncio
import aiohttp
import json
from unittest.mock import AsyncMock, MagicMock
from inference_perf.client.modelserver.openai_client import (
    _strip_prompt_token_ids,
    _strip_tokenized_prompt,
    openAIModelServerClientSession,
)
from inference_perf.apis import ErrorResponseInfo, InferenceInfo, StreamedInferenceResponseInfo
from inference_perf.config import APIConfig, APIType


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.uri = "http://test-uri"
    client.api_config = MagicMock()
    client.api_config.headers = {}
    client.api_config.response_format = None
    client.tokenizer = MagicMock()
    client.metrics_collector = MagicMock()
    client.cert_path = None
    client.key_path = None
    return client


@pytest.fixture
def mock_data() -> MagicMock:
    data = MagicMock()
    data.get_route.return_value = "/test"
    data.process_failure = AsyncMock(return_value=InferenceInfo())
    data.process_response = AsyncMock(return_value=InferenceInfo())
    data.to_payload = AsyncMock(return_value={"mock": "data"})
    return data


@pytest.mark.asyncio
async def test_process_request_timeout(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a TimeoutError
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("Test timeout"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_process_request_client_error(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a ClientError
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Test client error"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "ClientError"


@pytest.mark.asyncio
async def test_process_request_general_exception(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a generic Exception
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=ValueError("Test general error"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "ValueError"


def test_strip_prompt_token_ids_from_json_and_sse() -> None:
    unary = json.dumps(
        {
            "choices": [
                {
                    "text": "hello",
                    "prompt_token_ids": [1, 2, 3],
                    "token_ids": [4, 5],
                }
            ]
        }
    )
    sanitized_unary = json.loads(_strip_prompt_token_ids(unary))
    assert "prompt_token_ids" not in sanitized_unary["choices"][0]
    assert sanitized_unary["choices"][0]["token_ids"] == [4, 5]

    stream = (
        'data: {"choices":[{"text":"hello","prompt_token_ids":[1,2,3],"token_ids":[4]}]}\n\n'
        'data: {"choices":[{"text":" world","token_ids":[5]}]}\n\n'
        "data: [DONE]\n\n"
    )
    sanitized_stream = _strip_prompt_token_ids(stream)
    assert "prompt_token_ids" not in sanitized_stream
    assert '"token_ids":[4]' in sanitized_stream
    assert '"token_ids":[5]' in sanitized_stream
    assert "data: [DONE]" in sanitized_stream


def test_strip_tokenized_prompt_from_saved_request() -> None:
    sanitized = json.loads(
        _strip_tokenized_prompt(json.dumps({"model": "test-model", "prompt": [1, 2, 3], "return_token_ids": True}))
    )
    assert "prompt" not in sanitized
    assert sanitized["return_token_ids"] is True

    text_prompt = json.dumps({"prompt": "keep this"})
    assert _strip_tokenized_prompt(text_prompt) == text_prompt


@pytest.mark.asyncio
async def test_completion_token_ids_are_requested_without_saving_prompt_ids(
    mock_client: MagicMock, mock_data: MagicMock
) -> None:
    mock_client.api_config = APIConfig(type=APIType.Completion, return_token_ids=True)
    mock_client.model_name = "test-model"
    mock_client.max_completion_tokens = 10
    mock_client.ignore_eos = False
    mock_client.api_key = None
    mock_data.to_payload = AsyncMock(return_value={"prompt": [1, 2, 3]})

    unary_response = json.dumps(
        {
            "choices": [
                {
                    "text": "world",
                    "prompt_token_ids": [1, 2, 3],
                    "token_ids": [4],
                }
            ]
        }
    )
    raw_stream = 'data: {"choices":[{"text":"world","prompt_token_ids":[1,2,3],"token_ids":[4]}]}\n\ndata: [DONE]\n\n'
    mock_data.process_response = AsyncMock(
        return_value=InferenceInfo(
            extra_info={"raw_response": raw_stream},
            response_info=StreamedInferenceResponseInfo(
                response_chunks=['{"choices":[{"text":"world","prompt_token_ids":[1,2,3],"token_ids":[4]}]}']
            ),
        )
    )

    response = MagicMock()
    response.status = 200
    response.text = AsyncMock(return_value=unary_response)
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    session = openAIModelServerClientSession(mock_client)
    await session.session.close()
    session.session = MagicMock()
    session.session.post.return_value = mock_post_ctx
    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    posted_payload = json.loads(session.session.post.call_args.kwargs["data"])
    assert posted_payload["return_token_ids"] is True
    assert posted_payload["prompt"] == [1, 2, 3]

    metric = mock_client.metrics_collector.record_metric.call_args.args[0]
    assert "prompt" not in json.loads(metric.request_data)
    assert metric.response_data is not None
    assert "prompt_token_ids" not in metric.response_data
    assert '"token_ids":[4]' in metric.response_data
    assert "prompt_token_ids" not in metric.info.extra_info["raw_response"]
    assert isinstance(metric.info.response_info, StreamedInferenceResponseInfo)
    assert "prompt_token_ids" not in metric.info.response_info.response_chunks[0]
