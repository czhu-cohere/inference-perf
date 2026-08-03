import numpy as np

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, DataGenType, DistributionType
from inference_perf.datagen.base import LazyLoadDataMixin
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from typing import Any


class DummyTokenizer:
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def encode(self, text: str) -> list[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return [4, 5, 6]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)


class DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return DummyTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def test_random_datagen_yields_string() -> None:
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = RandomDataGenerator(api_config, data_config, tokenizer)

    # RandomDataGenerator uses LazyLoadDataMixin, so get_data() yields LazyLoadInferenceAPIData
    data_gen = generator.get_data()
    lazy_data = next(data_gen)
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)

    # Load the real data
    real_data = generator.load_lazy_data(lazy_data)
    assert isinstance(real_data, CompletionAPIData)

    # Verify prompt is str
    assert isinstance(real_data.prompt, str)
    assert len(real_data.prompt) > 0


def test_worker_rng_uniqueness_per_worker() -> None:
    """Workers must produce different content even when all input lengths are identical (std_dev=0)."""
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=10, mean=10, std_dev=0.0, total_count=5),
        output_distribution=Distribution(min=5, max=5, mean=5, std_dev=0.0, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    # Simulate what Worker.run() does: reseed the datagen's rng with a worker-specific seed.
    # Worker 0 and Worker 1 use (base_seed + id) % 2**32, so they differ.
    base_seed = 42

    generator_w0 = RandomDataGenerator(api_config, data_config, tokenizer)
    generator_w0.rng = np.random.default_rng((base_seed + 0) % 2**32)

    generator_w1 = RandomDataGenerator(api_config, data_config, tokenizer)
    generator_w1.rng = np.random.default_rng((base_seed + 1) % 2**32)

    lazy = LazyLoadInferenceAPIData(data_index=0)
    result_w0 = generator_w0.load_lazy_data(lazy)
    result_w1 = generator_w1.load_lazy_data(lazy)

    assert isinstance(result_w0, CompletionAPIData)
    assert isinstance(result_w1, CompletionAPIData)
    assert result_w0.prompt != result_w1.prompt, "Workers with different seeds must produce different prompts"


def test_random_datagen_excludes_special_tokens() -> None:
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = RandomDataGenerator(api_config, data_config, tokenizer)
    data_gen = generator.get_data()
    lazy_data = next(data_gen)
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)
    real_data = generator.load_lazy_data(lazy_data)

    assert isinstance(real_data, CompletionAPIData)
    # Verify no special tokens in prompt by encoding it back
    encoded_ids = tokenizer.get_tokenizer().encode(real_data.prompt)
    for token in encoded_ids:
        assert token not in [1, 2, 3]


def _make_random_datagen(seed: int, total_count: int = 16) -> RandomDataGenerator:
    api_config = APIConfig(type=APIType.Completion, streaming=False)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=10, mean=10, std_dev=0.0, total_count=total_count),
        output_distribution=Distribution(min=5, max=5, mean=5, std_dev=0.0, total_count=total_count),
    )
    generator = RandomDataGenerator(api_config, data_config, DummyCustomTokenizer())
    generator.rng = np.random.default_rng(seed)
    return generator


def test_pregenerate_matches_lazy_path() -> None:
    """Pre-generated prompts must be byte-identical to the lazy path for the same seed.

    A worker pre-generates its assigned data_indices in ascending order, which advances the
    RNG exactly as the lazy dispatch path would when pulling those same indices in order.
    """
    seed = 123
    # Worker 0 of a 2-worker stage with num_requests=8 handles indices 0, 2, 4, 6.
    assigned = [0, 2, 4, 6]

    lazy_gen = _make_random_datagen(seed)
    expected = [lazy_gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=i)).prompt for i in assigned]  # type: ignore[attr-defined]

    pre_gen = _make_random_datagen(seed)
    pre_gen.pregenerate(assigned)
    got = [pre_gen.take_pregenerated(i).prompt for i in assigned]  # type: ignore[union-attr]

    assert got == expected


def test_get_request_uses_cache_without_regenerating() -> None:
    """get_request must return the cached payload and NOT call load_lazy_data on a cache hit."""
    seed = 7
    generator = _make_random_datagen(seed)
    generator.pregenerate([0, 1, 2])

    calls: list[int] = []
    original_load = generator.load_lazy_data

    def counting_load(data: LazyLoadInferenceAPIData) -> Any:
        calls.append(data.data_index)
        return original_load(data)

    generator.load_lazy_data = counting_load  # type: ignore[method-assign]

    # Cache hits: no generation on the dispatch path.
    for i in [0, 1, 2]:
        result = LazyLoadDataMixin.get_request(generator, LazyLoadInferenceAPIData(data_index=i))
        assert isinstance(result, CompletionAPIData)
    assert calls == [], "load_lazy_data must not be called for cached indices during dispatch"

    # Cache miss falls back to lazy generation.
    LazyLoadDataMixin.get_request(generator, LazyLoadInferenceAPIData(data_index=3))
    assert calls == [3]


def test_pregenerate_consumes_cache_on_take() -> None:
    """Taking a cached payload pops it so memory is released as requests dispatch."""
    generator = _make_random_datagen(seed=1)
    generator.pregenerate([0, 1])

    assert generator.take_pregenerated(0) is not None
    # Second take for the same index misses (already consumed).
    assert generator.take_pregenerated(0) is None
    assert generator.take_pregenerated(1) is not None


def test_pregenerate_large_batch_memory_sanity() -> None:
    """Pre-generating a large batch must complete and cache exactly the requested indices."""
    num_requests = 500
    generator = _make_random_datagen(seed=99, total_count=num_requests)
    indices = list(range(num_requests))
    generator.pregenerate(indices)

    # All requested indices are cached and unique.
    prompts = [generator.take_pregenerated(i) for i in indices]
    assert all(p is not None for p in prompts)
    assert len({p.prompt for p in prompts}) == num_requests  # type: ignore[union-attr]


def test_random_datagen_distribution_types() -> None:
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(
            type=DistributionType.FIXED,
            min=10,
            max=20,
            mean=15,
            std_dev=2,
            total_count=5,
        ),
        output_distribution=Distribution(
            type=DistributionType.FIXED,
            min=5,
            max=10,
            mean=7,
            std_dev=1,
            total_count=5,
        ),
    )
    tokenizer = DummyCustomTokenizer()

    generator = RandomDataGenerator(api_config, data_config, tokenizer)

    # With FIXED type, all generated lengths must be exactly equal to the mean
    assert len(generator.input_lengths) == 5
    for length in generator.input_lengths:
        assert length == 15

    assert len(generator.output_lengths) == 5
    for length in generator.output_lengths:
        assert length == 7
