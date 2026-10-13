from src.data.features import (
    activity_fingerprint,
    fit_vectorizers,
    meets_minimum_activity,
    stylometric_vector,
)
from src.schemas import Alias, Platform


def _alias(alias_id: str, messages: list[str], hours: list[int]) -> Alias:
    return Alias(alias_id=alias_id, platform=Platform.OPEN_WEB, source="test", messages=messages, post_hours_utc=hours)


def test_meets_minimum_activity_respects_both_floors():
    short = _alias("a", ["word " * 10], [1] * 5)
    assert not meets_minimum_activity(short)

    long_text_few_timestamps = _alias("b", ["word " * 2000], [1] * 5)
    assert not meets_minimum_activity(long_text_few_timestamps)

    enough = _alias("c", ["word " * 2000], list(range(35)))
    assert meets_minimum_activity(enough)


def test_activity_fingerprint_sums_to_one():
    alias = _alias("x", ["hi"] * 10, [9, 9, 9, 10, 10, 14, 20, 20, 20, 20])
    profile = activity_fingerprint(alias)
    assert profile.shape == (24,)
    assert abs(profile.sum() - 1.0) < 1e-9
    # hour 20 has 4 of 10 posts
    assert abs(profile[20] - 0.4) < 1e-9


def test_activity_fingerprint_empty_is_all_zero_not_nan():
    alias = _alias("empty", [], [])
    profile = activity_fingerprint(alias)
    assert profile.sum() == 0


def test_identical_text_has_similarity_one():
    text = "the quick brown fox jumps over the lazy dog " * 50
    a = _alias("a", [text], list(range(35)))
    b = _alias("b", [text], list(range(35)))
    vecs = fit_vectorizers([a.text, b.text])
    va, vb = stylometric_vector(a, vecs), stylometric_vector(b, vecs)
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(va, vb)[0, 0]
    assert sim > 0.999


def test_unrelated_text_has_low_similarity():
    a = _alias("a", ["apple banana cherry date " * 50], list(range(35)))
    b = _alias("b", ["xylophone zeppelin quasar nebula " * 50], list(range(35)))
    vecs = fit_vectorizers([a.text, b.text])
    va, vb = stylometric_vector(a, vecs), stylometric_vector(b, vecs)
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(va, vb)[0, 0]
    # Char n-grams alone give unrelated text a non-trivial baseline overlap
    # (shared spaces/common letter pairs), so "low" here means "clearly
    # below the identical-text case above", not "near zero".
    assert sim < 0.5
