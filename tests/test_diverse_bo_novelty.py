from __future__ import annotations

from xc.idea.diverse_bo_basin import _passes_novelty_threshold


def test_zero_threshold_rejects_exact_duplicate_but_allows_small_move() -> None:
    assert not _passes_novelty_threshold(0.0, 0.0)
    assert not _passes_novelty_threshold(1e-13, 0.0)
    assert _passes_novelty_threshold(1e-8, 0.0)
    assert _passes_novelty_threshold(0.05, 0.0)


def test_positive_threshold_preserves_inclusive_boundary() -> None:
    assert not _passes_novelty_threshold(0.099, 0.1)
    assert _passes_novelty_threshold(0.1, 0.1)
    assert _passes_novelty_threshold(0.2, 0.1)
    assert _passes_novelty_threshold(None, 0.1)
