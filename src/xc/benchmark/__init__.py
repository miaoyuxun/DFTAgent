"""XC functional benchmark package."""

from .result import BenchmarkResult, SubsetResult
from .vc_manifest_runner import VCManifestBenchmarkRunner, VCManifestRunnerConfig

try:  # Keep light-weight result models importable in environments without PySCF.
    from .runner import BenchmarkRunner, SEARCH_SUBSETS, VALIDATION_SUBSETS
except ImportError:  # pragma: no cover - depends on optional runtime dependency
    BenchmarkRunner = None
    SEARCH_SUBSETS = []
    VALIDATION_SUBSETS = []

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "SubsetResult",
    "VCManifestBenchmarkRunner",
    "VCManifestRunnerConfig",
    "SEARCH_SUBSETS",
    "VALIDATION_SUBSETS",
]
