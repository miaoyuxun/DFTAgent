from xc.benchmark.result import BenchmarkResult, SubsetResult
from xc.idea.reward import guard_regression_penalty, target_guard_raw_reward, tree_reward_from_raw


def _result(name: str, values: dict[str, float]) -> BenchmarkResult:
    result = BenchmarkResult(xc_name=name)
    for subset, mae in values.items():
        result.subsets[subset] = SubsetResult(
            subset=subset,
            mae_kcal=mae,
            rms_kcal=mae,
            max_error_kcal=mae,
            n_reactions=4,
            n_failures=0,
            walltime_s=0.0,
        )
    result.compute_overall()
    return result


def test_guard_penalty_uses_subset_specific_absolute_tolerances():
    baseline = _result("baseline", {"BH76": 2.0, "G21EA": 6.0, "G21IP": 4.5})
    candidate = _result("candidate", {"BH76": 5.0, "G21EA": 7.0, "G21IP": 5.0})
    penalty, units = guard_regression_penalty(candidate, baseline)
    assert units == {"BH76": 1.0, "G21EA": 1.0, "G21IP": 1.0}
    assert abs(penalty - 0.35) < 1e-12


def test_sigmoid_tree_reward_keeps_invalid_below_valid_negative_scores():
    assert tree_reward_from_raw(float("-inf")) == 0.0
    assert tree_reward_from_raw(-1.0) > 0.0
    assert tree_reward_from_raw(0.5) == 0.5
    assert target_guard_raw_reward(search_ratio=1.0, guard_penalty=0.0) == 0.5
