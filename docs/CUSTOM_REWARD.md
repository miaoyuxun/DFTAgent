# Custom reward and scientific task

The search algorithms maximize one scalar `SearchTaskEvaluation.reward`. A new
scientific objective should therefore be implemented once as a task adapter,
not by editing Basin Agent and BO separately.

## Required contract

Implement the `SearchTask` protocol from `src/xc/search_tasks/base.py`:

```python
class MyTask:
    name = "my_task"
    use_benchmark_component_priors = False

    def validate_functional(self, xc): ...
    def evaluate_many(self, xcs, *, labels, stage): ...
    def apply_to_candidate(self, candidate, evaluation): ...
    def prompt_instructions(self): ...
    def relative_feedback(self, child, parent): ...
```

`examples/custom_reward_task.py` supplies a reusable callback implementation.

## Minimal implementation path

1. Copy `examples/custom_reward_task.py` into the project task module.
2. Replace `evaluate_one()` with the scientific calculation.
3. Return the score-defining fields in `objective` and diagnostics in
   `proposal_feedback`.
4. Construct one task instance and pass that same instance to both search
   methods.
5. Evaluate or load the root result using exactly the same protocol.
6. Run a small budget and verify reward recomputation from saved artifacts
   before launching the full comparison.

## Reward conventions

- Larger reward must always be better.
- Invalid, missing, NaN, or scientifically unusable evaluations must return
  `valid=False`; the dataclass converts their reward to negative infinity.
- Put every quantity that mathematically defines reward under `objective`.
- Put non-ranking observations under `proposal_feedback`.
- Do not silently use runtime, convergence cycles, or diagnostics in ranking.
- Evaluate the root through the same code path, or load a numerically equivalent
  cached root result and document that choice.

For a minimization target, negate it:

```python
reward = -mean_absolute_error
```

For a metric that should be maximized, use it directly. Do not invert the sign
again inside an algorithm-specific launcher.

For target improvement with guards:

```python
target_gain = 1.0 - target_error / root_target_error
guard_excess = max(0.0, guard_error - free_tolerance)
reward = root_reward + target_weight * target_gain - guard_weight * guard_excess
```

Free tolerances, hard failures, and penalty weights are part of the scientific
method and must be reported.

## Connecting Basin Agent

`LLMBasinAgentSearch.search()` accepts `task` and `root_evaluation` directly:

```python
result = basin.search(
    root_xc=root_xc,
    idea_agent=idea_agent,
    output_dir=output_dir,
    task=my_task,
    root_evaluation=root_evaluation,
)
```

The text returned by `prompt_instructions()` is included in the proposal context.
It should state the reward formula, hard failures, editable variables, and the
meaning of diagnostics without suggesting a preferred component.

## Connecting BO

The task-aware reference adapter is
`TaskDiverseBOBasinSearch` in
`scripts/run_sie_crystal_guard_diverse_bo.py`. Despite the historical filename,
it consumes the same task methods through duck typing. Construct it with the same
root functional, task, root evaluation, component pool, and budget used by Basin
Agent.

For a publication comparison, verify that BO and Basin Agent share:

1. the exact candidate validator;
2. the same single-step or multi-action policy;
3. identical parameter bounds and mixing normalization;
4. identical invalid-result and timeout handling;
5. equal evaluated-candidate budgets.

## Diagnostics versus reward

The separation below is intentional:

```python
SearchTaskEvaluation(
    reward=-2.4,
    objective={
        "mean_mae": 2.4,
        "penalty": 0.0,
    },
    proposal_feedback={
        "scf_cycles": 12,
        "walltime_s": 84.2,
        "physical_warning": "fractional occupations",
    },
)
```

If fractional occupations should invalidate the candidate, enforce that in
`valid`. If they are merely diagnostic, do not let them alter reward indirectly.
