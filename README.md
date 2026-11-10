# DFTAgent

DFTAgent is an LLM-guided scientific search agent for discovering and refining
exchange-correlation (XC) functionals under a fixed evaluation budget.

Starting from an existing functional, the agent repeatedly:

1. selects promising parent functionals from the evaluated history;
2. uses an LLM to propose physically interpretable structural or parameter
   changes;
3. validates and deduplicates the proposed candidates;
4. evaluates candidates with a user-defined scientific objective;
5. optionally performs local evolutionary refinement around promising basins;
6. summarizes successful and unsuccessful search patterns for the next cycle.

The implementation supports component addition, replacement and removal,
mixing-weight changes, hybrid parameters, role-level scaling, and exposed Libxc
parameters. Candidate legality and the scientific objective are kept separate
from the agent itself through a common `SearchTask` interface.

> The accompanying paper is coming soon.

## Overview

```text
initial XC functional
        |
        v
successful parent pool
        |
        v
LLM proposals
  structural edits
  parameter rewires
        |
        v
validation and deduplication
        |
        v
scientific evaluation and reward
        |
        +---- optional local EA refinement
        |
        v
history summary and next search cycle
```

## Repository layout

- `src/xc/idea/basin_agent.py`: main basin-agent search loop.
- `src/xc/idea/agent.py`: LLM proposal generation and repair.
- `src/xc/idea/actions.py`: candidate action vocabulary and compiler.
- `src/xc/idea/evolutionary.py`: optional local evolutionary refinement.
- `src/xc/search_tasks/`: scientific objective and reward adapters.
- `src/xc/benchmark/`: molecular benchmark execution support.
- `examples/custom_reward_task.py`: minimal custom-reward template.
- `docs/CUSTOM_REWARD.md`: guide for connecting a new objective.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For PySCF evaluations:

```bash
python -m pip install -e ".[chemistry,test]"
```

Benchmark datasets are not redistributed. Place an authorized GMTKN55 copy
under `benchmark/GMTKN55/` when using the molecular examples.

## LLM configuration

```bash
cp config/llm.example.yaml config/llm.yaml
export OPENAI_API_KEY="..."
export OPENAI_MODEL="..."
```

Credentials, result directories, checkpoints and logs are ignored by Git.

## Running the agent

Validate the example configuration without submitting calculations:

```bash
python scripts/run_mean_mae_basin_agent.py --validate-config
```

After adapting the evaluator settings to the local machine, launch a run with:

```bash
python scripts/run_mean_mae_basin_agent.py \
  --config config/mean_mae_basin.yaml \
  --output-dir results/basin_seed42 \
  --seed 42
```

The example configuration is intentionally small. Search budget, component
pool, proposal batch size, local-EA settings and evaluator resources are all
configured in YAML.

## Custom reward

The agent maximizes the scalar `SearchTaskEvaluation.reward`. Implement a new
scientific objective as a task adapter rather than modifying the search loop.

```python
SearchTaskEvaluation(
    task_name="my_task",
    valid=True,
    reward=1.25,
    objective={"target_error": 0.31, "penalty": 0.10},
    proposal_feedback={"walltime_s": 48.0, "scf_cycles": 9},
)
```

Values defining the score belong in `objective`. Diagnostics visible to the LLM
but excluded from ranking belong in `proposal_feedback`. Invalid calculations
should return `valid=False`, and larger reward must always mean better.

See [docs/CUSTOM_REWARD.md](docs/CUSTOM_REWARD.md) and
[examples/custom_reward_task.py](examples/custom_reward_task.py) for the full
interface.

## Baseline

The repository also includes a Diverse Bayesian Optimization implementation for
budget-matched baseline experiments. It shares the same functional
representation, component pool, candidate validation, evaluator and reward
interface as the agent.

```bash
python scripts/run_mean_mae_diverse_bo.py --validate-config
```

`examples/root_evaluation.example.json` is synthetic and must be replaced with
an evaluation of the actual root functional before running a real baseline.