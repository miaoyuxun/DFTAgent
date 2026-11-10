---
name: descriptor-synthesis
description: Add one restricted GGA-level descriptor correction when ordinary component rewiring has stagnated.
---

## applicability
- stagnant_improvement
- self_interaction_error
- search_regression
- weak_generalization

## atomic_blueprint
- add_descriptor

## guardrails
- Add at most one descriptor correction.
- Use only approved descriptor templates: bounded_gradient, tanh_gradient, low_density_gradient, density_saturation.
- Keep absolute scale <= 0.05.
- Do not write Python code, gates, routers, conditionals, or dataset-specific rules.
- Preserve existing libxc exchange/correlation components unless the prompt explicitly asks for a paired edit.

## required_protocols
- descriptor_schema_check
- toy_density_derivative_check
- benchmark_regression_check

## execution_logic
1. Use descriptor synthesis only when simpler rewires or replacements are stagnating or sitting on a guard boundary.
2. Choose one physical descriptor family and explain what density regime it targets.
3. Provide initial parameters that are small enough to be a perturbative correction.
4. Let later rewire/refinement steps tune descriptor parameters; do not add a second descriptor.
