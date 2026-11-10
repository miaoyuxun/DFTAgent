---
name: component-simplification
description: Remove or simplify components that ablation evidence marks as redundant or harmful.
---

## applicability
- harmful
- redundant
- inconclusive

## atomic_blueprint
- remove
- rewire

## guardrails
- Touch exactly one removable component.
- Do not remove the final exchange or correlation path.
- Keep the redistribution explicit.

## required_protocols
- ablation
- regression

## execution_logic
1. Pick the least justified component from findings.
2. Remove it or simplify its contribution.
3. Redistribute only the necessary weight.
