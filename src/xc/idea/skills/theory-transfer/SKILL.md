---
name: theory-transfer
description: Transfer one better-grounded XC mechanism into the current functional when the evidence leaves an unresolved explanatory gap.
disabled: true
---

This skill has been merged into `component-replacement`.

Rationale: with the current action space, theory-transfer and component-replacement both compile to `replace`/`enforce`, but the separate skill caused repeated remove-vs-replace schema failures. The scientific requirements below are retained here as history and are now enforced by `component-replacement`.

## applicability
- unresolved
- open
- inconclusive

## atomic_blueprint
- replace
- enforce

## guardrails
- Transfer one principle, not a bundle of unrelated mechanisms.
- Keep the inserted mechanism physically interpretable.
- Add the smallest validation that can falsify the transfer.

## required_protocols
- ablation
- stress

## execution_logic
1. Name the unresolved question.
2. Choose one theory-backed mechanism that addresses it.
3. Replace the closest weak slot and state the falsification test.
