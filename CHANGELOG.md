# Changelog

## Unreleased

### Added
- Corner-aware optimization (`sizeagent/robust.py`, `scripts/robust_study.py`) and a dashboard section for it.
- Statistics module and `scripts/stats.py`: bootstrap/Wilson CIs, Mann-Whitney tests, censored sims-to-spec.
- Benchmark extended to 20 seeds per algorithm.
- `tests/fake_sim.py`: analytic stand-in for ngspice; optimizer, surrogate and PVT logic now has fast tests.
- Independent SPICE verification of the agent's submitted design (`submitted_metrics`, `submitted_meets_spec`).
- Actionable `SetupError` for missing ngspice / PDK; `SIZEAGENT_PDK` override.
- mypy (0 errors) enforced in CI.

### Fixed
- Agent crashed with an uncaught `BudgetExceeded` when `refine` was called after the budget was spent.
- Surrogate-assisted GA crashed (`IndexError`) when every warm-up simulation failed.
- `RobustEvaluator` silently dropped already-charged simulations when the budget ended mid-verification; they are now traced as `unverified`.
- Agent run lost all results on an API/quota error; it now stops and writes the partial trace.
- Groq requests were rejected (HTTP 403, error 1010) for the default Python User-Agent; stale Gemini/Groq default models replaced.
- Groq free-tier 8k tokens/min limit: older tool results are compacted in the agent prompt.
- TPE no longer logs spurious failed trials when the budget ends; `Evaluator` best-cost tracking is O(1).
