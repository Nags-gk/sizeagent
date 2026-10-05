# Roadmap

Status as of the audit: core pipeline, 20-seed benchmark, corner-aware study and CI are in place.
Items are ordered by impact on credibility first, breadth second.

## Now (blocked only on API quota / a few hours of compute)
1. **LLM agent benchmark.** Run Groq `openai/gpt-oss-120b`, Gemini and a local Ollama model for >=10 seeds each at the
   same 150-300 simulation budget; add them to the benchmark table and dashboard traces. Today only 1 Gemini run exists.
   Report `submitted_meets_spec` (independently verified), not the agent's own claim.
2. **Reconcile concurrent branches.** `feature/roadmap-improvements` and `fix/audit-issues` both carry work; merge into
   `main` through a reviewed PR and delete the stale branch.
3. ~~**Full-grid corner-aware runs, 20 seeds.**~~ Done: surrogate GA 20/20 robust-feasible on all 15 grid points
   (median 149 sims); see README. Remaining gap: supply (VDD) variation and mismatch are not in the search.

## Next (1-2 weeks)
4. ~~**Better surrogate.**~~ Done (negative result): GP, forest and boosting do not beat the MLP ensemble; conformal
   calibration fixes coverage (0.78 -> 0.89) but does not change sims-to-spec. Still open: a dedicated phase-margin model
   (R^2 ~0.7) and an active-learning ablation on dataset size.
5. **Worst-case surrogate.** Predict worst-over-corners metrics directly so screening is corner-aware and the 7x
   simulation overhead of robust search shrinks.
6. **Multi-objective search.** NSGA-II / Pareto front for power vs UGBW vs phase margin instead of one scalar cost.
7. **Reproducibility.** Stamp ngspice version, PDK commit and git SHA into every results file; add a lock file;
   publish the Docker image from CI.

## Later (portfolio depth)
8. **Circuit realism.** Transient/slew, noise, PSRR/CMRR, real passives and layout parasitic estimates.
9. **Second topology** (folded cascode or 5T OTA) and a second spec set to show the method generalizes.
10. **Agent upgrades.** Native Anthropic provider, structured-output tool calls, retry/repair of malformed calls,
    agent-in-the-loop with the surrogate as a free tool.
11. **Packaging.** Make `torch` an optional extra, remove `sys.path` hacks from `scripts/`, console-script entry
    points, PyPI release, coverage badge.
12. **Write-up.** Short blog/paper-style report with the benchmark methodology, a demo GIF in the README, and a
    one-page results summary for recruiters.
