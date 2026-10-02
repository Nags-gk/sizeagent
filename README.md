# SizeAgent

**This repo treats transistor sizing for a SKY130 two-stage op-amp as a combinatorial optimization problem, solved with search algorithms, a PyTorch surrogate model, and a tool-calling LLM agent. Every result is checked in ngspice.**

Live dashboard: **https://nags-gk.github.io/sizeagent/** (benchmark curves, agent traces, surrogate parity plots, an in-browser surrogate predictor, and PVT/Monte Carlo results)

![benchmark](docs/img/benchmark.png)

## Why this is a combinatorial problem

SkyWater SKY130 MOSFET models are *binned*: only characterized per-finger (W, L) pairs are valid. Each device group picks one of those geometries plus an integer multiplier. C<sub>c</sub>, R<sub>z</sub> and I<sub>bias</sub> come from fixed grids. Put together, that is **~3.6 × 10¹⁵ discrete sizings**. A design has to meet all of these targets at once:

| Spec | Target |
|---|---|
| DC gain | ≥ 60 dB |
| Unity-gain bandwidth (C<sub>L</sub> = 2 pF) | ≥ 20 MHz |
| Phase margin | ≥ 60° |
| Power (VDD = 1.8 V) | ≤ 250 µW |
| Every transistor in saturation | margin ≥ 50 mV |
| Systematic offset \|V<sub>out</sub> − 0.9 V\| | ≤ 50 mV |

## What's here

| Component | What it does |
|---|---|
| `scripts/setup_pdk.sh`, `build_models.py` | Fetches only the SKY130 1.8 V core FET models (tt/ff/ss/fs/sf), then patches them for ngspice the same way open_pdks does: subckt parameters, the `mult` → `m` pass-through, and per-instance Monte Carlo mismatch |
| `sizeagent/spice.py` | ngspice testbench runner. The open-loop AC sweep uses a 1 GH DC-feedback inductor to set the bias point, then measures gain, UGBW, phase margin, power, offset and each device's operating point (gm, gds, V<sub>dsat</sub>, …) |
| `sizeagent/optimizers/search.py` | Random search, genetic algorithm, simulated annealing, and Bayesian optimization (Optuna TPE) over the discrete genome |
| `sizeagent/surrogate.py` | MLP ensemble (PyTorch) that predicts SPICE metrics, plus a **surrogate-assisted GA** that screens 400 offspring per generation and simulates only the 10 most promising |
| `sizeagent/agent/` | Tool-calling LLM agent with four tools: `simulate`, `refine` (seeded local search), `verify_pvt`, `submit`. It works with any OpenAI-compatible endpoint: Gemini (free tier), Groq, or local Ollama |
| `sizeagent/pvt.py` | 5 process corners × 3 temperatures, plus mismatch Monte Carlo |

## Results

Each strategy gets the same budget of **300 unique SPICE simulations**, run over 6 seeds. Repeated designs are cached and not counted.

| Strategy | Met spec (of 6) | Median sims to spec* | Median best power | Lowest power |
|---|---|---|---|---|
| Surrogate-assisted GA | 6/6 | 65.0 | 63.7 µW | 57.4 µW |
| Genetic algorithm | 6/6 | 91.0 | 93.8 µW | 64.6 µW |
| Simulated annealing | 5/6 | 203.5 | 99.2 µW | 73.4 µW |
| Random search | 2/6 | >300 | 155.1 µW | 142.6 µW |
| Bayesian opt. (TPE) | 3/6 | >300 | 139.4 µW | 97.7 µW |

\*Median over all seeds, counting a failed run as >300.

The surrogate-assisted GA met spec in every seed. It needed a median of 65 simulations, versus 91 for the plain GA, and found the lowest-power designs (median 63.7 µW). Uniform random search met spec in only 2 of 6 seeds.

Surrogate vs. SPICE on held-out designs:

![parity](docs/img/surrogate_parity.png)

| Metric | Held-out R² | MAE |
|---|---|---|
| DC gain (dB) | 0.811 | 6.58 |
| log10 UGBW (MHz) | 0.755 | 0.402 |
| Phase margin (deg) | 0.686 | 15.2 |
| log10 power (uW) | 0.973 | 0.0314 |
| Min sat. margin (V) | 0.858 | 0.0427 |
| Vout offset from Vcm (V) | 0.732 | 0.0674 |

1600 training and 400 test designs. Accuracy is moderate: phase margin is hardest (R² 0.69). That is why the surrogate only *ranks* candidates and every reported number comes from SPICE.

**Robustness check.** The best design from each method was re-simulated at 5 corners × 3 temperatures, plus 100 mismatch Monte Carlo runs:

| Best design from | Corners passing | Offset σ (MC) |
|---|---|---|
| Genetic algorithm | 8/15 | 1.36 mV |
| Random search | 7/15 | 2.71 mV |
| Simulated annealing | 2/15 | 11.64 mV |
| Surrogate-assisted GA | 10/15 | 2.48 mV |
| Bayesian opt. (TPE) | 10/15 | 3.83 mV |

None of the nominal-only optima pass all 15 PVT corners. Optimizing at tt/27 °C pushes each design to the edge of the spec, which is exactly why industrial flows optimize worst-case over corners. Corner-aware optimization is the next step for this repo.

### Honest caveats
- Only the transistors use foundry models. C<sub>c</sub>, R<sub>z</sub> and C<sub>L</sub> are ideal elements, and there are no layout parasitics.
- Optimization runs at the tt corner, 27 °C. The PVT table shows how much margin each nominal optimum keeps; robust (worst-case) optimization isn't implemented yet.
- Phase margin and UGBW come from an open-loop AC analysis. There's no transient, slew-rate or noise analysis yet.
- Benchmarks are 6 seeds per method. Expect the spread you see in the IQR band.

## Run it

```bash
sudo apt-get install ngspice              # ngspice 42 tested
./scripts/setup_pdk.sh                    # ~10 MB of SKY130 SPICE models
pip install -e .[dev] && pytest -q

python scripts/run_benchmark.py --seeds 6 --budget 300
python scripts/surrogate_study.py && python scripts/pvt_study.py
python scripts/make_report.py             # -> docs/data/results.json, docs/img/*.png

# LLM agent (free Gemini key from https://aistudio.google.com/apikey)
export GEMINI_API_KEY=...
python -m sizeagent.agent --provider gemini --budget 150 --out results/agent_gemini_1.json
# or fully local:  ollama pull qwen2.5:7b && python -m sizeagent.agent --provider ollama
```

## License

Apache-2.0. The SKY130 models are fetched at setup time from [google/skywater-pdk-libs-sky130_fd_pr](https://github.com/google/skywater-pdk-libs-sky130_fd_pr) (Apache-2.0) and are not redistributed here.
