"""Tool-calling LLM agent loop for op-amp sizing.

The agent proposes sizings from analog design reasoning, reads SPICE results
and operating points, and can hand off to a combinatorial local search
(`refine`) or check PVT corners before it submits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..circuit import CL_PF, VDD
from ..specs import Evaluator, Spec, feasible
from ..spice import simulate
from .providers import ChatClient, FallbackClient
from .tools import TOOL_SPECS, AgentTools, design_from_args

SYSTEM = """You are an analog IC design agent sizing a two-stage Miller-compensated CMOS op-amp \
in the SkyWater SKY130 process (VDD = {vdd} V, load CL = {cl} pF, input common mode 0.9 V).

Topology (device groups share one W/L per finger; m* are integer multipliers):
- M1/M2 NMOS input pair (group inp, multiplier m1); M3/M4 PMOS mirror load (group load, m3)
- M8 NMOS bias diode fed by Ibias (group bias, m=1); M5 tail = Ibias*m5; M7 2nd-stage sink = Ibias*m7 (group bias)
- M6 PMOS common-source 2nd stage (group cs, m6); Miller cap Cc in series with nulling resistor Rz.

Target spec: {spec}.

Useful first-order relations: A0 = gm1/(gds2+gds4) * gm6/(gds6+gds7); UGBW ~ gm1/(2*pi*Cc);
non-dominant pole ~ gm6/CL, so phase margin improves with larger gm6/gm1 * CL/Cc ratio handled \
by keeping gm6 >~ 3-10x gm1 and Cc >~ 0.2 CL; Rz ~ 1/gm6 cancels the RHP zero. Longer L raises \
gain (lower gds) but costs speed; gm/Id of 10-20 is moderate inversion. Low systematic offset needs \
the M6 current to match the M7 sink: (W/L*m)_6 / (W/L*m)_4 ~ 2 * (m7/m5).
Power = VDD * Ibias * (1 + m5 + m7).

Rules: you have a budget of {budget} simulations. Think briefly before each tool call. Use \
`simulate` to test hypotheses and read the operating point; use `refine` when you are close \
(few violations) to let local search finish the job. Once a design meets spec, optionally run \
`verify_pvt` and then `submit`. Always submit before the budget runs out."""


def compact(messages: list[dict], keep: int = 4) -> list[dict]:
    """Shrink older tool results (drop operating points and design echoes) so the
    prompt stays small on token-limited tiers. Call/response pairing is untouched."""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = set(tool_idx[:-keep]) if len(tool_idx) > keep else set()
    out = []
    for i, m in enumerate(messages):
        if i in old:
            try:
                r = json.loads(m["content"])
                brief = {k: r[k] for k in ("metrics", "violations", "meets_spec", "pass_count", "error",
                                           "failing_corners") if k in r}
                if "design" in r:
                    brief["design"] = r["design"]
                m = {**m, "content": json.dumps(brief)}
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(m)
    return out


def verify_final(design, ev: Evaluator) -> dict:
    """Re-simulate the final design in SPICE outside the search budget, so the reported
    result never rests on the agent's own claim."""
    if design is None:
        return {"submitted_metrics": None, "submitted_meets_spec": False}
    c, m, _ = ev.cache.get(design.key()) or (None, None, None)
    if m is None:
        r = simulate(design, ev.corner, ev.temp)
        m = r.metrics() if r.ok else None
    return {"submitted_metrics": m, "submitted_meets_spec": feasible(m, ev.spec)}


def run_agent(provider: str = "gemini", model: str | None = None, budget: int = 150,
              max_turns: int = 40, spec: Spec | None = None, out: str | None = None,
              client=None, verbose: bool = True) -> dict:
    spec = spec or Spec()
    ev = Evaluator(spec=spec, budget=budget)
    tools = AgentTools(ev)
    client = client or (FallbackClient(model=model) if provider == "auto" else ChatClient(provider, model))
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM.format(vdd=VDD, cl=CL_PF, spec=spec.describe(), budget=budget)},
                {"role": "user", "content": "Size the op-amp to meet the spec with as few simulations as possible."}]
    log, api_error = [], None
    t0 = time.time()
    for turn in range(max_turns):
        try:
            msg = client.chat(compact(messages), TOOL_SPECS)
        except RuntimeError as e:      # quota or network failure: keep what we have
            api_error = str(e)[:300]
            if verbose:
                print(f"[{turn}] stopping: {api_error}")
            break
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if msg.get("content") and verbose:
            print(f"[{turn}] agent: {msg['content'][:400]}")
        if not calls:
            messages.append({"role": "user", "content": "Continue: call a tool (simulate, refine, verify_pvt or submit)."})
            continue
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            sims_before = ev.sims
            result = tools.call(name, args)
            if verbose:
                brief = {k: result[k] for k in ("metrics", "violations", "meets_spec", "sims_used", "error",
                                                "pass_count") if k in result}
                print(f"[{turn}] {name} -> {json.dumps(brief)[:300]}")
            log.append({"turn": turn, "thought": msg.get("content") or "", "tool": name, "args": args,
                        "result": result, "sims_before": sims_before, "sims_after": ev.sims})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name), "name": name,
                             "content": json.dumps(result)})
        if tools.submitted is not None:
            break
    best = ev.best()
    final = tools.submitted.to_dict() if tools.submitted else (best["design"] if best else None)
    verified = verify_final(tools.submitted if tools.submitted else (design_from_args(best["design"]) if best else None),
                            ev)
    summary = {"provider": provider, "model": getattr(client, "model", None), "budget": budget,
               "sims_used": ev.sims, "first_feasible_sim": ev.first_feasible(),
               "best": best, "submitted": final, **verified, "rationale": tools.rationale,
               "turns": len({x["turn"] for x in log}), "api_error": api_error, "wall_s": round(time.time() - t0, 1),
               "log": log, "trace": ev.trace}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM agent for SKY130 op-amp sizing")
    ap.add_argument("--provider", default="auto", choices=["auto", "gemini", "groq", "ollama"])
    ap.add_argument("--model")
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out", default="results/agent_run.json")
    a = ap.parse_args()
    s = run_agent(a.provider, a.model, a.budget, a.max_turns, out=a.out)
    print(f"\nfirst feasible at sim {s['first_feasible_sim']}, used {s['sims_used']} sims, saved {a.out}")


if __name__ == "__main__":
    main()
