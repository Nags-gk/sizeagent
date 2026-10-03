"""Agent loop wiring, using a scripted fake LLM (no network)."""
import json

from sizeagent.agent.loop import run_agent
from sizeagent.circuit import reference_design


class FakeClient:
    model = "fake"

    def __init__(self):
        d = reference_design().to_dict()
        self.design = {k: v for k, v in d.items()}
        self.script = [("simulate", {"design": self.design}),
                       ("refine", {"design": self.design, "sims": 5}),
                       ("submit", {"design": self.design, "rationale": "test"})]

    def chat(self, messages, tools):
        assert any(t["function"]["name"] == "simulate" for t in tools)
        name, args = self.script.pop(0)
        return {"content": f"calling {name}", "tool_calls": [
            {"id": name, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_agent_loop_runs_tools_and_submits(tmp_path):
    out = tmp_path / "run.json"
    s = run_agent(client=FakeClient(), budget=20, out=str(out), verbose=False)
    tools = [x["tool"] for x in s["log"]]
    assert tools == ["simulate", "refine", "submit"]
    assert 1 < s["sims_used"] <= 6
    assert s["log"][0]["result"]["operating_point"]["M1"]["id_ua"] > 0
    assert json.loads(out.read_text())["submitted"]["m1"] == 4


def test_unknown_tool_and_bad_args_are_reported():
    from sizeagent.agent.tools import AgentTools
    from sizeagent.specs import Evaluator
    t = AgentTools(Evaluator(budget=5))
    assert "error" in t.call("rm_rf", {})
    assert "error" in t.call("simulate", {"design": {"inp_w": 1}})


def test_compact_shrinks_old_tool_results_only():
    from sizeagent.agent.loop import compact
    big = json.dumps({"metrics": {"gain_db": 1}, "operating_point": {"M1": "x" * 5000}, "meets_spec": False})
    msgs = [{"role": "system", "content": "s"}] + [{"role": "tool", "content": big} for _ in range(6)]
    out = compact(msgs, keep=2)
    assert all("operating_point" not in m["content"] for m in out[1:5])
    assert all("operating_point" in m["content"] for m in out[5:])
    assert len(msgs) == len(out)


def test_api_failure_keeps_partial_results(tmp_path):
    class Dying(FakeClient):
        def chat(self, messages, tools):
            if not self.script:
                raise RuntimeError("quota")
            return super().chat(messages, tools)

    c = Dying()
    c.script = c.script[:1]
    s = run_agent(client=c, budget=10, out=str(tmp_path / "r.json"), verbose=False)
    assert s["api_error"] == "quota" and s["sims_used"] == 1 and (tmp_path / "r.json").exists()


def test_refine_after_budget_exhausted_returns_error_not_crash(monkeypatch):
    import sizeagent.agent.tools as tl
    from sizeagent.optimizers import encode
    from sizeagent.optimizers.search import local_search
    from sizeagent.specs import Evaluator

    t = tl.AgentTools(Evaluator(budget=0))
    assert "error" in t.refine(reference_design().to_dict(), 5)
    # local_search itself must also swallow an exhausted budget on the very first evaluation
    c, best = local_search(Evaluator(budget=0), encode(reference_design()), 5)
    assert c == float("inf") and best == encode(reference_design())
