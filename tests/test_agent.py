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
