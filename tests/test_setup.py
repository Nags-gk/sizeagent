import pytest

import sizeagent.circuit as circuit
import sizeagent.spice as spice
from sizeagent.circuit import SetupError, reference_design


def test_missing_ngspice_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(spice.shutil, "which", lambda _: None)
    with pytest.raises(SetupError, match="ngspice"):
        spice.simulate(reference_design())


def test_missing_pdk_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(circuit, "PDK_DIR", tmp_path)
    circuit.valid_geometries.cache_clear()
    try:
        with pytest.raises(SetupError, match="setup_pdk.sh"):
            circuit.valid_geometries()
    finally:
        circuit.valid_geometries.cache_clear()
