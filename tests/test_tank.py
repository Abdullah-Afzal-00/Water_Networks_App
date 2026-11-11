"""Tanks in the switched DAE and the EPANET comparison.

Reference values are the EXPECTED RESULTS block of MATLAB switchedDAE4_vs_EPANET.m
(reservoir - pipe - valve - pipe - valve - tank, f = 0.02, H = 30 m, A_t = 1 m^2, h0 = 0).
"""

import numpy as np
import pytest

from core.comparison import run_comparison
from core.dae_model import build_switched_dae, check_supported, steady_state
from core.dae_solver import analyze_mode, simulate
from core.epanet_export import export_epanet
from core.examples import specimen_physics, tank_specimen_network, tank_specimen_scenario
from core.models import Network
from core.params import missing_parameters
from core.plotdata import default_states
from core.scenario import ValveId

V1, V2 = ValveId("P2", "start"), ValveId("P2", "end")


@pytest.fixture(scope="module")
def model():
    net, _ = tank_specimen_network()
    return build_switched_dae(net, specimen_physics())


def test_tank_rows_in_E_and_A(model):
    E, A, f = model.matrices({V1: 1, V2: 1})
    row = model.labels.index("T1")
    assert E[row, row] == 1.0
    # level' = Q2 / A_t with A_t = 1, so A(9,2) = 1 as in the supplied matrices
    assert A[row, model.labels.index("Pipe_2_Flow")] == pytest.approx(1.0)
    assert A[row, model.labels.index("Pipe_1_Flow")] == 0.0
    assert f[row] == 0.0


def test_surge_eigenvalues(model):
    D = analyze_mode(model, {V1: 1, V2: 1})
    assert D.n1 == 2
    eig = np.linalg.eigvals(D.J)
    assert np.allclose(eig.real, 0.0, atol=1e-9)
    assert sorted(eig.imag) == pytest.approx([-0.04356, 0.04356], abs=1e-5)
    assert analyze_mode(model, {V1: 0, V2: 1}).n1 == 1
    assert analyze_mode(model, {V1: 1, V2: 0}).n1 == 1


def test_rest_state_and_surge(model):
    res = simulate(model, tank_specimen_scenario(1))
    level = res.series("T1")
    flow = res.series("Pipe_1_Flow")
    # index-2 constraint: x5 = b1*H/(b1+b2), not H
    assert res.series("J1")[0] == pytest.approx(11.538462, abs=1e-6)
    assert level.max() == pytest.approx(32.513, abs=1e-3)
    assert flow.max() == pytest.approx(0.4741, abs=1e-4)
    assert flow.min() == pytest.approx(-0.0858, abs=1e-4)
    assert level[-1] == pytest.approx(29.938, abs=1e-3)


@pytest.mark.parametrize(
    "k, expected",
    [
        (2, {"J1": 98.5700, "Pipe_2_L": -61.6063}),
        (3, {"J1": -27.2193, "Pipe_2_L": 17.0121}),
    ],
)
def test_dirac_impulses_close_v1(model, k, expected):
    res = simulate(model, tank_specimen_scenario(k))
    (imp,) = [i for i in res.impulses if np.any(np.abs(i.coefficients) > 1e-9)]
    for label, value in expected.items():
        assert imp.coefficients[res.labels.index(label)] == pytest.approx(value, abs=1e-3)


def test_three_switches(model):
    res = simulate(model, tank_specimen_scenario(7))
    by_time = {imp.time: imp for imp in res.impulses}
    lab = res.labels
    assert by_time[60.0].coefficients[lab.index("J1")] == pytest.approx(98.5700, abs=1e-3)
    # opening a valve into standing water: no water hammer
    if 200.0 in by_time:
        assert np.all(np.abs(by_time[200.0].coefficients) < 1e-9)
    assert by_time[400.0].x_minus[lab.index("Pipe_1_Flow")] == pytest.approx(0.04596, abs=1e-5)
    assert by_time[400.0].coefficients[lab.index("J1")] == pytest.approx(14.9068, abs=1e-3)
    assert by_time[400.0].coefficients[lab.index("Pipe_2_R")] == pytest.approx(24.2235, abs=1e-3)


def test_tank_freezes_when_valve_shut(model):
    res = simulate(model, tank_specimen_scenario(2))
    level = res.series("T1")
    after = res.t > 60.0 + 1e-6
    assert np.allclose(level[after], 22.8247, atol=1e-4)
    # with the tank cut off its equilibrium is not unique
    with pytest.raises(ValueError):
        steady_state(model, {V1: 0, V2: 1})
    assert steady_state(model, {V1: 1, V2: 1})[model.labels.index("T1")] == pytest.approx(30.0, abs=1e-6)


def test_epanet_tank_matches_matlab():
    net, pos = tank_specimen_network()
    res = run_comparison(net, tank_specimen_scenario(1), specimen_physics(), positions=pos)
    level = res.epanet.series("T1")
    assert res.epanet.series("Pipe_1_Flow")[0] == pytest.approx(0.534934, abs=1e-6)
    assert level[0] == pytest.approx(0.0, abs=1e-6)
    assert level[-1] == pytest.approx(30.0024, abs=1e-3)
    # EPANET has no inertia: it never overshoots the reservoir head, the DAE does
    assert level.max() < 30.01
    assert res.dae.series("T1").max() > 32.5
    # MATLAB switchedDAE4 figure order: Q1, tank level, then the pipe-end heads
    assert default_states(res) == ["Pipe_1_Flow", "T1", "Pipe_1_R", "Pipe_2_L", "Pipe_2_R"]


def test_export_tank_section():
    net, pos = tank_specimen_network()
    inp, export_map = export_epanet(net, tank_specimen_scenario(1), specimen_physics(), positions=pos)
    tanks = inp.split("[TANKS]")[1].split("[PIPES]")[0]
    # floor 1 m below the real bottom, same heads; diameter from A_t = 1 m^2
    assert "TANK_T1  -1.000000   1.000000   0   61.000000   1.12837917   0" in tanks
    assert " C_T1     T1_node  TANK_T1" in inp
    assert export_map.get_element("T1") == ("T1_node", "node")


def test_tank_parameters_required_and_saved():
    net = Network()
    net.add_node("reservoir", id="R1", elevation=0.0, head=30.0)
    net.add_node("tank", id="T1", elevation=0.0)
    net.add_pipe("R1", "T1", id="P1", length=100.0, diameter=0.3, roughness=1e-4, valve_end=True)
    assert "Tank T1: area is not set" in missing_parameters(net)
    assert "Tank T1: initial level is not set" in missing_parameters(net)
    assert check_supported(net)

    net.update_node("T1", area=2.5, init_level=4.0)
    assert not check_supported(net)
    again = Network.from_dict(net.to_dict())
    tank = next(n for n in again.nodes if n.id == "T1")
    assert (tank.area, tank.init_level) == (2.5, 4.0)


def test_tank_fields_rejected_elsewhere():
    net = Network()
    with pytest.raises(ValueError):
        net.add_node("junction", id="J1", area=1.0)
    with pytest.raises(ValueError):
        net.add_node("tank", id="T1", area=0.0)
    with pytest.raises(ValueError):
        net.add_node("tank", id="T2", area=1.0, init_level=-1.0)
