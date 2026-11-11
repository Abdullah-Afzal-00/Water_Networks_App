"""Sample networks for testing and GUI demo presets."""

from __future__ import annotations

from typing import Dict, List, Tuple

from core.models import Network
from core.scenario import Scenario, SwitchEvent, ValveId
from core.physics import PhysicsSettings


def example_paper_network() -> Network:
    """Create a valid, regular paper-inspired network with 1 reservoir, 1 tank, 4 junctions, 1 pump, 8 pipes."""
    net = Network()
    net.add_node("reservoir", id="R1", elevation=50.0)
    net.add_node("tank", id="T1", capacity=1000.0)
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_node("junction", id="J3")
    net.add_node("junction", id="J4")

    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2")
    net.add_pipe("J2", "J3", id="P3", has_pump=True, pump_power=15.0)
    net.add_pipe("J3", "J4", id="P4")
    net.add_pipe("J4", "T1", id="P5")
    net.add_pipe("J1", "J3", id="P6")
    net.add_pipe("J2", "J4", id="P7")
    net.add_pipe("J3", "T1", id="P8")

    return net


def example_broken_network() -> Network:
    """Create a broken network containing an unsupplied island and a pump cycle."""
    net = Network()
    net.add_node("reservoir", id="R1")
    net.add_node("junction", id="J1")
    net.add_node("junction", id="J2")
    net.add_node("junction", id="J3")
    net.add_node("junction", id="J4")
    net.add_node("junction", id="J5")

    # Supply chain with pump loop
    net.add_pipe("R1", "J1", id="P1")
    net.add_pipe("J1", "J2", id="P2", has_pump=True, pump_power=10.0)
    net.add_pipe("J2", "J3", id="P3", has_pump=True, pump_power=10.0)
    net.add_pipe("J3", "J1", id="P4", has_pump=True, pump_power=10.0)

    # Disconnected island without supply
    net.add_pipe("J4", "J5", id="P5")

    net.default_positions = example_broken_network_positions()
    return net


def example_broken_network_positions() -> Dict[str, Tuple[float, float]]:
    """Readable layout for the broken network with no overlapping nodes or labels."""
    return {
        "R1": (-0.8, 0.4),
        "J1": (-0.3, 0.4),
        "J2": (0.2, 0.7),
        "J3": (0.2, 0.1),
        "J4": (0.7, 0.4),
        "J5": (0.7, -0.2),
    }


example_broken_network.positions = example_broken_network_positions()


def specimen_network() -> Tuple[Network, Dict[str, Tuple[float, float]]]:
    """Create the MATLAB reference specimen network with left-to-right positions.

    Nodes:
      - R1: reservoir (elevation 20.0 m, head 50.0 m)
      - J1: junction (elevation 5.0 m)
      - R2: reservoir (elevation 0.0 m, head 46.0 m)

    Pipes:
      - P1: R1 -> J1 (length 400.0 m, diameter 0.4 m, roughness 0.000359 m)
      - P2: J1 -> R2 (length 250.0 m, diameter 0.4 m, roughness 0.000359 m)
        with valve_start=True and valve_end=True
    """
    net = Network()
    net.add_node("reservoir", id="R1", elevation=20.0, head=50.0)
    net.add_node("junction", id="J1", elevation=5.0)
    net.add_node("reservoir", id="R2", elevation=0.0, head=46.0)

    net.add_pipe(
        "R1", "J1", id="P1",
        length=400.0, diameter=0.4, roughness=0.000359,
    )
    net.add_pipe(
        "J1", "R2", id="P2",
        length=250.0, diameter=0.4, roughness=0.000359,
        valve_start=True, valve_end=True,
    )

    positions = {
        "R1": (-0.8, 0.0),
        "J1": (0.0, 0.0),
        "R2": (0.8, 0.0),
    }

    return net, positions


def specimen_scenario(k: int = 1) -> Scenario:
    """Factory for MATLAB reference specimen scenarios (k = 1..5)."""
    v1 = ValveId("P2", "start")
    v2 = ValveId("P2", "end")
    return Scenario.matlab_specimen(k, v1, v2)


def tank_specimen_network() -> Tuple[Network, Dict[str, Tuple[float, float]]]:
    """MATLAB switchedDAE4 reference: reservoir - pipe - valve - pipe - valve - tank.

    Every elevation is 0, so pressure head equals total head everywhere.
      - R1: reservoir, head 30 m
      - J1: junction
      - T1: tank, area 1 m^2, initial level 0 m, capacity 60 m^3 (wall height 60 m)
      - P1: R1 -> J1 (400 m), P2: J1 -> T1 (250 m) with valve_start=True (V1)
        and valve_end=True (V2); both D = 0.4 m, roughness 0.000359 m
    """
    net = Network()
    net.add_node("reservoir", id="R1", elevation=0.0, head=30.0)
    net.add_node("junction", id="J1", elevation=0.0)
    net.add_node("tank", id="T1", elevation=0.0, area=1.0, init_level=0.0, capacity=60.0)
    net.add_pipe("R1", "J1", id="P1", length=400.0, diameter=0.4, roughness=0.000359)
    net.add_pipe(
        "J1", "T1", id="P2",
        length=250.0, diameter=0.4, roughness=0.000359,
        valve_start=True, valve_end=True,
    )
    positions = {"R1": (-0.8, 0.0), "J1": (0.0, 0.0), "T1": (0.8, 0.0)}
    return net, positions


TANK_SCENARIOS: Dict[int, Tuple[str, Dict[str, int], List[Tuple[float, str, int]], float]] = {
    1: ("no switching: the DAE surges, EPANET fills monotonically", {"V1": 1, "V2": 1}, [], 600.0),
    2: ("valve 1 closed at t_s = 60 s, while the tank is filling fast", {"V1": 1, "V2": 1}, [(60.0, "V1", 0)], 300.0),
    3: ("valve 1 closed at t_s = 150 s, during the backflow of the surge", {"V1": 1, "V2": 1}, [(150.0, "V1", 0)], 400.0),
    4: ("valve 2 closed at t_s = 60 s", {"V1": 1, "V2": 1}, [(60.0, "V2", 0)], 300.0),
    5: ("valve 1 opened at t_s = 60 s (start with valve 1 closed)", {"V1": 0, "V2": 1}, [(60.0, "V1", 1)], 600.0),
    6: ("two switches: valve 1 closes at 60 s and reopens at 200 s", {"V1": 1, "V2": 1},
        [(60.0, "V1", 0), (200.0, "V1", 1)], 600.0),
    7: ("three switches: V1 closes at 60 s, reopens at 200 s, V2 closes at 400 s", {"V1": 1, "V2": 1},
        [(60.0, "V1", 0), (200.0, "V1", 1), (400.0, "V2", 0)], 600.0),
}
"""MATLAB switchedDAE4 scenarios 1..7: (title, initial states, (time, valve, state) events, t_end)."""


def tank_specimen_scenario(k: int = 3) -> Scenario:
    """MATLAB switchedDAE4 scenario k (1..7) on the tank specimen; 3 is MATLAB's default."""
    if k not in TANK_SCENARIOS:
        raise ValueError(f"Tank scenario must be one of {sorted(TANK_SCENARIOS)}, got {k!r}")
    title, initial, events, t_end = TANK_SCENARIOS[k]
    valves = {"V1": ValveId("P2", "start"), "V2": ValveId("P2", "end")}
    return Scenario(
        title=title,
        initial={valves[name]: state for name, state in initial.items()},
        events=[SwitchEvent(time, valves[name], state) for time, name, state in events],
        t_end=t_end,
    )


def specimen_physics() -> PhysicsSettings:
    """Physics settings for the specimen reference (constant friction f = 0.02)."""
    return PhysicsSettings(friction_mode="constant", friction_factor=0.02)
