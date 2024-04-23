"""This module holds general task helper function."""

import subprocess
from time import sleep
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline

from .. import reader


def cgs_pressure_to_mmgh(
    cgs_pressure: Union[Sequence, np.ndarray]
) -> np.ndarray:
    """Convert pressure from g/(cm s^2) to mmHg.

    Args:
        cgs_pressure: Pressure in CGS format.

    Returns:
        Pressure in mmHg.
    """
    return np.array(np.array(cgs_pressure) * 0.00075006156130264)


def cgs_flow_to_lh(cgs_flow: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Convert flow from cm^3/s to l/h.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return np.array(np.array(cgs_flow) * 3.6)


def cgs_flow_to_lmin(cgs_flow: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Convert flow from cm^3/s to l/min.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return np.array(np.array(cgs_flow) * 60.0 / 1000.0)


def refine_with_cubic_spline(y: np.ndarray, num: int) -> np.ndarray:
    """Refine a curve using cubic spline interpolation.

    Args:
        y: The data to refine.
        num: New number of points of the refined data.
    """
    y = y.copy()
    y[-1] = y[0]
    x_old = np.linspace(0.0, 100.0, len(y))
    x_new = np.linspace(0.0, 100.0, num)
    y_new = CubicSpline(x_old, y, bc_type="periodic")(x_new)
    return y_new


def refine_with_cubic_spline_derivative(
    x: np.ndarray, y: np.ndarray, num: int
) -> np.ndarray:
    """Refine a curve using cubic spline interpolation with derivative.

    Args:
        x: X-coordinates
        y: Y-coordinates
        num: New number of points of the refined data.


    Returns:
        new_y: New y-coordinates
        new_dy: New dy-coordinates
    """
    y = y.copy()
    y[-1] = y[0]
    x_new = np.linspace(x[0], x[-1], num)
    spline = CubicSpline(x, y, bc_type="periodic")
    new_y = spline(x_new)
    new_dy = spline.derivative()(x_new)
    return new_y, new_dy


def run_subprocess(
    args: list,
    logger: Callable,
    refresh_rate: float = 1.0,
    logprefix: str = "",
    cwd: Optional[str] = None,
) -> None:
    """Run a subprocess.

    Args:
        args: Arguments for the subprocess.
        logger: Logger to use for logging.
        refresh_rate: Rate to update logging (in seconds).
        logprefix: Prefix to put in front of lines when logging.
        cwd: Working directory of the subprocess.
    """
    process = subprocess.Popen(
        " ".join(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=cwd,
    )

    def check_io():  # type: ignore
        while True:
            output = process.stdout.readline().decode().strip()
            if output:
                logger(logprefix + output)
            else:
                break

    while process.poll() is None:
        check_io()
        sleep(refresh_rate)
    else:
        check_io()

    if process.returncode != 0:
        raise RuntimeError("Subprocess failed")


def map_centerline_result_to_0d_2(
    zerod_handler: reader.SvZeroDSolverInputHandler,
    cl_handler: reader.CenterlineHandler,
    threed_handler: reader.SvSolverInputHandler,
    results_handler: reader.CenterlineHandler,
    padding: bool = False,
) -> Tuple[dict, np.ndarray]:
    """Map centerine result onto 0d elements."""

    # calculate cycle period
    cycle_period = (
        zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][-1]
        - zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = results_handler.time_steps * threed_handler.time_step_size

    # Calculate start of last cycle
    start_last_cycle = (
        np.abs(times - (times[-1] - cycle_period))
    ).argmin() - 1

    def filter_last_cycle(data, seg_end_index):  # type: ignore
        if start_last_cycle == -1:
            return data[:, seg_end_index]
        return data[start_last_cycle:-1, seg_end_index]

    # Extract branch information of 0D config
    branchdata: dict = {}
    for vessel_config in zerod_handler.vessels.values():
        # Extract branch and segment id from name
        name = vessel_config["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        if branch_id not in branchdata:
            branchdata[branch_id] = {}

        branchdata[branch_id][seg_id] = {}
        branchdata[branch_id][seg_id]["length"] = vessel_config[
            "vessel_length"
        ]
        branchdata[branch_id][seg_id]["vessel_id"] = vessel_config["vessel_id"]

    results_branch_starts = {}
    for branch_id, branch in branchdata.items():
        results_branch_starts[branch_id] = results_handler.get_branch_data(
            branch_id
        )["points"][0]

    keys = list(results_branch_starts.keys())
    starts = np.array(list(results_branch_starts.values()))

    for branch_id, branch in branchdata.items():
        cl_data = cl_handler.get_branch_data(branch_id)
        start = cl_data["points"][0]

        new_id = keys[np.argmin(np.linalg.norm(starts - start, axis=1))]

        branch_data = results_handler.get_branch_data(new_id)

        seg_start = 0.0
        seg_start_index = 0

        if padding:
            branch_data["flow"][:, 0] = branch_data["flow"][:, 1]
            branch_data["flow"][:, -1] = branch_data["flow"][:, -2]
            branch_data["pressure"][:, 0] = branch_data["pressure"][:, 1]
            branch_data["pressure"][:, -1] = branch_data["pressure"][:, -2]

        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            length = segment["length"]

            seg_end_index = (
                np.abs(branch_data["path"] - length - seg_start)
            ).argmin()
            if (np.abs(branch_data["path"] - length - seg_start)).min() > 1e2:
                raise RuntimeError(
                    "Indexing mismatch between 0D solver input file and "
                    "centerline. Please check that 0D config and centerline "
                    "are matching."
                )
            seg_end = branch_data["path"][seg_end_index]

            segment.update(
                {
                    "flow_in": filter_last_cycle(
                        branch_data["flow"], seg_start_index
                    ),
                    "flow_out": filter_last_cycle(
                        branch_data["flow"], seg_end_index
                    ),
                    "pressure_in": filter_last_cycle(
                        branch_data["pressure"], seg_start_index
                    ),
                    "pressure_out": filter_last_cycle(
                        branch_data["pressure"], seg_end_index
                    ),
                    "x0": branch_data["points"][seg_start_index],
                    "x1": branch_data["points"][seg_end_index],
                }
            )

            seg_start = seg_end
            seg_start_index = seg_end_index

    if start_last_cycle == -1:
        times -= times[0]
    else:
        times = times[start_last_cycle:-1] - np.amin(times[start_last_cycle])

    return branchdata, times


def map_centerline_result_to_0d_3(
    zerod_handler: reader.SvZeroDSolverInputHandler,
    cl_handler: reader.CenterlineHandler,
    threed_handler: reader.SvSolverInputHandler,
    results_handler: reader.CenterlineHandler,
) -> Tuple[dict, np.ndarray]:
    """Map centerine result onto 0d elements."""

    # calculate cycle period
    cycle_period = (
        zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][-1]
        - zerod_handler.boundary_conditions["INFLOW"]["bc_values"]["t"][0]
    )

    # Extract time steps
    times = results_handler.time_steps * threed_handler.time_step_size

    # Calculate start of last cycle
    start_last_cycle = (
        np.abs(times - (times[-1] - cycle_period))
    ).argmin() - 1

    def filter_last_cycle(data):  # type: ignore
        if start_last_cycle == -1:
            return data[:]
        return data[start_last_cycle:-1]

    # Extract branch information of 0D config
    branchdata: dict = {}
    for vessel_config in zerod_handler.vessels.values():
        # Extract branch and segment id from name
        name = vessel_config["vessel_name"]
        branch_id, seg_id = name.split("_")
        branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])

        if branch_id not in branchdata:
            branchdata[branch_id] = {}

        branchdata[branch_id][seg_id] = {}
        branchdata[branch_id][seg_id]["length"] = vessel_config[
            "vessel_length"
        ]
        branchdata[branch_id][seg_id]["vessel_id"] = vessel_config["vessel_id"]

    # Add start and endpoint of branches
    for branch_id, branch in branchdata.items():
        cl_data = cl_handler.get_branch_data(branch_id)

        seg_start = 0.0
        seg_start_index = 0

        cum_len = sum([segment["length"] for segment in branch.values()])
        if abs(cum_len - cl_data["path"][-1]) > 1e4:
            raise RuntimeError(
                "Indexing mismatch between 0D solver input file and "
                "centerline. Please check that 0D config and centerline "
                "are matching."
            )

        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            length = segment["length"]
            seg_end_index = (
                np.abs(cl_data["path"] - length - seg_start)
            ).argmin()
            seg_end = cl_data["path"][seg_end_index]
            segment.update(
                {
                    "x0": cl_data["points"][seg_start_index],
                    "x1": cl_data["points"][seg_end_index],
                }
            )
            seg_start = seg_end
            seg_start_index = seg_end_index

    # Get field data
    for branch_id, branch in branchdata.items():
        for seg_id in range(len(branch)):
            segment = branch[seg_id]
            x0 = segment["x0"]
            x1 = segment["x1"]
            pressure_in, flow_in = results_handler.get_values_at_node(x0)
            pressure_out, flow_out = results_handler.get_values_at_node(x1)

            segment.update(
                {
                    "flow_in": filter_last_cycle(flow_in),
                    "flow_out": filter_last_cycle(flow_out),
                    "pressure_in": filter_last_cycle(pressure_in),
                    "pressure_out": filter_last_cycle(pressure_out),
                }
            )

    if start_last_cycle == -1:
        times -= times[0]
    else:
        times = times[start_last_cycle:-1] - np.amin(times[start_last_cycle])

    return branchdata, times


def set_initial_condition(
    zerod_handler: reader.SvZeroDSolverInputHandler, mapped_data: dict, times
) -> None:
    """Set initial condition of 0D configuration based on mapped 0D results.

    Args:
        zerod_handler: 0D simulation input handler.
        mapped_data: Mapped 3D result.
        times: Time steps.
    """

    nodes = zerod_handler.nodes

    bcs = zerod_handler.boundary_conditions

    initial_condition = {}
    initial_condition_d = {}
    for ele1, ele2 in nodes:
        if ele1.startswith("branch"):
            branch_id, seg_id = ele1.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            pressure = mapped_data[branch_id][seg_id]["pressure_out"]
            flow = mapped_data[branch_id][seg_id]["flow_out"]
            pres, dpres = refine_with_cubic_spline_derivative(
                times, pressure, len(times)
            )
            flow, dflow = refine_with_cubic_spline_derivative(
                times, flow, len(times)
            )
            initial_condition[f"pressure:{ele1}:{ele2}"] = pres[0]
            initial_condition_d[f"pressure:{ele1}:{ele2}"] = dpres[0]
            initial_condition[f"flow:{ele1}:{ele2}"] = flow[0]
            initial_condition_d[f"flow:{ele1}:{ele2}"] = dflow[0]

        if ele2.startswith("branch"):
            branch_id, seg_id = ele2.split("_")
            branch_id, seg_id = int(branch_id[6:]), int(seg_id[3:])
            pressure = mapped_data[branch_id][seg_id]["pressure_in"]
            flow = mapped_data[branch_id][seg_id]["flow_in"]
            pres, dpres = refine_with_cubic_spline_derivative(
                times, pressure, len(times)
            )
            flow, dflow = refine_with_cubic_spline_derivative(
                times, flow, len(times)
            )
            initial_condition[f"pressure:{ele1}:{ele2}"] = pres[0]
            initial_condition_d[f"pressure:{ele1}:{ele2}"] = dpres[0]
            initial_condition[f"flow:{ele1}:{ele2}"] = flow[0]
            initial_condition_d[f"flow:{ele1}:{ele2}"] = dflow[0]

        if ele2.startswith("RCR"):
            initial_condition[f"pressure_c:{ele2}"] = (
                initial_condition[f"pressure:{ele1}:{ele2}"]
                - bcs[ele2]["bc_values"]["Rp"]
                * initial_condition[f"flow:{ele1}:{ele2}"]
            )
            initial_condition_d[f"pressure_c:{ele2}"] = (
                initial_condition_d[f"pressure:{ele1}:{ele2}"]
                - bcs[ele2]["bc_values"]["Rp"]
                * initial_condition_d[f"flow:{ele1}:{ele2}"]
            )
        if ele2.startswith("RESISTANCE"):
            initial_condition[f"pressure_c:{ele2}"] = initial_condition[
                f"pressure:{ele1}:{ele2}"
            ]
            initial_condition_d[f"pressure_c:{ele2}"] = initial_condition_d[
                f"pressure:{ele1}:{ele2}"
            ]

    vessel_id_map = zerod_handler.vessel_id_to_name_map

    for junction_name, junction in zerod_handler.junctions.items():
        if len(junction["outlet_vessels"]) > 1:
            for i, outlet_vessel in enumerate(junction["outlet_vessels"]):
                ovessel_name = vessel_id_map[outlet_vessel]
                initial_condition[f"flow_{i}:{junction_name}"] = (
                    initial_condition[f"flow:{junction_name}:{ovessel_name}"]
                )
                initial_condition_d[f"flow_{i}:{junction_name}"] = (
                    initial_condition_d[f"flow:{junction_name}:{ovessel_name}"]
                )

    zerod_handler.data["initial_condition"] = initial_condition
    zerod_handler.data["initial_condition_d"] = initial_condition_d
