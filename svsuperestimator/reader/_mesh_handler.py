"""This module holds the MeshHandler class."""

from __future__ import annotations

import numpy as np

from ._vtk_handler import VtkHandler


class MeshHandler(VtkHandler):
    """Handler for SimVascular 3D mesh data."""

    @property
    def boundary_centers(self) -> dict[int, np.ndarray]:
        """Center coordinates of the different boundaries in the mesh."""
        boundary_data = self.threshold("ModelFaceID", lower=2)
        max_bc_id = np.max(boundary_data.get_cell_data_array("ModelFaceID"))
        middle_points = {}
        for bc_id in range(2, max_bc_id + 1):
            bc_bounds = boundary_data.threshold(
                "ModelFaceID", lower=bc_id, upper=bc_id
            ).bounds
            middle_points[bc_id] = (
                bc_bounds[[0, 2, 4]] + bc_bounds[[1, 3, 5]]
            ) / 2.0
        return middle_points
