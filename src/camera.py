from typing import Union

import numpy as np


class PinHoleCamera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """Initialize the camera with intrinsic parameters.

        Args:
            fx (float): Focal length in the x direction.
            fy (float): Focal length in the y direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
        """
        self.fx = fx
        self.fy = fy

        self.cx = cx
        self.cy = cy

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points to the image plane.


        Args:
            points_3d (np.ndarray): A numpy array of 3D points in the format [[X1, Y1, Z1], [X2, Y2, Z2], ...].

        Returns:
            np.ndarray: The corresponding 2D image coordinates [[u1, v1], [u2, v2], ...].
        """
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy
        result: np.ndarray[np.Any, np.dtype[np.float64]] = np.vstack((u, v)).T
        return result

    def unproject(
        self, points_2d: np.ndarray, depths: Union[np.ndarray, float]
    ) -> np.ndarray:
        """Unproject image points back to 3D points.

        Args:
            points_2d (np.ndarray): A numpy array of 2D image points in the format [[u1, v1], [u2, v2], ...].
            depths (Union[np.ndarray, float]): The depth of each point in the camera coordinate system. Can be a scalar or an array.

        Returns:
            np.ndarray: The corresponding 3D points [[X1, Y1, Z1], [X2, Y2, Z2], ...].
        """
        u, v = points_2d[:, 0], points_2d[:, 1]
        x = ((u - self.cx) * depths) / self.fx
        y = ((v - self.cy) * depths) / self.fy
        result: np.ndarray[np.Any, np.dtype[np.float64]] = np.vstack((x, y, depths)).T
        return result
