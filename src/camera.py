from typing import Union

import numpy as np


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class ProcessingError(RuntimeError):
    """Raised when data processing fails."""

    pass


class PinHoleCamera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """Initialize the camera with intrinsic parameters.

        Args:
            fx (float): Focal length in the x direction.
            fy (float): Focal length in the y direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
        """
        # Validate focal lengths
        if not isinstance(fx, (int, float)) or not np.isfinite(fx) or fx <= 0:
            raise ValidationError(
                f"Focal length fx must be a positive finite number, got {fx}"
            )
        if not isinstance(fy, (int, float)) or not np.isfinite(fy) or fy <= 0:
            raise ValidationError(
                f"Focal length fy must be a positive finite number, got {fy}"
            )

        # Validate principal points
        if not isinstance(cx, (int, float)) or not np.isfinite(cx):
            raise ValidationError(
                f"Principal point cx must be a finite number, got {cx}"
            )
        if not isinstance(cy, (int, float)) or not np.isfinite(cy):
            raise ValidationError(
                f"Principal point cy must be a finite number, got {cy}"
            )

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
        # Validate input
        if not isinstance(points_3d, np.ndarray):
            raise ValidationError(
                f"points_3d must be a numpy array, got {type(points_3d)}"
            )

        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValidationError(
                f"points_3d must have shape (N, 3), got {points_3d.shape}"
            )

        if points_3d.shape[0] == 0:
            raise ValidationError("points_3d cannot be empty")

        if not np.all(np.isfinite(points_3d)):
            raise ValidationError("points_3d must contain only finite values")

        # Check for zero Z coordinates (would cause division by zero)
        z_coords = points_3d[:, 2]
        if np.any(z_coords == 0):
            raise ProcessingError(
                "Cannot project points with zero Z coordinates (division by zero)"
            )

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
        # Validate points_2d
        if not isinstance(points_2d, np.ndarray):
            raise ValidationError(
                f"points_2d must be a numpy array, got {type(points_2d)}"
            )

        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise ValidationError(
                f"points_2d must have shape (N, 2), got {points_2d.shape}"
            )

        if points_2d.shape[0] == 0:
            raise ValidationError("points_2d cannot be empty")

        if not np.all(np.isfinite(points_2d)):
            raise ValidationError("points_2d must contain only finite values")

        # Validate depths
        num_points = points_2d.shape[0]

        if isinstance(depths, np.ndarray):
            if depths.ndim == 0:  # scalar array
                depths = float(depths.item())
            elif depths.ndim == 1:
                if depths.shape[0] != num_points:
                    raise ValidationError(
                        f"depths array must have shape ({num_points},), got {depths.shape}"
                    )
                if not np.all(np.isfinite(depths)):
                    raise ValidationError("depths must contain only finite values")
                if np.any(depths <= 0):
                    raise ValidationError("depths must be positive")
            elif depths.ndim == 2 and depths.shape == (num_points, 1):
                depths = depths.flatten()
                if not np.all(np.isfinite(depths)):
                    raise ValidationError("depths must contain only finite values")
                if np.any(depths <= 0):
                    raise ValidationError("depths must be positive")
            else:
                raise ValidationError(
                    f"depths array must have shape ({num_points},) or ({num_points}, 1), got {depths.shape}"
                )
        elif isinstance(depths, (int, float)):
            if not np.isfinite(depths):
                raise ValidationError("depths must be a finite number")
            if depths <= 0:
                raise ValidationError("depths must be positive")
            # Convert scalar to array for consistent processing
            depths = np.full(num_points, depths, dtype=float)
        else:
            raise ValidationError(
                f"depths must be a number or numpy array, got {type(depths)}"
            )

        u, v = points_2d[:, 0], points_2d[:, 1]
        x = ((u - self.cx) * depths) / self.fx
        y = ((v - self.cy) * depths) / self.fy
        result: np.ndarray[np.Any, np.dtype[np.float64]] = np.vstack((x, y, depths)).T
        return result
