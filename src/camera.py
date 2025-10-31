from typing import Union

import numpy as np

from .validation_error import ProcessingError, ValidationError


class PinHoleCamera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, k1: float = 0.0, k2: float = 0.0, k3: float = 0.0):
        """Initialize the camera with intrinsic parameters.

        Args:
            fx (float): Focal length in the x direction.
            fy (float): Focal length in the y direction.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
            k1 (float): First radial distortion coefficient. Default is 0.0.
            k2 (float): Second radial distortion coefficient. Default is 0.0.
            k3 (float): Third radial distortion coefficient. Default is 0.0.
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

        # Validate distortion coefficients
        if not isinstance(k1, (int, float)) or not np.isfinite(k1):
            raise ValidationError(
                f"Radial distortion coefficient k1 must be a finite number, got {k1}"
            )
        if not isinstance(k2, (int, float)) or not np.isfinite(k2):
            raise ValidationError(
                f"Radial distortion coefficient k2 must be a finite number, got {k2}"
            )
        if not isinstance(k3, (int, float)) or not np.isfinite(k3):
            raise ValidationError(
                f"Radial distortion coefficient k3 must be a finite number, got {k3}"
            )

        self.fx = fx
        self.fy = fy

        self.cx = cx
        self.cy = cy

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

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

        # Basic pinhole projection to normalized coordinates
        x_norm = x / z
        y_norm = y / z

        # Apply radial distortion if any coefficients are non-zero
        if self.k1 != 0.0 or self.k2 != 0.0 or self.k3 != 0.0:
            # Calculate radial distance from optical center
            r_squared = x_norm**2 + y_norm**2
            r = np.sqrt(r_squared)

            # Apply radial distortion formula: r_distorted = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)
            distortion_factor = 1.0 + self.k1 * r_squared + self.k2 * r_squared**2 + self.k3 * r_squared**3

            # Avoid division by zero for points at optical center
            r_distorted = r * distortion_factor
            scale_factor = np.where(r == 0, 1.0, r_distorted / r)

            x_norm = x_norm * scale_factor
            y_norm = y_norm * scale_factor

        # Convert to pixel coordinates
        u = (self.fx * x_norm) + self.cx
        v = (self.fy * y_norm) + self.cy
        result: np.ndarray[np.Any, np.dtype[np.float64]] = np.vstack((u, v)).T
        return result

    def undistort(self, points_2d: np.ndarray) -> np.ndarray:
        """Remove radial distortion from 2D image points.

        Args:
            points_2d (np.ndarray): A numpy array of distorted 2D image points in the format [[u1, v1], [u2, v2], ...].

        Returns:
            np.ndarray: The undistorted 2D image coordinates [[u1, v1], [u2, v2], ...].
        """
        # Validate input
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

        # If no distortion, return points as-is
        if self.k1 == 0.0 and self.k2 == 0.0 and self.k3 == 0.0:
            return points_2d.copy()

        # Convert to normalized coordinates
        u, v = points_2d[:, 0], points_2d[:, 1]
        x_norm_distorted = (u - self.cx) / self.fx
        y_norm_distorted = (v - self.cy) / self.fy

        # Iterative undistortion (typically converges in 1-2 iterations for reasonable distortion)
        x_norm = x_norm_distorted.copy()
        y_norm = y_norm_distorted.copy()

        for _ in range(5):  # Max 5 iterations
            r_squared = x_norm**2 + y_norm**2
            r = np.sqrt(r_squared)

            # Apply distortion formula to current estimate
            distortion_factor = 1.0 + self.k1 * r_squared + self.k2 * r_squared**2 + self.k3 * r_squared**3

            # Calculate correction
            r_distorted = r * distortion_factor
            scale_factor = np.where(r == 0, 1.0, r_distorted / r)

            x_norm_corrected = x_norm_distorted / scale_factor
            y_norm_corrected = y_norm_distorted / scale_factor

            # Update estimate
            x_norm = x_norm_corrected
            y_norm = y_norm_corrected

        # Convert back to pixel coordinates
        u_undistorted = (self.fx * x_norm) + self.cx
        v_undistorted = (self.fy * y_norm) + self.cy

        result: np.ndarray[np.Any, np.dtype[np.float64]] = np.vstack((u_undistorted, v_undistorted)).T
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
