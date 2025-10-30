import os
from typing import Optional

import numpy as np

from .camera import ProcessingError, ValidationError


class LandmarkGenerator:
    """Generate 3D landmarks in world coordinate frame for visual odometry simulation."""

    def __init__(
        self,
        bounds_x: tuple[float, float] = (-10.0, 10.0),
        bounds_y: tuple[float, float] = (-10.0, 10.0),
        bounds_z: tuple[float, float] = (0.0, 5.0),
    ) -> None:
        """Initialize the landmark generator with spatial bounds.

        Args:
            bounds_x (tuple[float, float]): Min and max X coordinates. Defaults to (-10.0, 10.0).
            bounds_y (tuple[float, float]): Min and max Y coordinates. Defaults to (-10.0, 10.0).
            bounds_z (tuple[float, float]): Min and max Z coordinates. Defaults to (0.0, 5.0).
        """
        # Validate bounds
        for bound_name, bound_values in [
            ("bounds_x", bounds_x),
            ("bounds_y", bounds_y),
            ("bounds_z", bounds_z),
        ]:
            if not isinstance(bound_values, tuple) or len(bound_values) != 2:
                raise ValidationError(
                    f"{bound_name} must be a tuple of (min, max), got {bound_values}"
                )
            min_val, max_val = bound_values
            if not isinstance(min_val, (int, float)) or not np.isfinite(min_val):
                raise ValidationError(
                    f"{bound_name} min value must be a finite number, got {min_val}"
                )
            if not isinstance(max_val, (int, float)) or not np.isfinite(max_val):
                raise ValidationError(
                    f"{bound_name} max value must be a finite number, got {max_val}"
                )
            if min_val >= max_val:
                raise ValidationError(
                    f"{bound_name} min ({min_val}) must be less than max ({max_val})"
                )

        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bounds_z = bounds_z

    def generate_random(
        self, num_landmarks: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate randomly distributed landmarks within bounds.

        Args:
            num_landmarks (int): Number of landmarks to generate.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Array of shape (num_landmarks, 3) containing 3D landmark positions.
        """
        # Validate input
        if not isinstance(num_landmarks, int) or num_landmarks <= 0:
            raise ValidationError(
                f"num_landmarks must be a positive integer, got {num_landmarks}"
            )
        if num_landmarks > 1_000_000:  # Reasonable upper limit
            raise ValidationError(
                f"num_landmarks too large ({num_landmarks}), maximum allowed is 1,000,000"
            )

        # Set random seed if provided
        rng = np.random.default_rng(seed)

        # Generate random coordinates within bounds
        x_coords = rng.uniform(self.bounds_x[0], self.bounds_x[1], num_landmarks)
        y_coords = rng.uniform(self.bounds_y[0], self.bounds_y[1], num_landmarks)
        z_coords = rng.uniform(self.bounds_z[0], self.bounds_z[1], num_landmarks)

        # Stack into (N, 3) array
        landmarks = np.column_stack((x_coords, y_coords, z_coords))

        # Final validation
        if not np.all(np.isfinite(landmarks)):
            raise ProcessingError("Generated landmarks contain non-finite values")

        return landmarks

    def generate_on_ground_plane(
        self, num_landmarks: int, z_height: float = 0.0, seed: Optional[int] = None
    ) -> np.ndarray:
        """Generate landmarks on a horizontal ground plane.

        Args:
            num_landmarks (int): Number of landmarks to generate.
            z_height (float): Z-coordinate for all landmarks. Defaults to 0.0.
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Array of shape (num_landmarks, 3) containing 3D landmark positions.
        """
        # Validate z_height
        if not isinstance(z_height, (int, float)) or not np.isfinite(z_height):
            raise ValidationError(f"z_height must be a finite number, got {z_height}")

        if not (self.bounds_z[0] <= z_height <= self.bounds_z[1]):
            raise ValidationError(
                f"z_height ({z_height}) must be within bounds_z {self.bounds_z}"
            )

        # Validate num_landmarks
        if not isinstance(num_landmarks, int) or num_landmarks <= 0:
            raise ValidationError(
                f"num_landmarks must be a positive integer, got {num_landmarks}"
            )
        if num_landmarks > 1_000_000:
            raise ValidationError(
                f"num_landmarks too large ({num_landmarks}), maximum allowed is 1,000,000"
            )

        # Set random seed if provided
        rng = np.random.default_rng(seed)

        # Generate X and Y coordinates within bounds
        x_coords = rng.uniform(self.bounds_x[0], self.bounds_x[1], num_landmarks)
        y_coords = rng.uniform(self.bounds_y[0], self.bounds_y[1], num_landmarks)
        z_coords = np.full(num_landmarks, z_height)

        # Stack into (N, 3) array
        landmarks = np.column_stack((x_coords, y_coords, z_coords))

        return landmarks

    def generate_spherical_distribution(
        self,
        num_landmarks: int,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius_range: tuple[float, float] = (1.0, 5.0),
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate landmarks distributed on spheres within radius range.

        Args:
            num_landmarks (int): Number of landmarks to generate.
            center (tuple[float, float, float]): Center point of distribution. Defaults to (0.0, 0.0, 0.0).
            radius_range (tuple[float, float]): Min and max radius. Defaults to (1.0, 5.0).
            seed (Optional[int]): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Array of shape (num_landmarks, 3) containing 3D landmark positions.
        """
        # Validate center
        if not isinstance(center, tuple) or len(center) != 3:
            raise ValidationError(f"center must be a tuple of 3 floats, got {center}")
        for i, coord in enumerate(center):
            if not isinstance(coord, (int, float)) or not np.isfinite(coord):
                raise ValidationError(
                    f"center coordinate {i} must be finite, got {coord}"
                )

        # Validate radius_range
        if not isinstance(radius_range, tuple) or len(radius_range) != 2:
            raise ValidationError(
                f"radius_range must be a tuple of (min, max), got {radius_range}"
            )
        r_min, r_max = radius_range
        if r_min <= 0 or r_max <= 0 or r_min >= r_max:
            raise ValidationError(
                f"radius_range must have positive values with min < max, got {radius_range}"
            )

        # Validate num_landmarks
        if not isinstance(num_landmarks, int) or num_landmarks <= 0:
            raise ValidationError(
                f"num_landmarks must be a positive integer, got {num_landmarks}"
            )
        if num_landmarks > 1_000_000:
            raise ValidationError(
                f"num_landmarks too large ({num_landmarks}), maximum allowed is 1,000,000"
            )

        # Set random seed if provided
        rng = np.random.default_rng(seed)

        # Generate spherical coordinates
        radii = rng.uniform(r_min, r_max, num_landmarks)
        theta = rng.uniform(0, 2 * np.pi, num_landmarks)  # Azimuthal angle
        phi = rng.uniform(0, np.pi, num_landmarks)  # Polar angle

        # Convert to Cartesian coordinates
        x_coords = center[0] + radii * np.sin(phi) * np.cos(theta)
        y_coords = center[1] + radii * np.sin(phi) * np.sin(theta)
        z_coords = center[2] + radii * np.cos(phi)

        # Stack into (N, 3) array
        landmarks = np.column_stack((x_coords, y_coords, z_coords))

        # Ensure landmarks are within bounds (clip if necessary)
        landmarks[:, 0] = np.clip(landmarks[:, 0], self.bounds_x[0], self.bounds_x[1])
        landmarks[:, 1] = np.clip(landmarks[:, 1], self.bounds_y[0], self.bounds_y[1])
        landmarks[:, 2] = np.clip(landmarks[:, 2], self.bounds_z[0], self.bounds_z[1])

        # Check if clipping removed too many points (indicating bad parameters)
        clipped_count: int = np.sum(
            (landmarks[:, 0] != x_coords)
            | (landmarks[:, 1] != y_coords)
            | (landmarks[:, 2] != z_coords)
        )
        if clipped_count > num_landmarks * 0.1:  # More than 10% clipped
            raise ProcessingError(
                f"Too many landmarks ({clipped_count}/{num_landmarks}) fell outside bounds. "
                f"Consider adjusting center, radius_range, or bounds."
            )

        return landmarks

    def generate_grid(
        self,
        grid_size: tuple[int, int, int],
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Generate landmarks on a 3D grid.

        Args:
            grid_size (tuple[int, int, int]): Number of points in (x, y, z) directions.
            spacing (tuple[float, float, float]): Spacing between points in each direction. Defaults to (1.0, 1.0, 1.0).
            offset (tuple[float, float, float]): Offset from origin. Defaults to (0.0, 0.0, 0.0).

        Returns:
            np.ndarray: Array of shape (num_landmarks, 3) containing 3D landmark positions.
        """
        # Validate grid_size
        if not isinstance(grid_size, tuple) or len(grid_size) != 3:
            raise ValidationError(
                f"grid_size must be a tuple of 3 integers, got {grid_size}"
            )
        for i, size in enumerate(grid_size):
            if not isinstance(size, int) or size <= 0:
                raise ValidationError(
                    f"grid_size[{i}] must be a positive integer, got {size}"
                )

        # Validate spacing
        if not isinstance(spacing, tuple) or len(spacing) != 3:
            raise ValidationError(f"spacing must be a tuple of 3 floats, got {spacing}")
        for i, space in enumerate(spacing):
            if not isinstance(space, (int, float)) or space <= 0:
                raise ValidationError(f"spacing[{i}] must be positive, got {space}")

        # Validate offset
        if not isinstance(offset, tuple) or len(offset) != 3:
            raise ValidationError(f"offset must be a tuple of 3 floats, got {offset}")
        for i, off in enumerate(offset):
            if not isinstance(off, (int, float)) or not np.isfinite(off):
                raise ValidationError(f"offset[{i}] must be finite, got {off}")

        # Calculate total number of landmarks
        total_landmarks = grid_size[0] * grid_size[1] * grid_size[2]
        if total_landmarks > 1_000_000:
            raise ValidationError(
                f"Grid too large ({total_landmarks} points), maximum allowed is 1,000,000"
            )

        # Generate grid coordinates
        x_coords = offset[0] + np.arange(grid_size[0]) * spacing[0]
        y_coords = offset[1] + np.arange(grid_size[1]) * spacing[1]
        z_coords = offset[2] + np.arange(grid_size[2]) * spacing[2]

        # Create 3D meshgrid
        x_grid, y_grid, z_grid = np.meshgrid(
            x_coords, y_coords, z_coords, indexing="ij"
        )

        # Flatten into (N, 3) array
        landmarks = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

        # Check bounds
        if (
            np.any(landmarks[:, 0] < self.bounds_x[0])
            or np.any(landmarks[:, 0] > self.bounds_x[1])
            or np.any(landmarks[:, 1] < self.bounds_y[0])
            or np.any(landmarks[:, 1] > self.bounds_y[1])
            or np.any(landmarks[:, 2] < self.bounds_z[0])
            or np.any(landmarks[:, 2] > self.bounds_z[1])
        ):
            raise ValidationError(
                "Generated grid points fall outside the specified bounds. "
                "Adjust grid_size, spacing, offset, or bounds."
            )

        return landmarks

    def save_landmarks(self, landmarks: np.ndarray, filepath: str) -> None:
        """Save landmarks to a CSV file.

        Args:
            landmarks (np.ndarray): Array of shape (N, 3) containing 3D landmark positions.
            filepath (str): Path to save the landmarks file. Should end with .csv.

        Raises:
            ValidationError: If landmarks array is invalid or filepath is invalid.
            ProcessingError: If file writing fails.
        """
        # Validate landmarks
        if not isinstance(landmarks, np.ndarray):
            raise ValidationError(
                f"landmarks must be a numpy array, got {type(landmarks)}"
            )

        if landmarks.ndim != 2 or landmarks.shape[1] != 3:
            raise ValidationError(
                f"landmarks must have shape (N, 3), got {landmarks.shape}"
            )

        if landmarks.shape[0] == 0:
            raise ValidationError("landmarks cannot be empty")

        if not np.all(np.isfinite(landmarks)):
            raise ValidationError("landmarks must contain only finite values")

        # Validate filepath
        if not isinstance(filepath, str):
            raise ValidationError(f"filepath must be a string, got {type(filepath)}")

        if not filepath.endswith(".csv"):
            raise ValidationError(f"filepath must end with .csv, got {filepath}")

        # Check if directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            raise ValidationError(f"Directory does not exist: {directory}")

        try:
            # Save to CSV with headers
            np.savetxt(
                filepath,
                landmarks,
                delimiter=",",
                header="x,y,z",
                comments="",
                fmt="%.6f",
            )
        except OSError as e:
            raise ProcessingError(f"Failed to save landmarks to {filepath}: {e}") from e

    @classmethod
    def load_landmarks(cls, filepath: str) -> np.ndarray:
        """Load landmarks from a CSV file.

        Args:
            filepath (str): Path to the landmarks CSV file.

        Returns:
            np.ndarray: Array of shape (N, 3) containing 3D landmark positions.

        Raises:
            ValidationError: If filepath is invalid or file format is incorrect.
            ProcessingError: If file reading fails.
        """
        # Validate filepath
        if not isinstance(filepath, str):
            raise ValidationError(f"filepath must be a string, got {type(filepath)}")

        if not os.path.exists(filepath):
            raise ValidationError(f"File does not exist: {filepath}")

        if not os.path.isfile(filepath):
            raise ValidationError(f"Path is not a file: {filepath}")

        if not filepath.endswith(".csv"):
            raise ValidationError(f"filepath must end with .csv, got {filepath}")

        try:
            # First try to load with skiprows=1 (assuming header exists)
            try:
                landmarks = np.loadtxt(filepath, delimiter=",", skiprows=1)
            except ValueError:
                # If that fails, try without skipping rows (no header)
                try:
                    landmarks = np.loadtxt(filepath, delimiter=",", skiprows=0)
                except ValueError as e:
                    raise ProcessingError(
                        f"Invalid CSV format in {filepath}: {e}"
                    ) from e

            # Handle different array shapes
            if landmarks.ndim == 1:
                if landmarks.shape[0] == 3:
                    # Single landmark as 1D array
                    landmarks = landmarks.reshape(1, -1)
                else:
                    raise ProcessingError(f"Invalid landmark data shape in {filepath}")

            if landmarks.ndim != 2 or landmarks.shape[1] != 3:
                raise ProcessingError(
                    f"Landmarks must have shape (N, 3), got {landmarks.shape} in {filepath}"
                )

            # Validate data
            if not np.all(np.isfinite(landmarks)):
                raise ProcessingError(
                    f"Landmarks contain non-finite values in {filepath}"
                )

            return landmarks

        except OSError as e:
            raise ProcessingError(
                f"Failed to load landmarks from {filepath}: {e}"
            ) from e
        except ValueError as e:
            raise ProcessingError(f"Invalid data format in {filepath}: {e}") from e
