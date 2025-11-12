from typing import List, Optional

import numpy as np

from .geometry import GeometryUtils
from .validation_error import ValidationError


class CameraPose:
    """Represents a camera pose with position and orientation."""

    def __init__(
        self, position: np.ndarray, orientation: np.ndarray, timestamp: float = 0.0
    ):
        """Initialize camera pose.

        Args:
            position (np.ndarray): 3D position vector (x, y, z).
            orientation (np.ndarray): Orientation as quaternion [w, x, y, z].
            timestamp (Optional[float]): Timestamp for this pose. Defaults to None.
        """
        # Validate position
        if not isinstance(position, np.ndarray) or position.shape != (3,):
            raise ValidationError(
                f"position must be a numpy array of shape (3,), got {position.shape if isinstance(position, np.ndarray) else type(position)}"
            )
        if not np.all(np.isfinite(position)):
            raise ValidationError("position must contain only finite values")

        self.position = position.copy()

        # Validate orientation (quaternion only)
        if not isinstance(orientation, np.ndarray) or orientation.shape != (4,):
            raise ValidationError(
                f"orientation must be a numpy array of shape (4,) for quaternion [w, x, y, z], got {orientation.shape if isinstance(orientation, np.ndarray) else type(orientation)}"
            )

        if not GeometryUtils.validate_quaternion(orientation):
            raise ValidationError("orientation quaternion is not valid")

        self.orientation_quaternion = orientation.copy()
        self.timestamp = timestamp

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the rotation matrix representation."""
        return GeometryUtils.rotation_matrix_from_quaternion(
            self.orientation_quaternion
        )

    @property
    def quaternion(self) -> np.ndarray:
        """Get the quaternion representation."""
        return self.orientation_quaternion.copy()

    def transform_points_world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Transform points from world coordinates to camera coordinates.

        Args:
            points_world (np.ndarray): Points in world frame, shape (N, 3).

        Returns:
            np.ndarray: Points in camera frame, shape (N, 3).
        """
        # First translate, then rotate
        # Note: rotation_matrix is the rotation matrix from world to camera frame
        # so we R_CF_I (rotation frame matrix is the transpose of the rotation matrix)
        points_relative = points_world - self.position
        return points_relative @ self.rotation_matrix  # (p - t) * R

    def transform_points_camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """Transform points from camera coordinates to world coordinates.

        Args:
            points_camera (np.ndarray): Points in camera frame, shape (N, 3).

        Returns:
            np.ndarray: Points in world frame, shape (N, 3).
        """
        # First rotate, then translate
        return (self.rotation_matrix @ points_camera.T).T + self.position

    def get_transform_matrix(self) -> np.ndarray:
        """Get the 4x4 transformation matrix from world to camera coordinates.

        Returns:
            np.ndarray: 4x4 transformation matrix [R, t; 0, 1] where R is rotation
                       and t is translation.
        """
        t = np.eye(4)
        t[:3, :3] = self.rotation_matrix.T  # World to camera rotation
        t[:3, 3] = (
            -self.rotation_matrix.T @ self.position
        )  # World to camera translation
        return t

    @classmethod
    def from_transform_matrix(
        cls, transform_matrix: np.ndarray, timestamp: Optional[float] = None
    ) -> "CameraPose":
        """Create CameraPose from 4x4 transformation matrix.

        Args:
            transform_matrix (np.ndarray): 4x4 transformation matrix [R, t; 0, 1].
            timestamp (Optional[float]): Timestamp for this pose.

        Returns:
            CameraPose: The created camera pose.
        """
        if not isinstance(transform_matrix, np.ndarray) or transform_matrix.shape != (
            4,
            4,
        ):
            raise ValidationError(
                f"transform_matrix must be a 4x4 numpy array, got {transform_matrix.shape if isinstance(transform_matrix, np.ndarray) else type(transform_matrix)}"
            )

        rotation_matrix = transform_matrix[:3, :3]
        translation = transform_matrix[:3, 3]

        # For world to camera transform [R, t], the camera position in world is -R^T * t
        position = -rotation_matrix.T @ translation

        # Convert rotation matrix to quaternion
        quaternion = GeometryUtils.quaternion_from_rotation_matrix(rotation_matrix)

        return cls(position, quaternion, timestamp)

    @staticmethod
    def create_look_at_pose(
        camera_position: np.ndarray, target_position: np.ndarray, timestamp: float = 0.0
    ) -> "CameraPose":
        """Create a camera pose that points toward a target position.

        The camera's Z-axis will point toward the target position, and the camera's
        X-axis will be parallel to the world frame's X-axis.

        Args:
            camera_position (np.ndarray): 3D position of the camera (x, y, z).
            target_position (np.ndarray): 3D position of the target to look at (x, y, z).
            timestamp (float): Timestamp for this pose. Defaults to 0.0.

        Returns:
            CameraPose: Camera pose with the specified orientation.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate inputs
        for pos_name, pos in [
            ("camera_position", camera_position),
            ("target_position", target_position),
        ]:
            if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                raise ValidationError(
                    f"{pos_name} must be a 3D numpy array, got {pos.shape if isinstance(pos, np.ndarray) else type(pos)}"
                )
            if not np.all(np.isfinite(pos)):
                raise ValidationError(f"{pos_name} must contain finite values")

        # Check if camera and target positions are the same
        if np.allclose(camera_position, target_position):
            raise ValidationError(
                "Camera position and target position cannot be the same"
            )

        # Direction vector from camera to target
        direction = target_position - camera_position
        direction = GeometryUtils.normalize_vector(direction)

        # Camera Z-axis points toward target (in camera coordinates, +Z is forward)
        camera_z = direction

        # Camera X-axis should be parallel to world X-axis [1, 0, 0]
        world_x = np.array([1.0, 0.0, 0.0])

        # Check if camera_z is parallel to world_x (degenerate case)
        if np.allclose(np.abs(np.dot(camera_z, world_x)), 1.0):
            # If camera_z is parallel to world_x, we need a different approach
            # Use world Y-axis as the reference direction
            world_y = np.array([0.0, 1.0, 0.0])
            camera_x = world_y
        else:
            # Project world X onto plane perpendicular to camera_z
            camera_x = world_x - np.dot(world_x, camera_z) * camera_z
            camera_x = GeometryUtils.normalize_vector(camera_x)

        # Camera Y-axis is cross product of Z and X (right-hand rule)
        # Y = Z × X, so that X × Y = Z (forward direction)
        camera_y = np.cross(camera_z, camera_x)

        # Construct rotation matrix (camera to world transformation)
        # Columns are the camera axes expressed in world coordinates
        rotation_matrix = np.column_stack([camera_x, camera_y, camera_z])

        # Convert to quaternion
        quaternion = GeometryUtils.quaternion_from_rotation_matrix(rotation_matrix)

        return CameraPose(camera_position, quaternion, timestamp)

    def copy(self) -> "CameraPose":
        """Create a copy of the camera pose."""
        return CameraPose(
            self.position.copy(), self.orientation_quaternion.copy(), self.timestamp
        )

    def __repr__(self) -> str:
        """String representation of the camera pose."""
        timestamp_str = (
            f", t={self.timestamp:.3f}" if self.timestamp is not None else ""
        )
        return (
            f"CameraPose(position=[{self.position[0]:.3f}, {self.position[1]:.3f}, "
            f"{self.position[2]:.3f}], orientation=[{self.orientation_quaternion[0]:.3f}, "
            f"{self.orientation_quaternion[1]:.3f}, {self.orientation_quaternion[2]:.3f}, "
            f"{self.orientation_quaternion[3]:.3f}]{timestamp_str})"
        )


class TrajectoryGenerator:
    """Generate camera trajectories for visual odometry simulation."""

    def __init__(self, time_step: float = 0.1):
        """Initialize trajectory generator.

        Args:
            time_step (float): Default time step between poses in seconds.
        """
        if (
            not isinstance(time_step, (int, float))
            or not np.isfinite(time_step)
            or time_step <= 0
        ):
            raise ValidationError(
                f"time_step must be a positive finite number, got {time_step}"
            )
        self.time_step = time_step

    def generate_circular_trajectory(
        self,
        center: np.ndarray,
        radius: float,
        height: float,
        num_poses: int,
        start_angle: float = 0.0,
        angular_velocity: float = 1.0,
        look_at_center: bool = True,
        orientation_offset: Optional[np.ndarray] = None,
    ) -> List[CameraPose]:
        """Generate a circular camera trajectory.

        Args:
            center (np.ndarray): Center point of the circle (x, y).
            radius (float): Radius of the circular path.
            height (float): Z-coordinate (height) of the camera.
            num_poses (int): Number of poses to generate.
            start_angle (float): Starting angle in radians. Defaults to 0.
            angular_velocity (float): Angular velocity in rad/s. Defaults to 1.0.
            look_at_center (bool): If True, camera always looks at the center. Defaults to True.
            orientation_offset (Optional[np.ndarray]): Offset to the orientation quaternion. Defaults to None.

        Returns:
            List[CameraPose]: List of camera poses along the trajectory.
        """
        # Validate inputs
        if not isinstance(center, np.ndarray) or center.shape != (2,):
            raise ValidationError(
                f"center must be a 2D numpy array, got {center.shape if isinstance(center, np.ndarray) else type(center)}"
            )
        if not np.all(np.isfinite(center)):
            raise ValidationError("center must contain finite values")

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValidationError(f"radius must be positive, got {radius}")
        if not isinstance(height, (int, float)) or not np.isfinite(height):
            raise ValidationError(f"height must be finite, got {height}")
        if not isinstance(num_poses, int) or num_poses <= 0:
            raise ValidationError(
                f"num_poses must be positive integer, got {num_poses}"
            )

        poses = []

        for i in range(num_poses):
            # Calculate angle for this pose
            angle = start_angle + i * angular_velocity * self.time_step
            timestamp = i * self.time_step

            # Calculate position on circle
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            position = np.array([x, y, height])

            if look_at_center:
                # Camera looks at the center point
                # Direction vector from camera to center
                look_direction = np.array([center[0] - x, center[1] - y, 0])
                look_direction = look_direction / np.linalg.norm(look_direction)

                # Create quaternion that points camera toward center
                yaw = np.arctan2(look_direction[1], look_direction[0])
                quaternion = GeometryUtils.quaternion_from_axis_angle(
                    np.array([0.0, 0.0, 1.0]), yaw
                )
            else:
                # Camera faces tangent to the circle
                tangent_angle = angle + np.pi / 2  # 90 degrees ahead
                yaw = tangent_angle
                quaternion = GeometryUtils.quaternion_from_axis_angle(
                    np.array([0.0, 0.0, 1.0]), yaw
                )

            # Apply orientation offset
            if orientation_offset is not None:
                GeometryUtils.validate_quaternion(orientation_offset)
                quaternion = GeometryUtils.quaternion_multiply(
                    quaternion,
                    orientation_offset,
                )

            pose = CameraPose(position, quaternion, timestamp)
            poses.append(pose)

        return poses

    def generate_linear_trajectory(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray,
        num_poses: int,
        orientation: Optional[np.ndarray] = None,
    ) -> List[CameraPose]:
        """Generate a linear camera trajectory.

        Args:
            start_position (np.ndarray): Starting position (x, y, z).
            end_position (np.ndarray): Ending position (x, y, z).
            num_poses (int): Number of poses to generate.
            orientation (Optional[np.ndarray]): Fixed orientation as quaternion [w, x, y, z].
                                                If None, camera faces direction of motion.

        Returns:
            List[CameraPose]: List of camera poses along the trajectory.
        """
        # Validate inputs
        for pos_name, pos in [
            ("start_position", start_position),
            ("end_position", end_position),
        ]:
            if not isinstance(pos, np.ndarray) or pos.shape != (3,):
                raise ValidationError(
                    f"{pos_name} must be a 3D numpy array, got {pos.shape if isinstance(pos, np.ndarray) else type(pos)}"
                )
            if not np.all(np.isfinite(pos)):
                raise ValidationError(f"{pos_name} must contain finite values")

        if not isinstance(num_poses, int) or num_poses <= 1:
            raise ValidationError(f"num_poses must be integer > 1, got {num_poses}")

        # Handle orientation
        if orientation is None:
            # Camera faces direction of motion
            direction = end_position - start_position
            direction = direction / np.linalg.norm(direction)

            # Calculate yaw angle from direction vector
            yaw = np.arctan2(direction[1], direction[0])
            orientation = GeometryUtils.quaternion_from_axis_angle(
                np.array([0.0, 0.0, 1.0]), yaw
            )
        elif isinstance(orientation, np.ndarray):
            if orientation.shape != (4,):
                raise ValidationError(
                    f"orientation must be a quaternion of shape (4,), got {orientation.shape}"
                )
            if not GeometryUtils.validate_quaternion(orientation):
                raise ValidationError("orientation quaternion is not valid")

        poses = []

        for i in range(num_poses):
            # Linear interpolation between start and end positions
            t = i / (num_poses - 1)  # Parameter from 0 to 1
            position = start_position + t * (end_position - start_position)
            timestamp = i * self.time_step

            pose = CameraPose(position, orientation, timestamp)
            poses.append(pose)

        return poses

    def generate_figure_eight_trajectory(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        z_level: float,
        num_poses: int,
        speed: float = 1.0,
    ) -> List[CameraPose]:
        """Generate a figure-eight (lemniscate) camera trajectory.

        Args:
            center (np.ndarray): Center point of the figure-eight (x, y).
            width (float): Width of the figure-eight.
            height (float): Height of the figure-eight.
            z_level (float): Z-coordinate of the trajectory.
            num_poses (int): Number of poses to generate.
            speed (float): Speed parameter for the trajectory.

        Returns:
            List[CameraPose]: List of camera poses along the trajectory.
        """
        # Validate inputs
        if not isinstance(center, np.ndarray) or center.shape != (2,):
            raise ValidationError(
                f"center must be a 2D numpy array, got {center.shape if isinstance(center, np.ndarray) else type(center)}"
            )

        for param_name, param in [
            ("width", width),
            ("height", height),
            ("z_level", z_level),
            ("speed", speed),
        ]:
            if not isinstance(param, (int, float)) or not np.isfinite(param):
                raise ValidationError(f"{param_name} must be finite, got {param}")

        if width <= 0 or height <= 0:
            raise ValidationError("width and height must be positive")

        if not isinstance(num_poses, int) or num_poses <= 0:
            raise ValidationError(
                f"num_poses must be positive integer, got {num_poses}"
            )

        poses = []

        for i in range(num_poses):
            # Parameter for figure-eight: use sine/cosine with different frequencies
            t = i * self.time_step * speed
            timestamp = i * self.time_step

            # Figure-eight parametric equations
            x = center[0] + width * np.sin(t)
            y = center[1] + height * np.sin(2 * t) / 2
            position = np.array([x, y, z_level])

            # Calculate tangent direction for orientation
            # Derivative of figure-eight equations
            dx_dt = width * np.cos(t)
            dy_dt = height * np.cos(2 * t)
            tangent = np.array([dx_dt, dy_dt, 0])
            tangent = tangent / np.linalg.norm(tangent)

            # Calculate yaw angle from tangent vector
            yaw = np.arctan2(tangent[1], tangent[0])
            quaternion = GeometryUtils.quaternion_from_axis_angle(
                np.array([0.0, 0.0, 1.0]), yaw
            )

            pose = CameraPose(position, quaternion, timestamp)
            poses.append(pose)

        return poses

    def save_trajectory(self, poses: List[CameraPose], filepath: str) -> None:
        """Save trajectory to a CSV file.

        Args:
            poses (List[CameraPose]): List of camera poses.
            filepath (str): Path to save the trajectory file.
        """
        import os

        if not filepath.endswith(".csv"):
            raise ValidationError(f"filepath must end with .csv, got {filepath}")

        # Check directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Prepare data for saving
        data = []
        for pose in poses:
            row = [
                pose.timestamp if pose.timestamp is not None else 0.0,
                pose.position[0],
                pose.position[1],
                pose.position[2],
                pose.quaternion[0],
                pose.quaternion[1],
                pose.quaternion[2],
                pose.quaternion[3],
            ]
            data.append(row)

        np.savetxt(
            filepath,
            data,
            delimiter=",",
            header="timestamp,x,y,z,qw,qx,qy,qz",
            comments="",
            fmt="%.6f",
        )

    @classmethod
    def load_trajectory(cls, filepath: str) -> List[CameraPose]:
        """Load trajectory from a CSV file.

        Args:
            filepath (str): Path to the trajectory CSV file.

        Returns:
            List[CameraPose]: List of loaded camera poses.
        """
        import os

        if not os.path.exists(filepath):
            raise ValidationError(f"File does not exist: {filepath}")

        data = np.loadtxt(filepath, delimiter=",", skiprows=1)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        poses = []
        for row in data:
            timestamp = row[0] if len(row) > 7 else None
            position = row[1:4]
            quaternion = row[4:8]
            pose = CameraPose(position, quaternion, timestamp)
            poses.append(pose)

        return poses
