from typing import Tuple

import numpy as np

from .validation_error import ProcessingError, ValidationError


class GeometryUtils:
    """Utility functions for geometric operations including rotations, quaternions, and transformations."""

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length.

        Args:
            vector (np.ndarray): Input vector of shape (N,) or (N, 3).

        Returns:
            np.ndarray: Normalized vector(s).

        Raises:
            ValidationError: If input is invalid.
            ProcessingError: If normalization fails (zero-length vector).
        """
        if not isinstance(vector, np.ndarray):
            raise ValidationError(f"vector must be a numpy array, got {type(vector)}")

        if vector.ndim == 1:
            # Single vector
            if vector.shape[0] != 3:
                raise ValidationError(
                    f"Vector must have 3 elements, got {vector.shape[0]}"
                )
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ProcessingError("Cannot normalize zero-length vector")
            result: np.ndarray = vector / norm
            return result

        elif vector.ndim == 2:
            # Multiple vectors
            if vector.shape[1] != 3:
                raise ValidationError(
                    f"Vectors must have shape (N, 3), got {vector.shape}"
                )
            norms = np.linalg.norm(vector, axis=1)
            if np.any(norms == 0):
                raise ProcessingError("Cannot normalize zero-length vectors")
            result: np.ndarray = vector / norms[:, np.newaxis]
            return result
        else:
            raise ValidationError(f"Vector must be 1D or 2D, got {vector.ndim}D")

    @staticmethod
    def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix from axis-angle representation.

        Args:
            axis (np.ndarray): Rotation axis (will be normalized).
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        if not isinstance(axis, np.ndarray) or axis.shape != (3,):
            raise ValidationError(
                f"axis must be a numpy array of shape (3,), got {axis.shape if isinstance(axis, np.ndarray) else type(axis)}"
            )

        if not np.all(np.isfinite(axis)):
            raise ValidationError("axis must contain only finite values")

        if not np.isfinite(angle):
            raise ValidationError(f"angle must be finite, got {angle}")

        axis = GeometryUtils.normalize_vector(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)

        result: np.ndarray = np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )
        return result

    @staticmethod
    def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion [w, x, y, z].

        Args:
            axis (np.ndarray): Rotation axis (will be normalized).
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Quaternion as [w, x, y, z].
        """
        if not isinstance(axis, np.ndarray) or axis.shape != (3,):
            raise ValidationError(
                f"axis must be a numpy array of shape (3,), got {axis.shape if isinstance(axis, np.ndarray) else type(axis)}"
            )

        if not np.isfinite(angle):
            raise ValidationError(f"angle must be finite, got {angle}")

        axis = GeometryUtils.normalize_vector(axis)
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)

        result: np.ndarray = np.array([w, xyz[0], xyz[1], xyz[2]])
        return result

    @staticmethod
    def axis_angle_from_quaternion(quaternion: np.ndarray) -> Tuple[np.ndarray, float]:
        """Convert quaternion [w, x, y, z] to axis-angle representation.

        Args:
            quaternion (np.ndarray): Quaternion as [w, x, y, z].

        Returns:
            Tuple[np.ndarray, float]: (axis, angle) where axis is normalized.
        """
        if not isinstance(quaternion, np.ndarray) or quaternion.shape != (4,):
            raise ValidationError(
                f"quaternion must be a numpy array of shape (4,), got {quaternion.shape if isinstance(quaternion, np.ndarray) else type(quaternion)}"
            )

        if not np.all(np.isfinite(quaternion)):
            raise ValidationError("quaternion must contain only finite values")

        # Normalize quaternion if needed
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ProcessingError("Cannot convert zero quaternion to axis-angle")
        if abs(norm - 1.0) > 1e-6:
            quaternion = quaternion / norm

        w, x, y, z = quaternion

        # Handle singularity at angle = 0
        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        if abs(angle) < 1e-6:
            return np.array([1.0, 0.0, 0.0]), 0.0

        sin_half_angle = np.sin(angle / 2.0)
        if abs(sin_half_angle) < 1e-6:
            return np.array([1.0, 0.0, 0.0]), np.pi

        axis = np.array([x, y, z]) / sin_half_angle
        axis = GeometryUtils.normalize_vector(axis)

        return axis, angle

    @staticmethod
    def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix.

        Args:
            quaternion (np.ndarray): Quaternion as [w, x, y, z].

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        if not isinstance(quaternion, np.ndarray) or quaternion.shape != (4,):
            raise ValidationError(
                f"quaternion must be a numpy array of shape (4,), got {quaternion.shape if isinstance(quaternion, np.ndarray) else type(quaternion)}"
            )

        if not np.all(np.isfinite(quaternion)):
            raise ValidationError("quaternion must contain only finite values")

        # Normalize quaternion if needed
        norm = np.linalg.norm(quaternion)
        if norm == 0:
            raise ProcessingError("Cannot convert zero quaternion to rotation matrix")
        if abs(norm - 1.0) > 1e-6:
            quaternion = quaternion / norm

        w, x, y, z = quaternion

        result: np.ndarray = np.array(
            [
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                ],
                [
                    2 * x * y + 2 * w * z,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * w * x,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    1 - 2 * x * x - 2 * y * y,
                ],
            ]
        )
        return result

    @staticmethod
    def quaternion_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z].

        Args:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.

        Returns:
            np.ndarray: Quaternion as [w, x, y, z].
        """
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (
            3,
            3,
        ):
            raise ValidationError(
                f"rotation_matrix must be a numpy array of shape (3, 3), got {rotation_matrix.shape if isinstance(rotation_matrix, np.ndarray) else type(rotation_matrix)}"
            )

        if not np.all(np.isfinite(rotation_matrix)):
            raise ValidationError("rotation_matrix must contain only finite values")

        # Check if it's a valid rotation matrix (orthogonal with det=1)
        if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-6):
            raise ValidationError("rotation_matrix is not orthogonal")
        if abs(np.linalg.det(rotation_matrix) - 1.0) > 1e-6:
            raise ValidationError("rotation_matrix determinant is not 1")

        trace = np.trace(rotation_matrix)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if (
                rotation_matrix[0, 0] > rotation_matrix[1, 1]
                and rotation_matrix[0, 0] > rotation_matrix[2, 2]
            ):
                s = 2.0 * np.sqrt(
                    1.0
                    + rotation_matrix[0, 0]
                    - rotation_matrix[1, 1]
                    - rotation_matrix[2, 2]
                )
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(
                    1.0
                    + rotation_matrix[1, 1]
                    - rotation_matrix[0, 0]
                    - rotation_matrix[2, 2]
                )
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(
                    1.0
                    + rotation_matrix[2, 2]
                    - rotation_matrix[0, 0]
                    - rotation_matrix[1, 1]
                )
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s

        result: np.ndarray = np.array([w, x, y, z])
        return result

    @staticmethod
    def euler_angles_from_rotation_matrix(
        rotation_matrix: np.ndarray, convention: str = "ZYX"
    ) -> np.ndarray:
        """Convert rotation matrix to Euler angles.

        Args:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
            convention (str): Euler angle convention ('ZYX', 'ZYZ', etc.). Defaults to 'ZYX'.

        Returns:
            np.ndarray: Euler angles in radians [roll, pitch, yaw] for ZYX convention.
        """
        if not isinstance(rotation_matrix, np.ndarray) or rotation_matrix.shape != (
            3,
            3,
        ):
            raise ValidationError(
                f"rotation_matrix must be a numpy array of shape (3, 3), got {rotation_matrix.shape if isinstance(rotation_matrix, np.ndarray) else type(rotation_matrix)}"
            )

        if convention != "ZYX":
            raise ValidationError(
                f"Only 'ZYX' convention is currently supported, got {convention}"
            )

        # For ZYX convention (yaw, pitch, roll)
        # Check for gimbal lock
        if abs(rotation_matrix[2, 0]) >= 1.0 - 1e-6:
            # Gimbal lock case
            yaw = 0.0
            if rotation_matrix[2, 0] < 0:  # r31 == -1
                pitch = np.pi / 2.0
                roll = yaw + np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
            else:  # r31 == 1
                pitch = -np.pi / 2.0
                roll = -yaw + np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])
        else:
            pitch = -np.arcsin(rotation_matrix[2, 0])
            cos_pitch = np.cos(pitch)
            roll = np.arctan2(
                rotation_matrix[2, 1] / cos_pitch, rotation_matrix[2, 2] / cos_pitch
            )
            yaw = np.arctan2(
                rotation_matrix[1, 0] / cos_pitch, rotation_matrix[0, 0] / cos_pitch
            )

        return np.array([roll, pitch, yaw])

    @staticmethod
    def rotation_matrix_from_euler_angles(
        euler_angles: np.ndarray, convention: str = "ZYX"
    ) -> np.ndarray:
        """Convert Euler angles to rotation matrix.

        Args:
            euler_angles (np.ndarray): Euler angles in radians [roll, pitch, yaw] for ZYX convention.
            convention (str): Euler angle convention. Defaults to 'ZYX'.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        if not isinstance(euler_angles, np.ndarray) or euler_angles.shape != (3,):
            raise ValidationError(
                f"euler_angles must be a numpy array of shape (3,), got {euler_angles.shape if isinstance(euler_angles, np.ndarray) else type(euler_angles)}"
            )

        if not np.all(np.isfinite(euler_angles)):
            raise ValidationError("euler_angles must contain only finite values")

        if convention != "ZYX":
            raise ValidationError(
                f"Only 'ZYX' convention is currently supported, got {convention}"
            )

        roll, pitch, yaw = euler_angles

        # Rotation matrices for each axis
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        Rz = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        # For ZYX convention: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.

        Args:
            q1 (np.ndarray): First quaternion [w, x, y, z].
            q2 (np.ndarray): Second quaternion [w, x, y, z].

        Returns:
            np.ndarray: Result quaternion [w, x, y, z].
        """
        if not isinstance(q1, np.ndarray) or q1.shape != (4,):
            raise ValidationError(
                f"q1 must be a numpy array of shape (4,), got {q1.shape if isinstance(q1, np.ndarray) else type(q1)}"
            )
        if not isinstance(q2, np.ndarray) or q2.shape != (4,):
            raise ValidationError(
                f"q2 must be a numpy array of shape (4,), got {q2.shape if isinstance(q2, np.ndarray) else type(q2)}"
            )

        if not np.all(np.isfinite(q1)) or not np.all(np.isfinite(q2)):
            raise ValidationError("quaternions must contain only finite values")

        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate.

        Args:
            quaternion (np.ndarray): Quaternion [w, x, y, z].

        Returns:
            np.ndarray: Conjugate quaternion [w, -x, -y, -z].
        """
        if not isinstance(quaternion, np.ndarray) or quaternion.shape != (4,):
            raise ValidationError(
                f"quaternion must be a numpy array of shape (4,), got {quaternion.shape if isinstance(quaternion, np.ndarray) else type(quaternion)}"
            )

        result: np.ndarray = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
        )
        return result

    @staticmethod
    def quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse.

        Args:
            quaternion (np.ndarray): Quaternion [w, x, y, z].

        Returns:
            np.ndarray: Inverse quaternion.
        """
        if not isinstance(quaternion, np.ndarray) or quaternion.shape != (4,):
            raise ValidationError(
                f"quaternion must be a numpy array of shape (4,), got {quaternion.shape if isinstance(quaternion, np.ndarray) else type(quaternion)}"
            )

        norm_squared = np.sum(quaternion**2)
        if norm_squared == 0:
            raise ProcessingError("Cannot invert zero quaternion")

        result: np.ndarray = (
            GeometryUtils.quaternion_conjugate(quaternion) / norm_squared
        )
        return result

    @staticmethod
    def transform_points(
        points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
    ) -> np.ndarray:
        """Transform 3D points using rotation and translation.

        Args:
            points (np.ndarray): Points to transform, shape (N, 3).
            rotation (np.ndarray): Rotation matrix (3, 3) or quaternion (4,).
            translation (np.ndarray): Translation vector (3,).

        Returns:
            np.ndarray: Transformed points, shape (N, 3).
        """
        if (
            not isinstance(points, np.ndarray)
            or points.ndim != 2
            or points.shape[1] != 3
        ):
            raise ValidationError(
                f"points must be a numpy array of shape (N, 3), got {points.shape if isinstance(points, np.ndarray) else type(points)}"
            )

        if not isinstance(translation, np.ndarray) or translation.shape != (3,):
            raise ValidationError(
                f"translation must be a numpy array of shape (3,), got {translation.shape if isinstance(translation, np.ndarray) else type(translation)}"
            )

        if isinstance(rotation, np.ndarray):
            if rotation.shape == (3, 3):
                # Rotation matrix
                result: np.ndarray = (rotation @ points.T).T + translation
                return result
            elif rotation.shape == (4,):
                # Quaternion
                rot_matrix = GeometryUtils.rotation_matrix_from_quaternion(rotation)
                result: np.ndarray = (rot_matrix @ points.T).T + translation
                return result
            else:
                raise ValidationError(
                    f"rotation must be shape (3, 3) or (4,), got {rotation.shape}"
                )
        else:
            raise ValidationError(
                f"rotation must be a numpy array, got {type(rotation)}"
            )

    @staticmethod
    def validate_rotation_matrix(matrix: np.ndarray, atol: float = 1e-6) -> bool:
        """Validate if a matrix is a proper rotation matrix.

        Args:
            matrix (np.ndarray): Matrix to validate.
            atol (float): Absolute tolerance for validation.

        Returns:
            bool: True if valid rotation matrix.
        """
        if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
            return False

        # Check orthogonality
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=atol):
            return False

        # Check determinant is 1
        if abs(np.linalg.det(matrix) - 1.0) > atol:
            return False

        return True

    @staticmethod
    def validate_quaternion(quaternion: np.ndarray, atol: float = 1e-6) -> bool:
        """Validate if array represents a unit quaternion.

        Args:
            quaternion (np.ndarray): Quaternion to validate.
            atol (float): Absolute tolerance for norm check.

        Returns:
            bool: True if valid unit quaternion.
        """
        if not isinstance(quaternion, np.ndarray) or quaternion.shape != (4,):
            return False

        if not np.all(np.isfinite(quaternion)):
            return False

        norm = np.linalg.norm(quaternion)
        result: bool = abs(norm - 1.0) <= atol
        return result
