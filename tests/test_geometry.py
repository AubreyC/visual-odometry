from typing import Union

import numpy as np
import pytest

from src.camera import ProcessingError, ValidationError
from src.geometry import GeometryUtils


class TestGeometryUtils:
    """Test suite for GeometryUtils class."""

    def test_normalize_vector_single_vector(self) -> None:
        """Test normalizing a single 3D vector."""
        vector = np.array([3.0, 4.0, 0.0])
        result = GeometryUtils.normalize_vector(vector)
        expected = np.array([3.0 / 5.0, 4.0 / 5.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_vector_multiple_vectors(self) -> None:
        """Test normalizing multiple 3D vectors."""
        vectors = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
        result = GeometryUtils.normalize_vector(vectors)
        expected = np.array([[3.0 / 5.0, 4.0 / 5.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_vector_zero_length(self) -> None:
        """Test normalizing zero-length vector raises ProcessingError."""
        vector = np.array([0.0, 0.0, 0.0])
        with pytest.raises(
            ProcessingError, match="Cannot normalize zero-length vector"
        ):
            GeometryUtils.normalize_vector(vector)

    def test_normalize_vector_invalid_input(self) -> None:
        """Test normalizing invalid input raises ValidationError."""
        with pytest.raises(ValidationError, match="vector must be a numpy array"):
            GeometryUtils.normalize_vector([1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_normalize_vector_wrong_shape(self) -> None:
        """Test normalizing vector with wrong shape raises ValidationError."""
        vector = np.array([1.0, 2.0])  # 2 elements instead of 3
        with pytest.raises(ValidationError, match="Vector must have 3 elements"):
            GeometryUtils.normalize_vector(vector)

    def test_rotation_matrix_from_axis_angle(self) -> None:
        """Test creating rotation matrix from axis-angle."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2.0
        result = GeometryUtils.rotation_matrix_from_axis_angle(axis, angle)

        # The result should be a valid rotation matrix
        assert GeometryUtils.validate_rotation_matrix(result)

        # Test that it produces the expected rotation: (1,0,0) -> (0,-1,0) for this convention
        test_point = np.array([1.0, 0.0, 0.0])
        rotated = result @ test_point
        expected_rotated = np.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(rotated, expected_rotated, atol=1e-10)

    def test_quaternion_from_axis_angle(self) -> None:
        """Test creating quaternion from axis-angle."""
        axis = np.array([0.0, 0.0, 1.0])
        angle = np.pi / 2.0
        result = GeometryUtils.quaternion_from_axis_angle(axis, angle)
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_axis_angle_from_quaternion(self) -> None:
        """Test extracting axis-angle from quaternion."""
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        axis, angle = GeometryUtils.axis_angle_from_quaternion(quaternion)
        expected_axis = np.array([0.0, 0.0, 1.0])
        expected_angle = np.pi / 2.0
        np.testing.assert_array_almost_equal(axis, expected_axis)
        assert abs(angle - expected_angle) < 1e-10

    def test_axis_angle_from_quaternion_zero_rotation(self) -> None:
        """Test axis-angle extraction for identity quaternion."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        axis, angle = GeometryUtils.axis_angle_from_quaternion(quaternion)
        assert abs(angle) < 1e-6

    def test_axis_angle_from_quaternion_pi_rotation(self) -> None:
        """Test axis-angle extraction for 180-degree rotation."""
        quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around x-axis
        axis, angle = GeometryUtils.axis_angle_from_quaternion(quaternion)
        expected_axis = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(axis, expected_axis)
        assert abs(angle - np.pi) < 1e-10

    def test_rotation_matrix_from_quaternion(self) -> None:
        """Test creating rotation matrix from quaternion."""
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        result = GeometryUtils.rotation_matrix_from_quaternion(quaternion)
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_quaternion_from_rotation_matrix(self) -> None:
        """Test creating quaternion from rotation matrix."""
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = GeometryUtils.quaternion_from_rotation_matrix(rotation_matrix)
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_euler_angles_from_rotation_matrix(self) -> None:
        """Test extracting Euler angles from rotation matrix."""
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        result = GeometryUtils.euler_angles_from_rotation_matrix(rotation_matrix)
        expected = np.array([0.0, 0.0, np.pi / 2.0])  # roll, pitch, yaw
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_matrix_from_euler_angles(self) -> None:
        """Test creating rotation matrix from Euler angles."""
        euler_angles = np.array([0.0, 0.0, np.pi / 2.0])  # roll, pitch, yaw
        result = GeometryUtils.rotation_matrix_from_euler_angles(euler_angles)
        expected = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_quaternion_from_euler_angles(self) -> None:
        """Test creating quaternion from Euler angles."""
        euler_angles = np.array([0.0, 0.0, np.pi / 2.0])  # roll, pitch, yaw
        result = GeometryUtils.quaternion_from_euler_angles(euler_angles)
        expected = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        np.testing.assert_array_almost_equal(result, expected)

    def test_euler_angles_from_quaternion(self) -> None:
        """Test extracting Euler angles from quaternion."""
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        result = GeometryUtils.euler_angles_from_quaternion(quaternion)
        expected = np.array([0.0, 0.0, np.pi / 2.0])  # roll, pitch, yaw
        np.testing.assert_array_almost_equal(result, expected)

    def test_quaternion_euler_round_trip(self) -> None:
        """Test round-trip conversion between quaternion and Euler angles."""
        original_euler = np.array([0.1, 0.2, 0.3])
        quaternion = GeometryUtils.quaternion_from_euler_angles(original_euler)
        result_euler = GeometryUtils.euler_angles_from_quaternion(quaternion)
        np.testing.assert_array_almost_equal(result_euler, original_euler, decimal=10)

    def test_quaternion_multiply(self) -> None:
        """Test quaternion multiplication."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        q2 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # 90° rotation
        result = GeometryUtils.quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, q2)

    def test_quaternion_conjugate(self) -> None:
        """Test quaternion conjugation."""
        quaternion = np.array([1.0, 2.0, 3.0, 4.0])
        result = GeometryUtils.quaternion_conjugate(quaternion)
        expected = np.array([1.0, -2.0, -3.0, -4.0])
        np.testing.assert_array_equal(result, expected)

    def test_quaternion_inverse(self) -> None:
        """Test quaternion inversion."""
        quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # 180° rotation
        result = GeometryUtils.quaternion_inverse(quaternion)
        expected = np.array(
            [0.0, -1.0, 0.0, 0.0]
        )  # same as conjugate for unit quaternion
        np.testing.assert_array_almost_equal(result, expected)

    def test_quaternion_inverse_zero_quaternion(self) -> None:
        """Test quaternion inversion of zero quaternion raises ProcessingError."""
        quaternion = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ProcessingError, match="Cannot invert zero quaternion"):
            GeometryUtils.quaternion_inverse(quaternion)

    def test_transform_points_with_rotation_matrix(self) -> None:
        """Test transforming points with rotation matrix."""
        points = np.array([[1.0, 0.0, 0.0]])
        rotation = np.eye(3)  # identity
        translation = np.array([1.0, 2.0, 3.0])
        result = GeometryUtils.transform_points(points, rotation, translation)
        expected = np.array([[2.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_points_with_quaternion(self) -> None:
        """Test transforming points with quaternion."""
        points = np.array([[1.0, 0.0, 0.0]])
        rotation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        translation = np.array([1.0, 2.0, 3.0])
        result = GeometryUtils.transform_points(points, rotation, translation)
        expected = np.array([[2.0, 2.0, 3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_validate_rotation_matrix_valid(self) -> None:
        """Test validating a valid rotation matrix."""
        rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert GeometryUtils.validate_rotation_matrix(rotation_matrix)

    def test_validate_rotation_matrix_invalid_shape(self) -> None:
        """Test validating rotation matrix with invalid shape."""
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 instead of 3x3
        assert not GeometryUtils.validate_rotation_matrix(matrix)

    def test_validate_rotation_matrix_not_orthogonal(self) -> None:
        """Test validating rotation matrix that is not orthogonal."""
        matrix = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert not GeometryUtils.validate_rotation_matrix(matrix)

    def test_validate_quaternion_valid(self) -> None:
        """Test validating a valid unit quaternion."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        assert GeometryUtils.validate_quaternion(quaternion)

    def test_validate_quaternion_invalid_shape(self) -> None:
        """Test validating quaternion with invalid shape."""
        quaternion = np.array([1.0, 0.0, 0.0])  # 3 elements instead of 4
        assert not GeometryUtils.validate_quaternion(quaternion)

    def test_validate_quaternion_not_unit(self) -> None:
        """Test validating quaternion that is not unit length."""
        quaternion = np.array([2.0, 0.0, 0.0, 0.0])  # norm = 2
        assert not GeometryUtils.validate_quaternion(quaternion)

    def test_round_trip_conversions(self) -> None:
        """Test round-trip conversions maintain consistency."""
        # Start with axis-angle
        original_axis = np.array([1.0, 1.0, 1.0])
        original_angle = np.pi / 3.0

        # Convert to quaternion
        quaternion = GeometryUtils.quaternion_from_axis_angle(
            original_axis, original_angle
        )

        # Convert to rotation matrix
        rotation_matrix = GeometryUtils.rotation_matrix_from_quaternion(quaternion)

        # Convert back to quaternion
        quaternion2 = GeometryUtils.quaternion_from_rotation_matrix(rotation_matrix)

        # Convert back to axis-angle
        final_axis, final_angle = GeometryUtils.axis_angle_from_quaternion(quaternion2)

        # Check consistency
        np.testing.assert_array_almost_equal(np.abs(quaternion), np.abs(quaternion2))
        assert abs(original_angle - final_angle) < 1e-10

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "not_an_array",
            np.array([1.0, 2.0]),  # wrong shape
            np.array([float("inf"), 0.0, 1.0]),
            np.array([float("nan"), 0.0, 1.0]),
        ],
    )
    def test_rotation_matrix_from_axis_angle_invalid_input(
        self, invalid_input: Union[str, np.ndarray]
    ) -> None:
        """Test invalid inputs for rotation matrix from axis-angle."""
        if isinstance(invalid_input, str):
            with pytest.raises(ValidationError):
                GeometryUtils.rotation_matrix_from_axis_angle(invalid_input, np.pi / 2)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValidationError):
                GeometryUtils.rotation_matrix_from_axis_angle(invalid_input, np.pi / 2)

    @pytest.mark.parametrize(
        "invalid_angle", [float("inf"), float("-inf"), float("nan")]
    )
    def test_rotation_matrix_from_axis_angle_invalid_angle(
        self, invalid_angle: float
    ) -> None:
        """Test invalid angles for rotation matrix from axis-angle."""
        axis = np.array([0.0, 0.0, 1.0])
        with pytest.raises(ValidationError, match="angle must be finite"):
            GeometryUtils.rotation_matrix_from_axis_angle(axis, invalid_angle)

    @pytest.mark.parametrize(
        "invalid_input",
        [
            "not_an_array",
            np.array([1.0, 2.0, 3.0]),  # wrong shape (3 elements instead of 4)
            np.array([float("inf"), 0.0, 0.0, 1.0]),
            np.array([float("nan"), 0.0, 0.0, 1.0]),
        ],
    )
    def test_rotation_matrix_from_quaternion_invalid_input(
        self, invalid_input: Union[str, np.ndarray]
    ) -> None:
        """Test invalid inputs for rotation matrix from quaternion."""
        if isinstance(invalid_input, str):
            with pytest.raises(ValidationError):
                GeometryUtils.rotation_matrix_from_quaternion(invalid_input)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValidationError):
                GeometryUtils.rotation_matrix_from_quaternion(invalid_input)

    @pytest.mark.parametrize(
        "invalid_matrix",
        [
            "not_an_array",
            np.array([[1.0, 0.0], [0.0, 1.0]]),  # wrong shape
            np.array(
                [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]
            ),  # 3D array
            np.array([[float("inf"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([[float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ],
    )
    def test_quaternion_from_rotation_matrix_invalid_input(
        self, invalid_matrix: Union[str, np.ndarray]
    ) -> None:
        """Test invalid inputs for quaternion from rotation matrix."""
        if isinstance(invalid_matrix, str):
            with pytest.raises(ValidationError):
                GeometryUtils.quaternion_from_rotation_matrix(invalid_matrix)  # type: ignore[arg-type]
        else:
            with pytest.raises(ValidationError):
                GeometryUtils.quaternion_from_rotation_matrix(invalid_matrix)

    def test_rotation_matrix_not_orthogonal(self) -> None:
        """Test quaternion from non-orthogonal matrix raises ValidationError."""
        matrix = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValidationError, match="rotation_matrix is not orthogonal"):
            GeometryUtils.quaternion_from_rotation_matrix(matrix)

    def test_rotation_matrix_wrong_determinant(self) -> None:
        """Test quaternion from matrix with wrong determinant raises ValidationError."""
        matrix = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(
            ValidationError, match="rotation_matrix determinant is not 1"
        ):
            GeometryUtils.quaternion_from_rotation_matrix(matrix)
