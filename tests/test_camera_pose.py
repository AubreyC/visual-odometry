import os
import tempfile

import numpy as np
import pytest

from src.camera import ValidationError
from src.camera_pose import CameraPose, TrajectoryGenerator


class TestCameraPose:
    """Test suite for CameraPose class."""

    def test_valid_initialization(self) -> None:
        """Test valid camera pose initialization."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        timestamp = 1.5

        pose = CameraPose(position, quaternion, timestamp)

        assert np.allclose(pose.position, position)
        assert np.allclose(pose.quaternion, quaternion)
        assert pose.timestamp == timestamp

    def test_initialization_without_timestamp(self) -> None:
        """Test camera pose initialization without timestamp."""
        position = np.array([1.0, 2.0, 3.0])
        # 90 degree rotation around Z axis (properly normalized quaternion)
        quaternion = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

        pose = CameraPose(position, quaternion)

        assert np.allclose(pose.position, position)
        assert np.allclose(pose.quaternion, quaternion)
        assert pose.timestamp == 0.0

    @pytest.mark.parametrize(
        "invalid_position",
        [
            [1.0, 2.0, 3.0],  # Python list instead of numpy array
            np.array([1.0, 2.0]),  # Wrong shape
            np.array([1.0, 2.0, 3.0, 4.0]),  # Wrong shape
            np.array([[1.0], [2.0], [3.0]]),  # Wrong shape
        ],
    )
    def test_invalid_position(self, invalid_position: np.ndarray) -> None:
        """Test invalid position raises ValidationError."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(
            ValidationError, match="position must be a numpy array of shape"
        ):
            CameraPose(invalid_position, quaternion)

    @pytest.mark.parametrize(
        "invalid_position",
        [
            np.array([float("inf"), 2.0, 3.0]),
            np.array([1.0, float("-inf"), 3.0]),
            np.array([1.0, 2.0, float("nan")]),
        ],
    )
    def test_non_finite_position(self, invalid_position: np.ndarray) -> None:
        """Test non-finite position values raise ValidationError."""
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(
            ValidationError, match="position must contain only finite values"
        ):
            CameraPose(invalid_position, quaternion)

    @pytest.mark.parametrize(
        "invalid_quaternion",
        [
            [1.0, 0.0, 0.0, 0.0],  # Python list instead of numpy array
            np.array([1.0, 0.0, 0.0]),  # Wrong shape (3 elements)
            np.array([1.0, 0.0, 0.0, 0.0, 1.0]),  # Wrong shape (5 elements)
            np.array([[1.0], [0.0], [0.0], [0.0]]),  # Wrong shape (4, 1)
        ],
    )
    def test_invalid_quaternion_shape(self, invalid_quaternion: np.ndarray) -> None:
        """Test invalid quaternion shape raises ValidationError."""
        position = np.array([1.0, 2.0, 3.0])
        with pytest.raises(
            ValidationError, match="orientation must be a numpy array of shape"
        ):
            CameraPose(position, invalid_quaternion)

    @pytest.mark.parametrize(
        "invalid_quaternion",
        [
            np.array([float("inf"), 0.0, 0.0, 0.0]),
            np.array([1.0, float("-inf"), 0.0, 0.0]),
            np.array([1.0, 0.0, float("nan"), 0.0]),
        ],
    )
    def test_non_finite_quaternion(self, invalid_quaternion: np.ndarray) -> None:
        """Test non-finite quaternion values raise ValidationError."""
        position = np.array([1.0, 2.0, 3.0])
        with pytest.raises(
            ValidationError, match="orientation quaternion is not valid"
        ):
            CameraPose(position, invalid_quaternion)

    @pytest.mark.parametrize(
        "invalid_quaternion",
        [
            np.array([0.0, 0.0, 0.0, 0.0]),  # Zero quaternion
            np.array([2.0, 0.0, 0.0, 0.0]),  # Non-unit quaternion
            np.array(
                [0.6, 0.6, 0.6, 0.6]
            ),  # Non-unit quaternion (norm = sqrt(1.44) ≈ 1.2)
        ],
    )
    def test_invalid_unit_quaternion(self, invalid_quaternion: np.ndarray) -> None:
        """Test invalid (non-unit) quaternions raise ValidationError."""
        position = np.array([1.0, 2.0, 3.0])
        with pytest.raises(
            ValidationError, match="orientation quaternion is not valid"
        ):
            CameraPose(position, invalid_quaternion)

    def test_rotation_matrix_property(self) -> None:
        """Test rotation matrix property returns correct 3x3 matrix."""
        position = np.array([1.0, 2.0, 3.0])

        # Identity quaternion should give identity rotation matrix
        identity_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        pose = CameraPose(position, identity_quaternion)
        rotation_matrix = pose.rotation_matrix

        assert rotation_matrix.shape == (3, 3)
        assert np.allclose(rotation_matrix, np.eye(3))

        # 90 degree rotation around Z axis
        z_rotation_quaternion = np.array(
            [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        )
        pose_rotated = CameraPose(position, z_rotation_quaternion)
        rotation_matrix_rotated = pose_rotated.rotation_matrix

        expected_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        assert np.allclose(rotation_matrix_rotated, expected_rotation, atol=1e-4)

    def test_quaternion_property(self) -> None:
        """Test quaternion property returns copy of quaternion."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array(
            [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
        )  # 90 degree rotation around Y
        pose = CameraPose(position, quaternion)

        returned_quaternion = pose.quaternion
        assert np.allclose(returned_quaternion, quaternion)
        assert returned_quaternion is not quaternion  # Should be a copy

    def test_transform_points_world_to_camera(self) -> None:
        """Test transforming points from world to camera coordinates."""
        # Camera at origin with identity orientation
        position = np.array([0.0, 0.0, 0.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        pose = CameraPose(position, quaternion)

        world_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        camera_points = pose.transform_points_world_to_camera(world_points)

        # With identity pose, world and camera coordinates should be the same
        assert np.allclose(camera_points, world_points)

        # Camera translated by [1, 0, 0] with identity orientation
        position_translated = np.array([1.0, 0.0, 0.0])
        pose_translated = CameraPose(position_translated, quaternion)
        camera_points_translated = pose_translated.transform_points_world_to_camera(
            world_points
        )

        expected = world_points - position_translated  # Translation only (since R = I)
        assert np.allclose(camera_points_translated, expected)

    def test_transform_points_camera_to_world(self) -> None:
        """Test transforming points from camera to world coordinates."""
        # Camera at origin with identity orientation
        position = np.array([0.0, 0.0, 0.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        pose = CameraPose(position, quaternion)

        camera_points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        world_points = pose.transform_points_camera_to_world(camera_points)

        # With identity pose, camera and world coordinates should be the same
        assert np.allclose(world_points, camera_points)

        # Camera translated by [1, 0, 0] with identity orientation
        position_translated = np.array([1.0, 0.0, 0.0])
        pose_translated = CameraPose(position_translated, quaternion)
        world_points_translated = pose_translated.transform_points_camera_to_world(
            camera_points
        )

        expected = camera_points + position_translated  # Translation only
        assert np.allclose(world_points_translated, expected)

    def test_get_transform_matrix(self) -> None:
        """Test getting 4x4 transformation matrix."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        pose = CameraPose(position, quaternion)

        transform_matrix = pose.get_transform_matrix()

        assert transform_matrix.shape == (4, 4)
        assert np.allclose(transform_matrix[:3, :3], np.eye(3))  # Identity rotation
        assert np.allclose(transform_matrix[:3, 3], -position)  # Translation
        assert transform_matrix[3, 3] == 1.0

    @pytest.mark.parametrize(
        "invalid_matrix",
        [
            np.eye(3),  # Wrong shape
            np.eye(5),  # Wrong shape
            np.ones((4, 4)),  # Not proper transform matrix
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # Python list
        ],
    )
    def test_from_transform_matrix_invalid_input(
        self, invalid_matrix: np.ndarray
    ) -> None:
        """Test from_transform_matrix with invalid input raises ValidationError."""
        with pytest.raises(ValidationError):
            CameraPose.from_transform_matrix(invalid_matrix)

    def test_from_transform_matrix_roundtrip(self) -> None:
        """Test that from_transform_matrix and get_transform_matrix are inverses."""
        original_position = np.array([1.0, 2.0, 3.0])
        original_quaternion = np.array(
            [0.0, 1.0, 0.0, 0.0]
        )  # 180 degree rotation around X
        original_pose = CameraPose(original_position, original_quaternion)

        transform_matrix = original_pose.get_transform_matrix()
        reconstructed_pose = CameraPose.from_transform_matrix(transform_matrix)

        assert np.allclose(reconstructed_pose.position, original_position)
        assert np.allclose(
            reconstructed_pose.quaternion, original_quaternion, atol=1e-6
        )

    def test_string_representation(self) -> None:
        """Test string representation of CameraPose."""
        position = np.array([1.5, 2.5, 3.5])
        quaternion = np.array(
            [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
        )  # 90 degree rotation around Y
        timestamp = 1.234

        pose = CameraPose(position, quaternion, timestamp)
        repr_str = repr(pose)

        assert "CameraPose" in repr_str
        assert "1.500" in repr_str
        assert "2.500" in repr_str
        assert "3.500" in repr_str
        assert "0.707" in repr_str
        assert "1.234" in repr_str

    def test_string_representation_no_timestamp(self) -> None:
        """Test string representation without timestamp."""
        position = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        pose = CameraPose(position, quaternion)
        repr_str = repr(pose)

        assert "CameraPose" in repr_str
        assert "t" in repr_str

    def test_create_look_at_pose_basic(self) -> None:
        """Test basic look-at pose creation."""
        camera_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([1.0, 0.0, 0.0])
        timestamp = 1.5

        pose = CameraPose.create_look_at_target(camera_pos, target_pos, timestamp)

        assert np.allclose(pose.position, camera_pos)
        assert pose.timestamp == timestamp

        # Camera should face toward target (positive X direction)
        # With camera at origin looking toward [1,0,0], camera Z should be [1,0,0]
        # Camera X should be parallel to world X [1,0,0], but since camera Z is already [1,0,0],
        # it uses world Y [0,1,0] as camera X
        expected_camera_z = np.array([1.0, 0.0, 0.0])
        expected_camera_x = np.array(
            [0.0, 1.0, 0.0]
        )  # World Y since camera Z parallel to world X
        expected_camera_y = np.cross(expected_camera_z, expected_camera_x)

        rotation_matrix = pose.rotation_matrix
        assert np.allclose(
            rotation_matrix[:, 2], expected_camera_z, atol=1e-6
        )  # Z column
        assert np.allclose(
            rotation_matrix[:, 0], expected_camera_x, atol=1e-6
        )  # X column
        assert np.allclose(
            rotation_matrix[:, 1], expected_camera_y, atol=1e-6
        )  # Y column

    def test_create_look_at_pose_diagonal(self) -> None:
        """Test look-at pose with diagonal target direction."""
        camera_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([1.0, 1.0, 1.0])

        pose = CameraPose.create_look_at_target(camera_pos, target_pos)

        # Camera Z should point toward normalized target direction
        expected_direction = np.array([1.0, 1.0, 1.0]) / np.linalg.norm([1.0, 1.0, 1.0])
        assert np.allclose(pose.rotation_matrix[:, 2], expected_direction, atol=1e-6)

        # Camera X should be projection of world X onto plane perpendicular to camera Z
        world_x = np.array([1.0, 0.0, 0.0])
        camera_z = expected_direction
        camera_x = world_x - np.dot(world_x, camera_z) * camera_z
        camera_x = camera_x / np.linalg.norm(camera_x)
        assert np.allclose(pose.rotation_matrix[:, 0], camera_x, atol=1e-6)

    @pytest.mark.parametrize(
        "invalid_pos",
        [
            [1.0, 2.0, 3.0],  # Python list instead of numpy array
            np.array([1.0, 2.0]),  # Wrong shape
            np.array([1.0, 2.0, 3.0, 4.0]),  # Wrong shape
            np.array([[1.0], [2.0], [3.0]]),  # Wrong shape
        ],
    )
    def test_create_look_at_pose_invalid_positions(
        self, invalid_pos: np.ndarray
    ) -> None:
        """Test create_look_at_pose with invalid position inputs."""
        valid_pos = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValidationError, match="must be a 3D numpy array"):
            CameraPose.create_look_at_target(invalid_pos, valid_pos)

        with pytest.raises(ValidationError, match="must be a 3D numpy array"):
            CameraPose.create_look_at_target(valid_pos, invalid_pos)

    @pytest.mark.parametrize(
        "invalid_pos",
        [
            np.array([float("inf"), 0.0, 0.0]),
            np.array([0.0, float("-inf"), 0.0]),
            np.array([0.0, 0.0, float("nan")]),
        ],
    )
    def test_create_look_at_pose_non_finite_positions(
        self, invalid_pos: np.ndarray
    ) -> None:
        """Test create_look_at_pose with non-finite position values."""
        valid_pos = np.array([0.0, 0.0, 0.0])

        with pytest.raises(ValidationError, match="must contain finite values"):
            CameraPose.create_look_at_target(invalid_pos, valid_pos)

        with pytest.raises(ValidationError, match="must contain finite values"):
            CameraPose.create_look_at_target(valid_pos, invalid_pos)

    def test_create_look_at_pose_same_position(self) -> None:
        """Test create_look_at_pose with same camera and target positions."""
        position = np.array([1.0, 2.0, 3.0])

        with pytest.raises(
            ValidationError,
            match="Camera position and target position cannot be the same",
        ):
            CameraPose.create_look_at_target(position, position)


class TestTrajectoryGenerator:
    """Test suite for TrajectoryGenerator class."""

    def test_initialization(self) -> None:
        """Test TrajectoryGenerator initialization."""
        generator = TrajectoryGenerator(time_step=0.1)
        assert generator.time_step == 0.1

        # Test default time step
        generator_default = TrajectoryGenerator()
        assert generator_default.time_step == 0.1

    @pytest.mark.parametrize(
        "invalid_time_step", [-0.1, 0.0, float("inf"), float("nan")]
    )
    def test_invalid_time_step(self, invalid_time_step: float) -> None:
        """Test invalid time step raises ValidationError."""
        with pytest.raises(
            ValidationError, match="time_step must be a positive finite number"
        ):
            TrajectoryGenerator(time_step=invalid_time_step)

    def test_generate_circular_trajectory_look_at_center(self) -> None:
        """Test circular trajectory generation with look_at_center=True."""
        generator = TrajectoryGenerator(time_step=0.1)
        center = np.array([0.0, 0.0])
        poses = generator.generate_circular_trajectory(
            center=center, radius=2.0, height=1.0, num_poses=4, look_at_center=True
        )

        assert len(poses) == 4
        for i, pose in enumerate(poses):
            assert isinstance(pose, CameraPose)
            assert pose.timestamp == i * 0.1

            # Check that camera is at correct height
            assert abs(pose.position[2] - 1.0) < 1e-10

            # Check that camera is at correct distance from center (on circle)
            distance_from_center = np.linalg.norm(pose.position[:2] - center)
            assert abs(distance_from_center - 2.0) < 1e-10

    def test_generate_circular_trajectory_tangent(self) -> None:
        """Test circular trajectory generation with look_at_center=False."""
        generator = TrajectoryGenerator(time_step=0.1)
        center = np.array([0.0, 0.0])
        poses = generator.generate_circular_trajectory(
            center=center, radius=2.0, height=1.0, num_poses=4, look_at_center=False
        )

        assert len(poses) == 4
        for pose in poses:
            assert isinstance(pose, CameraPose)
            # Check that camera is at correct height and radius
            assert abs(pose.position[2] - 1.0) < 1e-10
            distance_from_center = np.linalg.norm(pose.position[:2] - center)
            assert abs(distance_from_center - 2.0) < 1e-10

    def test_generate_linear_trajectory_with_orientation(self) -> None:
        """Test linear trajectory generation with explicit orientation."""
        generator = TrajectoryGenerator(time_step=0.1)
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 0.0, 0.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        poses = generator.generate_linear_trajectory(
            start_pos, end_pos, num_poses=3, orientation=orientation
        )

        assert len(poses) == 3
        assert np.allclose(poses[0].position, start_pos)
        assert np.allclose(poses[2].position, end_pos)
        assert np.allclose(poses[1].position, [0.5, 0.0, 0.0])  # Midpoint

        # All poses should have the same orientation
        for pose in poses:
            assert np.allclose(pose.quaternion, orientation)

    def test_generate_linear_trajectory_auto_orientation(self) -> None:
        """Test linear trajectory generation with automatic orientation."""
        generator = TrajectoryGenerator(time_step=0.1)
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 0.0, 0.0])

        poses = generator.generate_linear_trajectory(start_pos, end_pos, num_poses=2)

        assert len(poses) == 2
        # Camera should face in the direction of motion (positive X)
        # This should result in identity quaternion (no rotation needed)
        assert np.allclose(poses[0].quaternion, [1.0, 0.0, 0.0, 0.0], atol=1e-4)

    def test_generate_figure_eight_trajectory(self) -> None:
        """Test figure-eight trajectory generation."""
        generator = TrajectoryGenerator(time_step=0.1)
        center = np.array([0.0, 0.0])
        poses = generator.generate_figure_eight_trajectory(
            center=center, width=2.0, height=1.0, z_level=1.5, num_poses=5
        )

        assert len(poses) == 5
        for pose in poses:
            assert isinstance(pose, CameraPose)
            assert abs(pose.position[2] - 1.5) < 1e-10  # Z level

    def test_save_and_load_trajectory(self) -> None:
        """Test saving and loading trajectory to/from CSV."""
        # Create some test poses
        poses = [
            CameraPose(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
            CameraPose(
                np.array([1.0, 0.0, 0.0]),
                np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]),
                0.1,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_trajectory.csv")

            # Save trajectory
            generator = TrajectoryGenerator()
            generator.save_trajectory(poses, filepath)

            # Load trajectory
            loaded_poses = TrajectoryGenerator.load_trajectory(filepath)

            assert len(loaded_poses) == len(poses)
            for original, loaded in zip(poses, loaded_poses):
                assert np.allclose(original.position, loaded.position)
                assert np.allclose(original.quaternion, loaded.quaternion)
                assert original.timestamp == loaded.timestamp

    def test_save_trajectory_invalid_filepath(self) -> None:
        """Test saving trajectory with invalid filepath raises ValidationError."""
        poses = [CameraPose(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))]
        generator = TrajectoryGenerator()

        with pytest.raises(ValidationError, match="filepath must end with .csv"):
            generator.save_trajectory(poses, "invalid_path.txt")

    def test_load_trajectory_invalid_filepath(self) -> None:
        """Test loading trajectory with invalid filepath raises ValidationError."""
        with pytest.raises(ValidationError, match="File does not exist"):
            TrajectoryGenerator.load_trajectory("nonexistent_file.csv")

    @pytest.mark.parametrize(
        "invalid_center", [[0.0, 0.0], np.array([0.0]), np.array([0.0, 0.0, 0.0])]
    )
    def test_circular_trajectory_invalid_center(
        self, invalid_center: np.ndarray
    ) -> None:
        """Test circular trajectory with invalid center raises ValidationError."""
        generator = TrajectoryGenerator()
        with pytest.raises(ValidationError, match="center must be a 2D numpy array"):
            generator.generate_circular_trajectory(
                center=invalid_center, radius=1.0, height=1.0, num_poses=5
            )

    @pytest.mark.parametrize(
        "invalid_radius,expected_error",
        [
            (-1.0, "radius must be positive"),
            (0.0, "radius must be positive"),
            (float("inf"), "angle must be finite"),  # inf radius creates nan angles
        ],
    )
    def test_circular_trajectory_invalid_radius(
        self, invalid_radius: float, expected_error: str
    ) -> None:
        """Test circular trajectory with invalid radius raises ValidationError."""
        generator = TrajectoryGenerator()
        center = np.array([0.0, 0.0])
        with pytest.raises(ValidationError, match=expected_error):
            generator.generate_circular_trajectory(
                center=center, radius=invalid_radius, height=1.0, num_poses=5
            )

    @pytest.mark.parametrize("invalid_num_poses", [-1, 0])
    def test_circular_trajectory_invalid_num_poses(
        self, invalid_num_poses: int
    ) -> None:
        """Test circular trajectory with invalid num_poses raises ValidationError."""
        generator = TrajectoryGenerator()
        center = np.array([0.0, 0.0])
        with pytest.raises(ValidationError, match="num_poses must be positive integer"):
            generator.generate_circular_trajectory(
                center=center, radius=1.0, height=1.0, num_poses=invalid_num_poses
            )

    @pytest.mark.parametrize("invalid_num_poses", [0, 1])
    def test_linear_trajectory_invalid_num_poses(self, invalid_num_poses: int) -> None:
        """Test linear trajectory with invalid num_poses raises ValidationError."""
        generator = TrajectoryGenerator()
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValidationError, match="num_poses must be integer > 1"):
            generator.generate_linear_trajectory(start_pos, end_pos, invalid_num_poses)

    def test_linear_trajectory_invalid_orientation(self) -> None:
        """Test linear trajectory with invalid orientation raises ValidationError."""
        generator = TrajectoryGenerator()
        start_pos = np.array([0.0, 0.0, 0.0])
        end_pos = np.array([1.0, 0.0, 0.0])
        invalid_orientation = np.array([1.0, 0.0, 0.0])  # Wrong shape

        with pytest.raises(
            ValidationError, match="orientation must be a quaternion of shape"
        ):
            generator.generate_linear_trajectory(
                start_pos, end_pos, num_poses=2, orientation=invalid_orientation
            )
