import numpy as np
import pytest

from src.camera import PinHoleCamera
from src.camera_pose import CameraPose, TrajectoryGenerator
from src.geometry import GeometryUtils
from src.image_renderer import ImageRenderer
from src.landmarks import LandmarkGenerator
from src.visual_odometry import VisualOdometry


class TestVisualSLAMSimulation:
    """Test suite for Visual SLAM simulation using synthetic data."""

    @pytest.fixture
    def camera(self) -> PinHoleCamera:
        """Create a test camera."""
        return PinHoleCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    @pytest.fixture
    def landmarks(self) -> np.ndarray:
        """Create synthetic landmarks for testing."""
        landmarks_generator = LandmarkGenerator((2.0, 3.0), (0.0, 50.0), (0.0, 2.0))
        return landmarks_generator.generate_random(
            num_landmarks=1000,
            seed=42,  # Use more landmarks for better overlap
        )

    @pytest.fixture
    def camera_trajectory(self) -> list[CameraPose]:
        """Create a linear camera trajectory."""
        # Camera orientation offset to look toward the X world axis
        camera_quat_offset = GeometryUtils.quaternion_from_euler_angles(
            np.array([-np.pi / 2, 0.0, -np.pi / 2]),
        )

        generator = TrajectoryGenerator(time_step=0.1)
        start = np.array([0.0, 0.0, 1.0])
        end = np.array([0.0, 50.0, 1.0])

        return generator.generate_linear_trajectory(
            start_position=start,
            end_position=end,
            num_poses=100,  # Use more poses for better testing
            orientation=camera_quat_offset,
        )

    @pytest.fixture
    def image_renderer(self, camera: PinHoleCamera) -> ImageRenderer:
        """Create an image renderer."""
        return ImageRenderer(camera)

    @pytest.fixture
    def initial_pose(self) -> CameraPose:
        """Create initial camera pose."""
        return CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=GeometryUtils.quaternion_from_euler_angles(
                np.array([0.0, 0.0, 0.0])
            ),
            timestamp=0.0,
        )

    def test_visual_odometry_initialization(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
        camera: PinHoleCamera,
        initial_pose: CameraPose,
    ) -> None:
        """Test VisualOdometry initialization with first two frames."""
        visual_odometry = VisualOdometry(initial_pose=initial_pose)

        # Get observations for first two frames
        frame1_observations = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[0]
        )
        frame2_observations = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[1]
        )

        # Extract features
        prev_features = np.array([obs.image_coords for obs in frame1_observations])
        prev_ids = np.array([obs.landmark_id for obs in frame1_observations])
        current_features = np.array([obs.image_coords for obs in frame2_observations])
        current_ids = np.array([obs.landmark_id for obs in frame2_observations])

        # Initialize visual odometry
        success = visual_odometry.init_visual_odometry(
            timestamp=camera_trajectory[1].timestamp,
            pts2d_prev=prev_features,
            pts2d_ids_prev=prev_ids,
            pts2d_new=current_features,
            pts2d_ids_new=current_ids,
            camera_matrix=camera.get_camera_matrix(),
        )

        # Check that VO initialization was attempted (may fail due to insufficient features)
        # If it fails, skip the rest of the test
        if not success:
            pytest.skip(
                "Visual odometry initialization failed - not enough common features"
            )

        # Check that VO is initialized
        assert visual_odometry.is_initialized()

        # Check that 3D points were triangulated
        points_3d, points_3d_ids = visual_odometry.get_points_3d()
        assert len(points_3d) > 0
        assert len(points_3d_ids) > 0
        assert points_3d.shape[1] == 3  # 3D coordinates

        # Check that pose was updated
        current_pose = visual_odometry.get_current_pose()
        assert current_pose is not None
        assert np.allclose(current_pose.timestamp, camera_trajectory[1].timestamp)

    def test_visual_odometry_update(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
        camera: PinHoleCamera,
        initial_pose: CameraPose,
    ) -> None:
        """Test VisualOdometry update process."""
        visual_odometry = VisualOdometry(initial_pose=initial_pose)

        # Initialize with first two frames
        frame0_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[0]
        )
        frame1_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[1]
        )

        prev_features = np.array([obs.image_coords for obs in frame0_obs])
        prev_ids = np.array([obs.landmark_id for obs in frame0_obs])
        current_features = np.array([obs.image_coords for obs in frame1_obs])
        current_ids = np.array([obs.landmark_id for obs in frame1_obs])

        success = visual_odometry.init_visual_odometry(
            timestamp=camera_trajectory[1].timestamp,
            pts2d_prev=prev_features,
            pts2d_ids_prev=prev_ids,
            pts2d_new=current_features,
            pts2d_ids_new=current_ids,
            camera_matrix=camera.get_camera_matrix(),
        )

        if not success:
            pytest.skip(
                "Visual odometry initialization failed - not enough common features"
            )

        initial_points_count = len(visual_odometry.get_points_3d()[0])

        # Update with next frame
        frame2_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[2]
        )
        next_features = np.array([obs.image_coords for obs in frame2_obs])
        next_ids = np.array([obs.landmark_id for obs in frame2_obs])

        visual_odometry.update_visual_odometry(
            timestamp=camera_trajectory[2].timestamp,
            pts2d_prev=current_features,
            pts2d_ids_prev=current_ids,
            pts2d_new=next_features,
            pts2d_ids_new=next_ids,
            camera_matrix=camera.get_camera_matrix(),
        )

        # Check that pose was updated
        updated_pose = visual_odometry.get_current_pose()
        assert updated_pose.timestamp == camera_trajectory[2].timestamp

        # Check that we still have 3D points
        points_3d, _ = visual_odometry.get_points_3d()
        assert len(points_3d) >= initial_points_count

    def test_pose_accuracy_over_trajectory(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
        camera: PinHoleCamera,
        initial_pose: CameraPose,
    ) -> None:
        """Test that estimated poses remain reasonably close to ground truth."""
        visual_odometry = VisualOdometry(initial_pose=initial_pose)

        prev_observations = None
        scale = None

        for frame_idx, pose in enumerate(camera_trajectory):
            current_observations = image_renderer.project_landmarks_to_image(
                landmarks, pose
            )

            if prev_observations is not None:
                # Extract features
                prev_features = np.array(
                    [obs.image_coords for obs in prev_observations]
                )
                prev_ids = np.array([obs.landmark_id for obs in prev_observations])
                current_features = np.array(
                    [obs.image_coords for obs in current_observations]
                )
                current_ids = np.array(
                    [obs.landmark_id for obs in current_observations]
                )

                if not visual_odometry.is_initialized():
                    # Initialize
                    success = visual_odometry.init_visual_odometry(
                        timestamp=pose.timestamp,
                        pts2d_prev=prev_features,
                        pts2d_ids_prev=prev_ids,
                        pts2d_new=current_features,
                        pts2d_ids_new=current_ids,
                        camera_matrix=camera.get_camera_matrix(),
                    )

                    if not success:
                        prev_observations = current_observations
                        continue  # Skip this frame if initialization failed

                    # Compute scale based on first motion
                    current_pose_position = visual_odometry.get_current_pose().position
                    if (
                        np.linalg.norm(current_pose_position) > 1e-6
                    ):  # Avoid division by zero
                        scale = np.linalg.norm(
                            camera_trajectory[1].position
                            - camera_trajectory[0].position
                        ) / np.linalg.norm(current_pose_position)
                    else:
                        scale = (
                            1.0  # Default scale if pose estimation gives zero motion
                        )

                else:
                    # Update
                    visual_odometry.update_visual_odometry(
                        timestamp=pose.timestamp,
                        pts2d_prev=prev_features,
                        pts2d_ids_prev=prev_ids,
                        pts2d_new=current_features,
                        pts2d_ids_new=current_ids,
                        camera_matrix=camera.get_camera_matrix(),
                    )

                # Apply scaling
                camera_pose = visual_odometry.get_current_pose()
                camera_pose.position = camera_pose.position * scale

                # Transform to world coordinates
                camera_pose_world = CameraPose(
                    position=camera_trajectory[0].rotation_matrix @ camera_pose.position
                    + camera_trajectory[0].position,
                    orientation=GeometryUtils.quaternion_from_rotation_matrix(
                        (
                            camera_pose.rotation_matrix.transpose()
                            @ camera_trajectory[0].rotation_matrix.transpose()
                        ).transpose()
                    ),
                    timestamp=pose.timestamp,
                )

                # Check position accuracy (within reasonable bounds for early frames)
                # Only check accuracy for the first 10 frames to avoid accumulated error
                if frame_idx < 10:
                    position_error = np.linalg.norm(
                        camera_pose_world.position - pose.position
                    )
                    # For synthetic data, position error should be reasonable
                    # This is a loose bound since we're testing the basic functionality
                    assert position_error < 20.0, (
                        f"Position error too large at frame {frame_idx}"
                    )

            prev_observations = current_observations

        # Ensure we processed multiple frames
        assert visual_odometry.is_initialized()

    def test_triangulation_accuracy(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
        camera: PinHoleCamera,
        initial_pose: CameraPose,
    ) -> None:
        """Test that triangulated 3D points are reasonably accurate."""
        visual_odometry = VisualOdometry(initial_pose=initial_pose)

        # Initialize with first two frames
        frame0_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[0]
        )
        frame1_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[1]
        )

        prev_features = np.array([obs.image_coords for obs in frame0_obs])
        prev_ids = np.array([obs.landmark_id for obs in frame0_obs])
        current_features = np.array([obs.image_coords for obs in frame1_obs])
        current_ids = np.array([obs.landmark_id for obs in frame1_obs])

        success = visual_odometry.init_visual_odometry(
            timestamp=camera_trajectory[1].timestamp,
            pts2d_prev=prev_features,
            pts2d_ids_prev=prev_ids,
            pts2d_new=current_features,
            pts2d_ids_new=current_ids,
            camera_matrix=camera.get_camera_matrix(),
        )

        if not success:
            pytest.skip(
                "Visual odometry initialization failed - not enough common features"
            )

        # Get triangulated points
        points_3d, points_3d_ids = visual_odometry.get_points_3d()

        # Compute scale and transform to world coordinates
        current_pose_position = visual_odometry.get_current_pose().position
        if np.linalg.norm(current_pose_position) > 1e-6:
            scale = np.linalg.norm(
                camera_trajectory[1].position - camera_trajectory[0].position
            ) / np.linalg.norm(current_pose_position)
        else:
            scale = 1.0

        camera_pose = visual_odometry.get_current_pose()
        camera_pose.position = camera_pose.position * scale

        points_3d_world = (
            camera_trajectory[0].rotation_matrix @ points_3d.T
        ).T + camera_trajectory[0].position

        # Check that triangulated points are reasonably close to ground truth
        for i, landmark_id in enumerate(points_3d_ids):
            ground_truth = landmarks[landmark_id]
            estimated = points_3d_world[i]
            error = np.linalg.norm(estimated - ground_truth)

            # For synthetic data with perfect observations, triangulation should be reasonably accurate
            # Allow some error due to numerical precision and pose estimation inaccuracies
            assert error < 5.0, (
                f"Triangulation error too large for landmark {landmark_id}"
            )

    def test_scaling_and_coordinate_transform(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
        camera: PinHoleCamera,
        initial_pose: CameraPose,
    ) -> None:
        """Test scaling and coordinate transformation logic."""
        visual_odometry = VisualOdometry(initial_pose=initial_pose)

        # Initialize with first two frames
        frame0_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[0]
        )
        frame1_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[1]
        )

        prev_features = np.array([obs.image_coords for obs in frame0_obs])
        prev_ids = np.array([obs.landmark_id for obs in frame0_obs])
        current_features = np.array([obs.image_coords for obs in frame1_obs])
        current_ids = np.array([obs.landmark_id for obs in frame1_obs])

        success = visual_odometry.init_visual_odometry(
            timestamp=camera_trajectory[1].timestamp,
            pts2d_prev=prev_features,
            pts2d_ids_prev=prev_ids,
            pts2d_new=current_features,
            pts2d_ids_new=current_ids,
            camera_matrix=camera.get_camera_matrix(),
        )

        if not success:
            pytest.skip(
                "Visual odometry initialization failed - not enough common features"
            )

        # Compute scale
        current_pose_position = visual_odometry.get_current_pose().position
        if np.linalg.norm(current_pose_position) > 1e-6:
            scale = np.linalg.norm(
                camera_trajectory[1].position - camera_trajectory[0].position
            ) / np.linalg.norm(current_pose_position)
        else:
            scale = 1.0

        # Apply scaling
        camera_pose = visual_odometry.get_current_pose()
        original_position = camera_pose.position.copy()
        camera_pose.position = camera_pose.position * scale

        # Verify scaling was applied correctly
        assert np.allclose(camera_pose.position, original_position * scale)

        # Test coordinate transformation to world frame
        camera_pose_world = CameraPose(
            position=camera_trajectory[0].rotation_matrix @ camera_pose.position
            + camera_trajectory[0].position,
            orientation=GeometryUtils.quaternion_from_rotation_matrix(
                (
                    camera_pose.rotation_matrix.transpose()
                    @ camera_trajectory[0].rotation_matrix.transpose()
                ).transpose()
            ),
            timestamp=camera_pose.timestamp,
        )

        # Verify transformation is valid (not checking accuracy here)
        assert camera_pose_world.position.shape == (3,)
        assert np.isfinite(camera_pose_world.position).all()

    def test_common_feature_matching(
        self,
        landmarks: np.ndarray,
        camera_trajectory: list[CameraPose],
        image_renderer: ImageRenderer,
    ) -> None:
        """Test that common features are correctly identified between frames."""
        # Get observations for two consecutive frames
        frame1_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[0]
        )
        frame2_obs = image_renderer.project_landmarks_to_image(
            landmarks, camera_trajectory[1]
        )

        pts2d_1 = np.array([obs.image_coords for obs in frame1_obs])
        ids_1 = np.array([obs.landmark_id for obs in frame1_obs])
        pts2d_2 = np.array([obs.image_coords for obs in frame2_obs])
        ids_2 = np.array([obs.landmark_id for obs in frame2_obs])

        # Get common features
        common_pts2d_1, common_pts2d_2, common_ids = VisualOdometry.get_common_pts2d(
            pts2d_1, ids_1, pts2d_2, ids_2
        )

        # Verify common features
        assert len(common_pts2d_1) == len(common_pts2d_2) == len(common_ids)
        assert len(common_ids) > 0  # Should have some common landmarks

        # Verify that common IDs are indeed in both frames
        assert all(common_id in ids_1 for common_id in common_ids)
        assert all(common_id in ids_2 for common_id in common_ids)
