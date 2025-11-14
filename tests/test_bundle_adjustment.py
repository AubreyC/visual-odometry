import numpy as np
import pytest

from src.bundle_adjustment import BundleAdjustment
from src.camera import PinHoleCamera
from src.camera_pose import CameraPose
from src.feature_observation import ImageFeatures, Landmarks3D
from src.geometry import GeometryUtils
from src.landmarks import LandmarkGenerator


class TestBundleAdjustment:
    """Test suite for BundleAdjustment class."""

    @pytest.fixture
    def camera_model(self) -> PinHoleCamera:
        """Create a standard pinhole camera model for testing."""
        image_width = 640
        image_height = 480
        return PinHoleCamera(
            fx=1000.0,
            fy=1000.0,
            cx=image_width / 2.0,
            cy=image_height / 2.0,
        )

    @pytest.fixture
    def synthetic_data(self, camera_model: PinHoleCamera) -> dict:
        """Generate synthetic test data similar to run_bundle_adjustement.py."""
        np.random.seed(42)  # For reproducible results

        # Generate landmarks
        landmarks_generator = LandmarkGenerator((2.8, 3.2), (-0.5, 0.5), (-0.5, 0.5))
        landmarks = landmarks_generator.generate_random(num_landmarks=50, seed=42)

        # Create ground truth camera poses
        quat_cam = GeometryUtils.quaternion_from_euler_angles(
            np.array([-np.pi / 2, 0.0, -np.pi / 2])
        )
        camera_poses_gt = [
            CameraPose(
                position=np.array([-0.5, 0.1, 0.0]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.02, 0.03, 0.0])
                    ),
                ),
                timestamp=0.0,
            ),
            CameraPose(
                position=np.array([0.0, -0.2, 0.0]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.04, -0.1, 0.2])
                    ),
                ),
                timestamp=1.0,
            ),
            CameraPose(
                position=np.array([0.0, 0.0, 0.3]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.06, 0.02, 0.4])
                    ),
                ),
                timestamp=2.0,
            ),
            CameraPose(
                position=np.array([0.0, 0.0, -0.3]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.08, 0.3, 0.7])
                    ),
                ),
                timestamp=3.0,
            ),
        ]

        # Generate image features by projecting landmarks
        image_width = 640
        image_height = 480
        camera_id = 0
        image_features = []
        for camera_pose in camera_poses_gt:
            pts3d_camera_frame = camera_pose.transform_points_world_to_camera(landmarks)
            image_feature = ImageFeatures.from_points_3d(
                camera_pose.timestamp,
                image_width,
                image_height,
                camera_id,
                camera_model,
                pts3d_camera_frame,
                np.arange(len(landmarks)),
            )
            image_features.append(image_feature)

        # Convert to CF0 frame (first camera pose)
        camera_poses_first = camera_poses_gt[0].copy()
        landmarks_cf0 = Landmarks3D(landmarks, np.arange(len(landmarks)))
        landmarks_3d_cf0 = Landmarks3D(
            camera_poses_first.transform_points_world_to_camera(
                landmarks_cf0.get_points_3d()
            ),
            landmarks_cf0.get_ids(),
        )

        # Convert camera poses to CF0 frame
        camera_poses_cf0 = []
        for camera_pose in camera_poses_gt:
            pose = camera_pose.convert_to_new_frame(camera_poses_first)
            camera_poses_cf0.append(pose)

        return {
            "landmarks_cf0": landmarks_3d_cf0,
            "camera_poses_cf0": camera_poses_cf0,
            "image_features": image_features,
            "camera_model": camera_model,
        }

    def test_bundle_adjustment_convergence(self, synthetic_data: dict) -> None:
        """Test that bundle adjustment reduces reprojection error."""
        np.random.seed(42)  # For reproducible results

        # Extract data
        landmarks_cf0 = synthetic_data["landmarks_cf0"]
        camera_poses_cf0 = synthetic_data["camera_poses_cf0"]
        image_features = synthetic_data["image_features"]
        camera_model = synthetic_data["camera_model"]

        # Create noisy initial guesses for landmarks
        landmark_noise_std = 0.2
        landmarks_guess = Landmarks3D(
            landmarks_cf0.get_points_3d()
            + np.random.randn(len(landmarks_cf0.get_points_3d()), 3)
            * landmark_noise_std,
            landmarks_cf0.get_ids(),
        )

        # Create noisy initial guesses for camera poses (skip first pose)
        camera_pose_noise_pos_std = 0.1
        camera_pose_noise_rot_std = 0.05
        camera_poses_guess = []
        for camera_pose in camera_poses_cf0[1:]:
            camera_pose_guess = camera_pose.copy()

            # Add noise to position
            camera_pose_guess.position = (
                camera_pose_guess.position
                + np.random.randn(3) * camera_pose_noise_pos_std
            )

            # Add noise to orientation
            camera_pose_guess.orientation_quaternion = (
                GeometryUtils.quaternion_multiply(
                    camera_pose_guess.orientation_quaternion,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.random.randn(3) * camera_pose_noise_rot_std
                    ),
                )
            )
            camera_poses_guess.append(camera_pose_guess)

        # Calculate initial reprojection error
        initial_error = self._calculate_reprojection_error(
            image_features,
            camera_poses_cf0[0],
            camera_poses_guess,
            landmarks_guess,
            camera_model,
        )

        # Run bundle adjustment
        bundle_adjustment = BundleAdjustment()
        optimized_image_features, optimized_points3d, optimized_camera_poses = (
            bundle_adjustment.optimize(
                image_features,
                camera_poses_cf0[0],  # First pose is fixed
                camera_model,
                camera_poses_guess,
                landmarks_guess,
            )
        )

        # Calculate final reprojection error
        final_error = self._calculate_reprojection_error(
            optimized_image_features,
            optimized_camera_poses[0],
            optimized_camera_poses[1:],
            optimized_points3d,
            camera_model,
        )

        # Assert that optimization reduced the error
        assert final_error < initial_error, (
            f"Bundle adjustment should reduce reprojection error. "
            f"Initial error: {initial_error:.6f}, Final error: {final_error:.6f}"
        )

        # Assert reasonable error reduction (at least 50% improvement)
        improvement_ratio = (initial_error - final_error) / initial_error
        assert improvement_ratio > 0.5, (
            f"Bundle adjustment should significantly improve the solution. "
            f"Improvement ratio: {improvement_ratio:.3f}"
        )

    def test_bundle_adjustment_landmark_accuracy(self, synthetic_data: dict) -> None:
        """Test that bundle adjustment improves landmark position accuracy."""
        np.random.seed(42)  # For reproducible results

        # Extract data
        landmarks_cf0 = synthetic_data["landmarks_cf0"]
        camera_poses_cf0 = synthetic_data["camera_poses_cf0"]
        image_features = synthetic_data["image_features"]
        camera_model = synthetic_data["camera_model"]

        # Create noisy initial guesses for landmarks
        landmark_noise_std = 0.3
        landmarks_guess = Landmarks3D(
            landmarks_cf0.get_points_3d()
            + np.random.randn(len(landmarks_cf0.get_points_3d()), 3)
            * landmark_noise_std,
            landmarks_cf0.get_ids(),
        )

        # Create noisy initial guesses for camera poses
        camera_pose_noise_pos_std = 0.05
        camera_pose_noise_rot_std = 0.02
        camera_poses_guess = []
        for camera_pose in camera_poses_cf0[1:]:
            camera_pose_guess = camera_pose.copy()

            # Add noise to position
            camera_pose_guess.position = (
                camera_pose_guess.position
                + np.random.randn(3) * camera_pose_noise_pos_std
            )

            # Add noise to orientation
            camera_pose_guess.orientation_quaternion = (
                GeometryUtils.quaternion_multiply(
                    camera_pose_guess.orientation_quaternion,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.random.randn(3) * camera_pose_noise_rot_std
                    ),
                )
            )
            camera_poses_guess.append(camera_pose_guess)

        # Calculate initial landmark error
        initial_landmark_error = np.mean(
            np.linalg.norm(
                landmarks_guess.get_points_3d() - landmarks_cf0.get_points_3d(), axis=1
            )
        )

        # Run bundle adjustment
        bundle_adjustment = BundleAdjustment()
        _, optimized_points3d, _ = bundle_adjustment.optimize(
            image_features,
            camera_poses_cf0[0],  # First pose is fixed
            camera_model,
            camera_poses_guess,
            landmarks_guess,
        )

        # Calculate final landmark error (after scaling to match ground truth scale)
        scale = np.linalg.norm(landmarks_cf0.get_points_3d()[0, :]) / np.linalg.norm(
            optimized_points3d.get_points_3d()[0, :]
        )
        optimized_points3d_scaled = optimized_points3d.get_points_3d() * scale

        final_landmark_error = np.mean(
            np.linalg.norm(
                optimized_points3d_scaled - landmarks_cf0.get_points_3d(), axis=1
            )
        )

        # Assert that optimization improved landmark accuracy
        assert final_landmark_error < initial_landmark_error, (
            f"Bundle adjustment should improve landmark accuracy. "
            f"Initial error: {initial_landmark_error:.6f}, Final error: {final_landmark_error:.6f}"
        )

    def _calculate_reprojection_error(
        self,
        image_features: list,
        camera_pose_initial: CameraPose,
        camera_poses_guess: list,
        landmarks_guess: Landmarks3D,
        camera_model: PinHoleCamera,
    ) -> float:
        """Calculate total reprojection error for given poses and landmarks."""
        total_error = 0.0
        num_observations = 0

        all_camera_poses = [camera_pose_initial] + camera_poses_guess

        for pose_idx, image_feature in enumerate(image_features):
            camera_pose = all_camera_poses[pose_idx]

            # Get observed landmark IDs
            obs_ids = image_feature.get_points_2d().get_ids()

            # Get corresponding 3D points
            points_3d_selected = landmarks_guess.get_selected_ids(obs_ids)
            landmark_cf = camera_pose.transform_points_world_to_camera(
                points_3d_selected.get_points_3d()
            )

            # Project to image plane
            projected = camera_model.project(landmark_cf)

            # Calculate reprojection error
            observed_points = image_feature.get_points_2d().get_points_2d()
            errors = np.linalg.norm(projected - observed_points, axis=1)

            total_error += np.sum(errors**2)
            num_observations += len(errors)

        return total_error / num_observations if num_observations > 0 else float("inf")
