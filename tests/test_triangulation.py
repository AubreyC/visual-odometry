from typing import List

import numpy as np

from src.camera import PinHoleCamera
from src.camera_pose import CameraPose
from src.feature_observation import FeatureObservation, ImageObservations
from src.geometry import GeometryUtils
from src.image_renderer import ImageRenderer
from src.visual_odometry import VisualOdometry


class TestTriangulation:
    """Test suite for  class."""

    def test_triangulation(self) -> None:
        landmarks = np.array(
            [
                [2.0773956, -0.37188637, 0.05458479],
                [2.04388784, -0.04961406, -0.43618274],
                [2.08585979, -0.12920198, 0.32763117],
                [2.0697368, 0.42676499, 0.1316644],
                [2.00941773, 0.14386512, 0.25808774],
                [2.09756224, 0.32276161, -0.14547403],
                [2.07611397, -0.0565858, 0.47069802],
                [2.07860643, -0.27276128, 0.39312112],
            ]
        )

        # This is so that the camera is looking at toward the X world axis
        quat_cam = GeometryUtils.quaternion_from_euler_angles(
            np.array([-np.pi / 2, 0.0, -np.pi / 2]),
        )

        poses = [
            CameraPose(
                position=np.array([0.0, 0.2, 0.0]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.02, 0.03, 0.0])
                    ),
                ),
                timestamp=0.0,
            ),
            CameraPose(
                position=np.array([0.0, 0.3, -0.3]),
                orientation=GeometryUtils.quaternion_multiply(
                    quat_cam,
                    GeometryUtils.quaternion_from_euler_angles(
                        np.array([0.08, 0.3, 0.7])
                    ),
                ),
                timestamp=3.0,
            ),
        ]

        camera_width: int = 640
        camera_height: int = 480

        camera: PinHoleCamera = PinHoleCamera(
            fx=500.0, fy=500.0, cx=camera_width / 2, cy=camera_height / 2
        )
        image_renderer: ImageRenderer = ImageRenderer(camera)

        frame_observations: List[ImageObservations] = []
        for frame_idx, pose in enumerate(poses):
            list_of_observations: List[FeatureObservation] = (
                image_renderer.project_landmarks_to_image(landmarks, pose)
            )

            image_observations: ImageObservations = ImageObservations(
                camera_id=0,
                timestamp=pose.timestamp,
                feature_observations=list_of_observations,
                image_width=camera_width,
                image_height=camera_height,
            )
            frame_observations.append(image_observations)

        # Triangulation
        pose_0 = poses[0]
        pose_1 = poses[1]
        image_observations_0 = frame_observations[0]
        image_observations_1 = frame_observations[1]

        feature_ids_0 = np.array(
            [
                feature.landmark_id
                for feature in image_observations_0.feature_observations
            ]
        )
        feature_ids_1 = np.array(
            [
                feature.landmark_id
                for feature in image_observations_1.feature_observations
            ]
        )

        features_0 = []
        for feature in image_observations_0.feature_observations:
            features_0.append(feature.image_coords)
        features_0 = np.array(features_0)

        features_1 = []
        for feature in image_observations_1.feature_observations:
            features_1.append(feature.image_coords)
        features_1 = np.array(features_1)

        features_selected_0, features_selected_1, common_ids = (
            VisualOdometry.get_common_pts2d(
                features_0, feature_ids_0, features_1, feature_ids_1
            )
        )
        print("common_ids:\n", common_ids)
        for landmark_id, landmark in enumerate(landmarks):
            if landmark_id in common_ids:
                print(f"landmark {landmark_id} {landmark.T}")

        points_3d = VisualOdometry.triangulate_points(
            pose_0.rotation_matrix.transpose(),
            pose_0.position.reshape(3, 1),
            pose_1.rotation_matrix.transpose(),
            pose_1.position.reshape(3, 1),
            camera.get_camera_matrix(),
            features_selected_0,
            features_selected_1,
        )

        assert np.linalg.norm(points_3d - landmarks) < 1e-6
