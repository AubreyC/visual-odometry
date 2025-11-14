from typing import List, Tuple

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose, TrajectoryGenerator
from .feature_observation import FeatureObservation, ImageObservations
from .geometry import GeometryUtils
from .image_renderer import ImageRenderer
from .landmarks import LandmarkGenerator
from .matplotlib_visualizer import MatplotVisualizer
from .opencv_scene_visualizer import OpenCVSceneVisualizer
from .visual_odometry import VisualOdometry


def generate_straight_scenario() -> Tuple[np.ndarray, List[CameraPose]]:
    """Generate a straight line scenario.

    Returns:
        Tuple[np.ndarray, List[CameraPose]]: Landmarks, List of camera poses
    """

    # Generate landmarks and poses
    landmarks_generator: LandmarkGenerator = LandmarkGenerator(
        (2.0, 3.0), (0.0, 50.0), (0.0, 2.0)
    )
    landmarks_3d: np.ndarray = landmarks_generator.generate_random(
        num_landmarks=1000, seed=42
    )

    generator = TrajectoryGenerator(time_step=0.1)

    start = np.array([0.0, 0.0, 1.0])
    end = np.array([0.0, 50.0, 1.0])

    # Initial camera pose
    camera_initial_pose = CameraPose.create_look_straight_pose()
    poses = generator.generate_linear_trajectory(
        start_position=start,
        end_position=end,
        num_poses=1000,
        orientation=camera_initial_pose.quaternion,
    )

    return landmarks_3d, poses


def generate_circle_scenario() -> Tuple[np.ndarray, List[CameraPose]]:
    """Generate a circle motion scenario.

    Returns:
        Tuple[np.ndarray, List[CameraPose]]: Landmarks, List of camera poses
    """

    # Generate landmarks and poses
    landmarks_generator: LandmarkGenerator = LandmarkGenerator(
        (-4.0, 4.0), (-4.0, 4.0), (0.0, 2.0)
    )
    landmarks_3d: np.ndarray = landmarks_generator.generate_random(
        num_landmarks=500, seed=42
    )

    generator = TrajectoryGenerator(time_step=0.1)
    center = np.array([0.0, 0.0])
    camera_initial_pose = CameraPose.create_look_straight_pose()
    poses = generator.generate_circular_trajectory(
        center=center,
        radius=2.0,
        height=1.0,
        num_poses=200,
        look_at_center=True,
        angular_velocity=0.5,
        orientation_offset=camera_initial_pose.quaternion,
    )

    return landmarks_3d, poses


def main() -> None:
    # Create scenario:
    landmarks, poses = generate_circle_scenario()

    # Camera model:
    camera_width: int = 640
    camera_height: int = 480
    camera: PinHoleCamera = PinHoleCamera(
        fx=500.0, fy=500.0, cx=camera_width / 2, cy=camera_height / 2
    )
    image_renderer: ImageRenderer = ImageRenderer(camera)

    # Scene visualizer camera pose:
    scene_camera_pose = CameraPose.create_look_at_target(
        camera_position=np.array([-15.0, -15.0, 8.0]),
        target_position=np.array([0.0, 0.0, 0.0]),
        timestamp=0.0,
    )

    # Create matplotlib visualizer for 3D plots
    visualizer: MatplotVisualizer = MatplotVisualizer()
    fig = visualizer.plot_scene_overview(
        landmarks=landmarks,
        poses=poses,
        title="3D Scene",
        show_orientation=True,
        orientation_scale=0.5,
    )
    visualizer.show()

    # Create OpenCV scene visualizer for real-time 3D visualization
    opencv_visualizer = OpenCVSceneVisualizer(
        image_width=camera_width,
        image_height=camera_height,
        background_color=(255, 255, 255),
    )

    initial_pose = CameraPose.create_look_straight_pose()
    visual_odometry: VisualOdometry = VisualOdometry(initial_pose=initial_pose)

    scale = 1.0
    features_observations_prev = None
    pose_prev = None
    for frame_idx, pose in enumerate(poses):
        features_observations: List[FeatureObservation] = (
            image_renderer.project_landmarks_to_image(landmarks, pose)
        )

        image_observations: ImageObservations = ImageObservations(
            camera_id=0,
            timestamp=pose.timestamp,
            feature_observations=features_observations,
            image_width=camera_width,
            image_height=camera_height,
        )

        if features_observations_prev is not None:
            print(f"\nProcessing visual odometry: {frame_idx}")
            previous_ids = [
                feature.landmark_id for feature in features_observations_prev
            ]
            current_ids = [feature.landmark_id for feature in features_observations]

            prev_features = []
            for feature in features_observations_prev:
                prev_features.append(feature.image_coords)
            prev_features = np.array(prev_features)

            current_features = []
            for feature in features_observations:
                current_features.append(feature.image_coords)
            new_features = np.array(current_features)

            previous_ids = np.array(previous_ids)
            current_ids = np.array(current_ids)

            if not visual_odometry.is_initialized():
                visual_odometry.init_visual_odometry(
                    timestamp=pose.timestamp,
                    pts2d_prev=prev_features,
                    pts2d_ids_prev=previous_ids,
                    pts2d_new=new_features,
                    pts2d_ids_new=current_ids,
                    camera_matrix=camera.get_camera_matrix(),
                )

                # Compute scale based on the distance between the first and second pose
                scale = np.linalg.norm(
                    poses[1].position - poses[0].position
                ) / np.linalg.norm(visual_odometry.get_current_pose().position)

            else:
                visual_odometry.update_visual_odometry(
                    timestamp=pose.timestamp,
                    pts2d_prev=prev_features,
                    pts2d_ids_prev=previous_ids,
                    pts2d_new=new_features,
                    pts2d_ids_new=current_ids,
                    camera_matrix=camera.get_camera_matrix(),
                )

            # Scale camera pose:
            camera_pose = visual_odometry.get_current_pose()
            camera_pose.position = camera_pose.position * scale

            # Ground truth camera pose position in world frame:
            camera_pose_world = CameraPose(
                position=poses[0].rotation_matrix @ camera_pose.position
                + poses[0].position,
                orientation=GeometryUtils.quaternion_from_rotation_matrix(
                    (
                        camera_pose.rotation_matrix.transpose()
                        @ poses[0].rotation_matrix.transpose()
                    ).transpose()
                ),
                timestamp=pose.timestamp,
            )

            print("Estimated:\n")
            print("camera_pose_world position:\n", camera_pose_world.position)
            print(
                "camera_pose_world orientation:\n",
                GeometryUtils.euler_angles_from_rotation_matrix(
                    camera_pose_world.rotation_matrix
                ),
            )
            print("Truth:\n")
            print("camera_pose_world position:\n", pose.position)
            print(
                "camera_pose_world orientation:\n",
                GeometryUtils.euler_angles_from_rotation_matrix(pose.rotation_matrix),
            )

        features_observations_prev = features_observations
        pose_prev = pose

        # Show feature observations and tracking on camera frame:
        image_features = ImageObservations.to_opencv_image(image_observations)

        # Get estimated 3D points and their ids:
        pts_3d, pts_3d_ids = visual_odometry.get_last_used_points_3d()
        pts_3d = pts_3d * scale

        # Get estimated camera pose:
        camera_pose = visual_odometry.get_current_pose()
        camera_pose.position = camera_pose.position * scale

        # Show 3D scene with 3D points and camera pose:
        image_scene = opencv_visualizer.show_scene_static(
            scene_camera_pose=scene_camera_pose,
            camera_pose=camera_pose,
            landmarks=pts_3d,
            landmark_ids=pts_3d_ids,
            show_axes=True,
        )

        # Show results
        image = OpenCVSceneVisualizer.merge_images([image_features, image_scene])
        cv2.imshow("Feature Tracking and 3D Scene", image)
        key = cv2.waitKey(0)
        if key == ord("w"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
