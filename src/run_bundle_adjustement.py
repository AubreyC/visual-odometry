from typing import List

import numpy as np

from .bundle_adjustment import BundleAdjustment
from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import ImageFeatures, Landmarks3D
from .geometry import GeometryUtils
from .matplotlib_visualizer import MatplotVisualizer


def run_bundle_adjustment() -> None:
    # Landmarks:
    landmarks: np.ndarray = np.array(
        [
            [3.0, 0.5, 0.1],
            [3.0, 0.4, 0.2],
            [3.0, -0.3, 0.4],
            [3.0, 0.3, 0.4],
            [3.0, 0.1, 0.0],
            [3.0, -0.2, -0.25],
            [3.0, -0.4, -0.5],
            [3.0, 0.5, -0.2],
        ]
    )

    # landmarks_generator: LandmarkGenerator = LandmarkGenerator(
    #     (1.8, 2.4), (-0.5, 0.5), (-0.5, 0.5)
    # )
    # landmarks: np.ndarray = landmarks_generator.generate_random(
    #     num_landmarks=100, seed=42
    # )

    # Camera poses:
    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )
    camera_poses = [
        CameraPose(
            position=np.array([-0.5, 0.1, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.02, 0.03, 0.0])),
            ),
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.04, -0.1, 0.2])),
            ),
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.06, 0.02, 0.4])),
            ),
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.08, 0.3, 0.7])),
            ),
            timestamp=3.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.1, 0.05, -0.2])),
            ),
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.12, 0.0, 0.0])),
            ),
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.14, 0.0, 0.0])),
            ),
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.16, 0.0, 0.0])),
            ),
            timestamp=3.0,
        ),
    ]

    # Camera model:
    image_width = 640
    image_height = 480
    camera_model = PinHoleCamera(
        fx=1000.0,
        fy=1000.0,
        cx=image_width / 2.0,
        cy=image_height / 2.0,
    )

    # Render images:
    camera_id = 0
    image_features = []
    for camera_pose in camera_poses:
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

    # Create initial 3D points from the known landmarks
    np.random.seed(42)  # For reproducible results
    landmarks_3d = Landmarks3D(landmarks, np.arange(len(landmarks)))

    # Convert landmarks to CF0 frame (first camera pose)
    camera_poses_first = camera_poses[0].copy()
    landmarks_3d_CF0: Landmarks3D = Landmarks3D(
        camera_poses_first.transform_points_world_to_camera(
            landmarks_3d.get_points_3d()
        ),
        landmarks_3d.get_ids(),
    )

    # Convert camera poses to CF0 frame
    camera_poses_CF0: List[CameraPose] = []
    for camera_pose in camera_poses:
        pose = camera_pose.convert_to_new_frame(camera_poses_first)
        camera_poses_CF0.append(pose)

    # Initial guess for landmarks (adding moise)
    landmarks_3d_guess = Landmarks3D(
        # landmarks_3d_CF0.get_points_3d(),
        landmarks_3d_CF0.get_points_3d()
        + np.random.randn(len(landmarks_3d.get_points_3d()), 3) * 0.5,
        landmarks_3d_CF0.get_ids(),
    )

    # Initial guess for landmarks (adding moise)
    camera_pose_initial = camera_poses_CF0[0].copy()
    camera_poses_guess: List[CameraPose] = []
    for camera_pose in camera_poses_CF0[1:]:
        camera_pose_guess = camera_pose.copy()

        # Add noise to position and orientation
        camera_pose_guess.position = (
            camera_pose_guess.position + np.random.randn(3) * 0.5
        )
        camera_pose_guess.orientation_quaternion = GeometryUtils.quaternion_multiply(
            camera_pose_guess.orientation_quaternion,
            GeometryUtils.quaternion_from_euler_angles(np.random.randn(3) * 0.1),
        )
        camera_poses_guess.append(camera_pose_guess)

    bundle_adjustment = BundleAdjustment()
    optimized_image_features, optimized_points3d, optimized_camera_poses = (
        bundle_adjustment.optimize(
            image_features,
            camera_pose_initial,
            camera_model,
            camera_poses_guess,
            landmarks_3d_guess,
        )
    )

    print("Original landmarks:")
    print(landmarks)
    # print("Noisy initial landmarks:")
    # print(landmarks_noisy)
    # print("Optimized landmarks:")
    # print(optimized_points3d.get_points_3d())

    # Calculate errors
    landmark_error = np.linalg.norm(
        optimized_points3d.get_points_3d() - landmarks_3d_CF0.get_points_3d(), axis=1
    )
    print(f"Mean landmark error: {np.mean(landmark_error):.6f}")
    print(f"Max landmark error: {np.max(landmark_error):.6f}")

    scale = np.linalg.norm(
        landmarks_3d_CF0.get_points_3d()[0, :]
        / np.linalg.norm(optimized_points3d.get_points_3d()[0, :])
    )

    # Rescale solution:
    optimized_points3d.points_3d = optimized_points3d.get_points_3d() * scale
    for camera in optimized_camera_poses[1:]:
        camera.position = camera.position * scale

    print("points3d_guess landmarks scaled:")
    print(landmarks_3d_CF0.get_points_3d())

    print("Optimized landmarks scaled:")
    print(optimized_points3d.get_points_3d())

    for camera_pose_truth, camera_pose_optimized in zip(
        camera_poses_CF0, optimized_camera_poses
    ):
        print(f"{'Camera pose:':25}")
        print(f"{'truth position:':25} {camera_pose_truth.position}")
        print(f"{'Optimized position:':25} {camera_pose_optimized.position}")
        print(f"{'Truth orientation:':25} {camera_pose_truth.quaternion}")
        print(f"{'Optimized orientation:':25} {camera_pose_optimized.quaternion}")

    # Create matplotlib visualizer for 3D plots
    visualizer: MatplotVisualizer = MatplotVisualizer()
    fig = visualizer.plot_scene_overview(
        landmarks=landmarks_3d_CF0.get_points_3d(),
        landmarks_second=optimized_points3d.get_points_3d(),
        poses=camera_poses_CF0,
        poses_second=optimized_camera_poses,
        title="3D Scene",
        show_orientation=True,
        orientation_scale=0.5,
    )
    visualizer.show()


if __name__ == "__main__":
    run_bundle_adjustment()
