from typing import List

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import FeatureObservation, ImageObservations
from .geometry import GeometryUtils
from .image_renderer import ImageRenderer
from .landmarks import LandmarkGenerator
from .visual_odometry import VisualOdometry
from .visualization import OpenCVSceneVisualizer, Visualizer


def main() -> None:
    # Generate landmarks and poses
    landmarks_generator: LandmarkGenerator = LandmarkGenerator(
        (2.0, 2.1), (-0.5, 0.5), (-0.5, 0.5)
    )
    landmarks_3d: np.ndarray = landmarks_generator.generate_random(
        num_landmarks=20, seed=42
    )

    # landmarks_3d: np.ndarray = np.array(
    #     [
    #         [2.0, 0.5, 0.1],
    #         [2.0, 0.4, 0.2],
    #         [2.0, -0.3, 0.4],
    #         [2.0, 0.3, 0.4],
    #         [2.0, 0.1, 0.0],
    #         [2.0, -0.2, -0.25],
    #         [2.0, -0.4, -0.5],
    #         [2.0, 0.5, -0.2],
    #     ]
    # )
    # landmarks_3d = np.array(
    #     [
    #         [2.0773956, -0.37188637, 0.05458479],
    #         [2.04388784, -0.04961406, -0.43618274],
    #         [2.08585979, -0.12920198, 0.32763117],
    #         [2.0697368, 0.42676499, 0.1316644],
    #         [2.00941773, 0.14386512, 0.25808774],
    #         [2.09756224, 0.32276161, -0.14547403],
    #         [2.07611397, -0.0565858, 0.47069802],
    #         [2.07860643, -0.27276128, 0.39312112],
    #     ]
    # )
    landmarks = landmarks_3d.copy()
    print("landmarks:\n", landmarks)

    # This is so that the camera is looking at toward the X world axis
    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )

    poses = [
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
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

    camera_width: int = 640
    camera_height: int = 480

    camera: PinHoleCamera = PinHoleCamera(
        fx=500.0, fy=500.0, cx=camera_width / 2, cy=camera_height / 2
    )
    image_renderer: ImageRenderer = ImageRenderer(camera)

    scene_quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )

    scene_quat_cam = GeometryUtils.quaternion_multiply(
        scene_quat_cam,
        GeometryUtils.quaternion_from_euler_angles(
            np.array([-np.pi / 4, -np.pi / 4, 0.0])
        ),
    )

    scene_camera_pose = CameraPose(
        position=np.array([-4.0, -4.0, 4.0]),
        orientation=scene_quat_cam,
        timestamp=0.0,
    )

    # Create matplotlib visualizer for 3D plots
    visualizer: Visualizer = Visualizer()
    fig = visualizer.plot_scene_overview(
        landmarks=landmarks,
        poses=poses,
        title="Test Scene",
        show_orientation=True,
        orientation_scale=0.5,
    )
    visualizer.show()

    # Create OpenCV scene visualizer for real-time 3D visualization
    opencv_visualizer = OpenCVSceneVisualizer(
        image_width=1280, image_height=720, background_color=(255, 255, 255)
    )

    # Demonstrate the OpenCV visualizer with different camera poses
    print("Demonstrating OpenCV Scene Visualizer...")

    # Show scene from first camera pose
    for pose in poses:
        print("Showing scene from first camera pose...")
        opencv_visualizer.show_scene(
            scene_camera_pose=scene_camera_pose,
            camera_pose=pose,
            landmarks=landmarks,
            landmark_ids=list(range(len(landmarks))),
            window_name="OpenCV 3D Scene Visualizer",
            show_axes=True,
        )

    prev_list = None
    prev_pose = None

    initial_pose = CameraPose(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=GeometryUtils.quaternion_from_euler_angles(
            np.array([0.0, 0.0, 0.0])
        ),
        timestamp=0.0,
    )
    visual_odometry: VisualOdometry = VisualOdometry(initial_pose=initial_pose)

    current_pos_truth = poses[0].position
    current_rot_truth = poses[0].rotation_matrix

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

        if prev_list is not None:
            print(f"\nProcessing visual odometry: {frame_idx}")
            previous_ids = [feature.landmark_id for feature in prev_list]
            current_ids = [feature.landmark_id for feature in list_of_observations]

            prev_features = []
            for feature in prev_list:
                prev_features.append(feature.image_coords)
            prev_features = np.array(prev_features)

            current_features = []
            for feature in list_of_observations:
                current_features.append(feature.image_coords)
            new_features = np.array(current_features)

            camera_pose = visual_odometry.run_visual_odometry(
                timestamp=pose.timestamp,
                prev_features=prev_features,
                prev_features_ids=previous_ids,
                new_features=new_features,
                new_features_ids=current_ids,
                camera_matrix=camera.get_camera_matrix(),
            )

            # Position check: We normalize truth delta position between frames as the visual odometry only provides delta position up to scale
            pos_truth = pose.position - prev_pose.position
            pos_truth = pos_truth / np.linalg.norm(pos_truth)
            pos_truth = poses[0].rotation_matrix.transpose() @ pos_truth
            current_pos_truth = current_pos_truth + pos_truth
            current_rot_truth = pose.rotation_matrix.transpose() @ (
                poses[0].rotation_matrix
            )

            print("camera_pose position truth:\n", current_pos_truth)
            print("camera_pose position:\n", camera_pose.position)

            # Orientaiton check:
            print(
                "camera_pose orientation truth:\n",
                GeometryUtils.euler_angles_from_rotation_matrix(current_rot_truth),
            )

            print(
                "camera_pose orientation:\n",
                GeometryUtils.euler_angles_from_rotation_matrix(
                    camera_pose.rotation_matrix
                ),
            )

        prev_list = list_of_observations
        prev_pose = pose

        image = ImageObservations.to_opencv_image(image_observations)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
