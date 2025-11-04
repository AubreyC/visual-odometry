# Add src directory to path for imports

from typing import List

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import FeatureObservation, ImageObservations
from .geometry import GeometryUtils
from .image_renderer import ImageRenderer
from .landmarks import LandmarkGenerator
from .visualization import OpenCVSceneVisualizer, Visualizer


def main() -> None:
    # Generate landmarks and poses
    landmarks_generator: LandmarkGenerator = LandmarkGenerator()
    landmarks: np.ndarray = landmarks_generator.generate_random(
        num_landmarks=500, seed=42
    )

    # landmarks: np.ndarray = np.array(
    #     [
    #         [2.0, 0.0, 0.0],
    #         [2.0, 0.0, 0.3],
    #         [2.0, 0.0, -0.3],
    #         [2.0, 0.2, 0.0],
    #         [2.0, 0.2, 0.3],
    #     ]
    # )

    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )

    # quat_cam = GeometryUtils.quaternion_multiply(
    #     quat_cam,
    #     GeometryUtils.quaternion_from_euler_angles(np.array([0.0, 0.0, 0.0])),
    # )

    poses = [
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.02, 0.0, 0.0])),
            ),
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.04, 0.0, 0.0])),
            ),
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.06, 0.0, 0.0])),
            ),
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.08, 0.0, 0.0])),
            ),
            timestamp=3.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=GeometryUtils.quaternion_multiply(
                quat_cam,
                GeometryUtils.quaternion_from_euler_angles(np.array([0.1, 0.0, 0.0])),
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

    # poses_generator: TrajectoryGenerator = TrajectoryGenerator(time_step=0.1)
    # poses: List[CameraPose] = poses_generator.generate_circular_trajectory(
    #     center=np.array([0.0, 0.0]),
    #     radius=2.0,
    #     height=1.0,
    #     num_poses=200,
    #     look_at_center=False,
    # )

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

    # Scene camera pose

    # Show scene from first camera pose
    # for pose in poses:
    #     print("Showing scene from first camera pose...")
    #     opencv_visualizer.show_scene(
    #         scene_camera_pose=scene_camera_pose,
    #         camera_pose=pose,
    #         landmarks=landmarks,
    #         landmark_ids=list(range(len(landmarks))),
    #         window_name="OpenCV 3D Scene Visualizer",
    #         show_axes=True,
    #     )

    cur_R = np.eye(3)
    cur_t = np.zeros(3)
    timestamp = 0.0

    prev_list = None
    prev_pose = None
    for pose in poses:
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
            id_prev = [feature.landmark_id for feature in prev_list]
            id_curr = [feature.landmark_id for feature in list_of_observations]

            ids = list(filter(lambda x: x in id_prev, id_curr))

            list_of_prev = []
            for feature in prev_list:
                if feature.landmark_id in ids:
                    list_of_prev.append(feature.image_coords)
            list_of_prev = np.array(list_of_prev)

            list_of_curr = []
            for feature in list_of_observations:
                if feature.landmark_id in ids:
                    list_of_curr.append(feature.image_coords)
            list_of_curr = np.array(list_of_curr)

            E, mask = cv2.findEssentialMat(
                list_of_prev,
                list_of_curr,
                camera.get_camera_matrix(),
                cv2.RANSAC,
            )
            print("\ncamera.get_camera_matrix():\n", camera.get_camera_matrix())
            inliers1 = list_of_prev[mask]
            inliers2 = list_of_curr[mask]

            _, R, t, mask_pose = cv2.recoverPose(
                E, inliers1, inliers2, camera.get_camera_matrix()
            )

            t = t / np.linalg.norm(t)  # normalize translation
            cur_R = R @ cur_R
            cur_t = cur_t + (cur_R.T @ t).reshape(3)

            delta_pos_I = pose.position - prev_pose.position
            R_CFp_I = prev_pose.rotation_matrix.transpose()
            delta_pos_CF = R_CFp_I.dot(delta_pos_I)

            delta_pos_CF_normalized = delta_pos_CF / np.linalg.norm(delta_pos_CF)
            delta_pos_CF_normalized_recovered = (-R.transpose() @ t).reshape(3)

            R_CFc_I = pose.rotation_matrix.transpose()
            R_CFc_CFp = R_CFc_I @ R_CFp_I.transpose()

            print("delta_pos_CF_normalized:\n", delta_pos_CF_normalized)
            print(
                "delta_pos_CF_curr_normalized_recovered:\n",
                delta_pos_CF_normalized_recovered,
            )

            print(
                "R_CFc_CFp:\n",
                GeometryUtils.euler_angles_from_rotation_matrix(R_CFc_CFp),
            )
            print("R:\n", GeometryUtils.euler_angles_from_rotation_matrix(R))

        prev_list = list_of_observations
        prev_pose = pose

        arr = np.array([feature.image_coords for feature in list_of_observations])
        print(arr.shape)

        image = ImageObservations.to_opencv_image(image_observations)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
