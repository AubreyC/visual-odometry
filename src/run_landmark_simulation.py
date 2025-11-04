from typing import List

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose, TrajectoryGenerator
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

    quat_cam = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )

    quat_cam = GeometryUtils.quaternion_multiply(
        quat_cam,
        GeometryUtils.quaternion_from_euler_angles(np.array([0.0, 0.1, 0.0])),
    )

    poses = [
        CameraPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=quat_cam,
            timestamp=0.0,
        ),
        CameraPose(
            position=np.array([0.0, -0.2, 0.0]),
            orientation=quat_cam,
            timestamp=1.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, 0.3]),
            orientation=quat_cam,
            timestamp=2.0,
        ),
        CameraPose(
            position=np.array([0.0, 0.0, -0.3]),
            orientation=quat_cam,
            timestamp=3.0,
        ),
    ]

    poses_generator: TrajectoryGenerator = TrajectoryGenerator(time_step=0.1)
    poses: List[CameraPose] = poses_generator.generate_circular_trajectory(
        center=np.array([0.0, 0.0]),
        radius=2.0,
        height=1.0,
        num_poses=200,
        look_at_center=False,
    )

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
        poses=[scene_camera_pose],
        title="Test Scene",
        show_orientation=True,
        orientation_scale=0.5,
    )
    visualizer.show()

    # Show scene view:
    show_scene = False
    if show_scene:
        # Create OpenCV scene visualizer for real-time 3D visualization
        opencv_visualizer = OpenCVSceneVisualizer(
            image_width=1280, image_height=720, background_color=(255, 255, 255)
        )

        # Show scene from first camera pose
        for pose in poses:
            opencv_visualizer.show_scene(
                scene_camera_pose=scene_camera_pose,
                camera_pose=pose,
                landmarks=landmarks,
                landmark_ids=list(range(len(landmarks))),
                window_name="OpenCV 3D Scene Visualizer",
                show_axes=True,
            )

    # Show camera images
    show_camera_images = True
    if show_camera_images:
        for frame_index, pose in enumerate(poses):
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

            image = ImageObservations.to_opencv_image(image_observations)
            cv2.imshow(f"Camera frame {frame_index}", image)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
