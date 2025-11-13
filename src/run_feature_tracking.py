from pathlib import Path

import cv2
import numpy as np

from .camera_pose import CameraPose
from .euroc_dataset import EurocDatasetReader
from .feature_tracker import FeatureTracker
from .opencv_scene_visualizer import OpenCVSceneVisualizer
from .visual_odometry import VisualOdometry


def main() -> None:
    # Path to dataset
    dataset_path = Path(__file__).parent.parent / "dataset" / "mav0"

    # Initialize dataset reader and feature detector
    print("Loading EuRoC dataset")
    reader = EurocDatasetReader(str(dataset_path))

    # Load cam0 data
    cam0_data = reader.cam0
    total_frames = len(cam0_data.timestamps)
    frames_to_process = total_frames

    # Initialize feature tracker
    feature_tracker = FeatureTracker()

    # Initialize visual odometry with initial pose of the camera looking straight ahead
    initial_pose = CameraPose.create_look_straight_pose()
    visual_odometry: VisualOdometry = VisualOdometry(initial_pose=initial_pose)

    # Create OpenCV scene visualizer for real-time 3D visualization
    scene_camera_pose = CameraPose.create_look_at_pose(
        camera_position=np.array([-10.0, -10.0, 8.0]),
        target_position=np.array([0.0, -5.0, 8.0]),
        timestamp=0.0,
    )
    opencv_visualizer = OpenCVSceneVisualizer(
        image_width=cam0_data.resolution[0],
        image_height=cam0_data.resolution[1],
        fx=200,
        fy=200,
        background_color=(255, 255, 255),
    )

    # Iterate over frames and process each frame
    pts_2d_prev = None
    pts_2d_ids_prev = None
    for frame_idx in range(frames_to_process):
        # Get image from dataset
        timestamp = cam0_data.timestamps[frame_idx]
        image_path = reader.get_image_path("cam0", timestamp)

        curr_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        curr_img = cv2.undistort(
            curr_img,
            cam0_data.get_camera_matrix(),
            cam0_data.distortion_coeffs,
        )

        if curr_img is None:
            print(f"Failed to load image at {image_path}")
            continue

        # Run feature tracker
        pts_2d, pts_2d_ids = feature_tracker.run_tracking(curr_img)

        # Run visual odometry
        if pts_2d_prev is not None:
            # If visual odometry is not initialized, initialize it
            if not visual_odometry.is_initialized():
                visual_odometry.init_visual_odometry(
                    timestamp=frame_idx,
                    pts2d_prev=pts_2d_prev,
                    pts2d_ids_prev=pts_2d_ids_prev,
                    pts2d_new=pts_2d,
                    pts2d_ids_new=pts_2d_ids,
                    camera_matrix=cam0_data.get_camera_matrix(),
                )

            # If visual odometry is initialized, update it
            else:
                visual_odometry.update_visual_odometry(
                    timestamp=frame_idx,
                    pts2d_prev=pts_2d_prev,
                    pts2d_ids_prev=pts_2d_ids_prev,
                    pts2d_new=pts_2d,
                    pts2d_ids_new=pts_2d_ids,
                    camera_matrix=cam0_data.get_camera_matrix(),
                )
        pts_2d_prev = pts_2d.copy()
        pts_2d_ids_prev = pts_2d_ids.copy()

        # Show features and scene
        show_features = True
        if show_features:
            image_features = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
            image_features = feature_tracker.draw_features(
                image_features, pts_2d, pts_2d_ids
            )

            # Arbitrary scale to visualize 3D points and camera pose
            scale = 0.1
            pts_3d, pts_3d_ids = visual_odometry.get_last_used_points_3d()
            pts_3d = pts_3d * scale
            camera_pose = visual_odometry.get_current_pose()
            camera_pose.position = camera_pose.position * scale

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
            key = cv2.waitKey(1)
            if key & 0xFF == ord("w"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
