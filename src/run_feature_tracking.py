from pathlib import Path

import cv2
import numpy as np

from .camera_pose import CameraPose
from .euroc_dataset import EurocDatasetReader
from .feature_tracker import FeatureTracker
from .geometry import GeometryUtils
from .visual_odometry import VisualOdometry
from .visualization import OpenCVSceneVisualizer, Visualizer


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

    feature_tracker = FeatureTracker()

    visualizer: Visualizer = Visualizer()

    initial_pose = CameraPose(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=GeometryUtils.quaternion_from_euler_angles(
            np.array([-np.pi / 2, 0.0, -np.pi / 2])
        ),
        timestamp=0.0,
    )

    # scene_quat_cam = GeometryUtils.quaternion_from_euler_angles(
    #     np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    # )
    # scene_quat_cam = GeometryUtils.quaternion_multiply(
    #     scene_quat_cam,
    #     GeometryUtils.quaternion_from_euler_angles(np.array([0.0, -np.pi / 4, 0.0])),
    # )

    # scene_camera_pose = CameraPose(
    #     position=np.array([-5.0, -10.0, 8.0]),
    #     orientation=scene_quat_cam,
    #     timestamp=0.0,
    # )

    camera_quat_offset = GeometryUtils.quaternion_from_euler_angles(
        np.array([-np.pi / 2, 0.0, -np.pi / 2]),
    )

    scene_camera_pose = CameraPose.create_look_at_pose(
        camera_position=np.array([-10.0, -10.0, 8.0]),
        target_position=np.array([0.0, 0.0, 5.0]),
        timestamp=0.0,
    )

    visual_odometry: VisualOdometry = VisualOdometry(initial_pose=initial_pose)

    # Create OpenCV scene visualizer for real-time 3D visualization
    opencv_visualizer = OpenCVSceneVisualizer(
        image_width=1280,
        image_height=720,
        fx=200,
        fy=200,
        background_color=(255, 255, 255),
    )

    timestamp = 0.0

    prev_features = None
    prev_features_ids = None
    fig = None
    for frame_idx in range(frames_to_process):
        # if frame_idx < 1000:
        #     continue

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

        curr_pts, feature_ids = feature_tracker.run_tracking(curr_img)
        print(f"Current number of features:{len(curr_pts)}")

        if prev_features is not None:
            if not visual_odometry.is_initialized():
                visual_odometry.init_visual_odometry(
                    timestamp=frame_idx,
                    pts2d_prev=prev_features,
                    pts2d_ids_prev=prev_features_ids,
                    pts2d_new=curr_pts,
                    pts2d_ids_new=feature_ids,
                    camera_matrix=cam0_data.get_camera_matrix(),
                )

            else:
                visual_odometry.update_visual_odometry(
                    timestamp=frame_idx,
                    pts2d_prev=prev_features,
                    pts2d_ids_prev=prev_features_ids,
                    pts2d_new=curr_pts,
                    pts2d_ids_new=feature_ids,
                    camera_matrix=cam0_data.get_camera_matrix(),
                )

        prev_features = curr_pts.copy()
        prev_features_ids = feature_ids.copy()

        # Draw tracked points
        show_features = True
        if show_features:
            vis = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)

            for p, id in zip(curr_pts, feature_ids):
                colort_tracked = (255, 255, 0)

                cv2.circle(
                    vis,
                    (int(p[0]), int(p[1])),
                    2,
                    colort_tracked,
                    -1,
                )
                cv2.putText(
                    vis,
                    str(id),
                    (int(p[0]) + 5, int(p[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

            print("visual_odometry.points_3d: ", visual_odometry.points_3d.shape)
            scale = 0.1
            pts_3d, pts_3d_ids = visual_odometry.get_last_used_points_3d()
            pts_3d = pts_3d * scale
            camera_pose = visual_odometry.get_current_pose()
            camera_pose.position = camera_pose.position * scale

            print("camera_pose: ", camera_pose.position)
            # pts_3d = np.empty((0, 3))
            # if fig is None:
            #     fig = visualizer.plot_scene_overview(
            #         landmarks=pts_3d,
            #         poses=[camera_pose],
            #         poses_second=[],
            #         title="Test Scene",
            #         show_orientation=True,
            #         orientation_scale=0.5,
            #     )
            # else:
            #     fig = visualizer.plot_scene_overview(
            #         landmarks=pts_3d,
            #         poses=[camera_pose],
            #         poses_second=[],
            #         title="Test Scene",
            #         show_orientation=True,
            #         orientation_scale=0.5,
            #         fig=fig,
            #         ax=fig.axes[0],
            #     )

            # visualizer.show()

            image = opencv_visualizer.show_scene_static(
                scene_camera_pose=scene_camera_pose,
                camera_pose=camera_pose,
                landmarks=pts_3d,
                landmark_ids=pts_3d_ids,
                show_axes=True,
            )

            pts_3d_unused, pts_3d_ids_unused = (
                visual_odometry.get_last_unused_points_3d()
            )
            pts_3d_unused = pts_3d_unused * scale

            # image = opencv_visualizer.draw_landmarks(
            #     image,
            #     pts_3d_unused,
            #     scene_camera_pose,
            #     cross_size=10,
            #     color=(0, 255, 0),
            #     thickness=1,
            #     landmark_ids=pts_3d_ids_unused,
            # )

            cv2.imshow("Feature Tracking", vis)
            cv2.imshow("3D Scene", image)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("w"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
