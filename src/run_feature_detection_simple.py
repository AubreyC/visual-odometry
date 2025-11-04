from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .camera_pose import CameraPose
from .euroc_dataset import EurocDatasetReader
from .geometry import GeometryUtils
from .visualization import OpenCVSceneVisualizer

# --- Parameters ---
MIN_FEATURES = 80  # minimum features before re-detection
ORB_FEATURES = 200  # max ORB features per detection
MIN_DISTANCE = 10.0  # minimum distance to existing points

orb = cv2.ORB_create(nfeatures=ORB_FEATURES)


# --- Helper functions ---


def to_camera_matrix(intrinsics: np.ndarray) -> np.ndarray:
    result = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    return result


def detect_features(img_gray: np.ndarray) -> np.ndarray:
    """Detect ORB features and return points as Nx1x2 float32 array."""
    keypoints = orb.detect(img_gray, None)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return points


def track_features(
    prev_img: np.ndarray, curr_img: np.ndarray, prev_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track points using KLT optical flow."""
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None)
    return curr_pts, status, err


def add_new_features(
    curr_img: np.ndarray, existing_pts: np.ndarray, min_distance: float = MIN_DISTANCE
) -> np.ndarray:
    """Detect new ORB features and add only those far enough from existing ones."""
    new_pts = detect_features(curr_img)
    if len(existing_pts) == 0:
        return new_pts

    mask = []
    for p in new_pts:
        dists = np.linalg.norm(existing_pts.reshape(-1, 2) - p.ravel(), axis=1)
        mask.append(np.min(dists) > min_distance)

    filtered_pts = new_pts[mask]
    return filtered_pts


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

    # Create OpenCV scene visualizer for real-time 3D visualization
    opencv_visualizer = OpenCVSceneVisualizer(
        image_width=1280, image_height=720, background_color=(255, 255, 255)
    )

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
        position=np.array([-60.0, -60.0, 60.0]),
        orientation=scene_quat_cam,
        timestamp=0.0,
    )

    # Demonstrate the OpenCV visualizer with different camera poses
    print("Demonstrating OpenCV Scene Visualizer...")

    prev_img = None
    prev_pts = None
    feature_ids = None
    feature_ids_counter = 0

    # camera_pose = CameraPose(
    #     position=np.array([0, 0, 0]),
    #     orientation=GeometryUtils.quaternion_from_euler_angles(
    #         np.array([0.0, 0.0, 0.0])
    #     ),
    #     timestamp=0.0,
    # )

    cur_R = np.eye(3)
    cur_t = np.zeros(3)
    timestamp = 0.0
    for frame_idx in range(frames_to_process):
        timestamp = cam0_data.timestamps[frame_idx]
        image_path = reader.get_image_path("cam0", timestamp)

        curr_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        curr_img = cv2.undistort(
            curr_img,
            to_camera_matrix(cam0_data.intrinsics),
            cam0_data.distortion_coeffs,
        )

        if curr_img is None:
            print(f"Failed to load image at {image_path}")
            continue

        if prev_img is None:
            prev_pts = detect_features(curr_img)
            feature_ids = np.arange(len(prev_pts))
            feature_ids_counter += len(prev_pts)
            prev_img = curr_img.copy()
            continue

        # Track existing features
        curr_pts, status, error = track_features(prev_img, curr_img, prev_pts)

        #
        feature_ids = feature_ids[status.ravel() == 1]
        curr_pts = curr_pts[status.ravel() == 1]
        prev_pts = prev_pts[status.ravel() == 1]

        E, mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            to_camera_matrix(cam0_data.intrinsics),
            cv2.RANSAC,
        )
        inliers1 = prev_pts[mask]
        inliers2 = curr_pts[mask]

        _, R, t, mask_pose = cv2.recoverPose(
            E, inliers1, inliers2, to_camera_matrix(cam0_data.intrinsics)
        )

        # Unknown scale:
        t = t / np.linalg.norm(t)  # normalize translation
        # Update current pose
        cur_t = cur_t + cur_R.dot(t).reshape(3)
        cur_R = R.dot(cur_R)
        timestamp = timestamp + 0.1
        print(cur_t.shape)
        print("\nEstimated Rotation Matrix (R):\n", R)
        print("\nEstimated Translation Vector (t):\n", t)

        print("\nEstimated Rotation Matrix (cur_R):\n", cur_R)
        print("\nEstimated Translation Vector (cur_t):\n", cur_t)

        camera_pose = CameraPose(
            position=cur_t,
            orientation=GeometryUtils.quaternion_from_rotation_matrix(cur_R),
            timestamp=timestamp + 0.1,
        )

        # # Show scene from first camera pose
        print("Showing scene from first camera pose...")
        # opencv_visualizer.show_scene(
        #     scene_camera_pose=scene_camera_pose,
        #     camera_pose=camera_pose,
        #     landmarks=np.zeros((0, 3)),
        #     landmark_ids=[],
        #     window_name="OpenCV 3D Scene Visualizer",
        #     show_axes=True,
        # )

        # Create matplotlib visualizer for 3D plots
        # visualizer: Visualizer = Visualizer()
        # fig = visualizer.plot_scene_overview(
        #     landmarks=np.zeros((0, 3)),
        #     poses=[camera_pose],
        #     title="Test Scene",
        #     show_orientation=True,
        #     orientation_scale=0.5,
        # )
        # visualizer.show()

        # If not enough features, detect new ones and merge
        if len(curr_pts) < MIN_FEATURES:
            new_pts = add_new_features(curr_img, curr_pts)
            if len(new_pts) > 0:
                curr_pts = np.vstack((curr_pts, new_pts))
                new_ids = np.arange(
                    feature_ids_counter, feature_ids_counter + len(new_pts)
                )
                print(new_ids.shape)
                print(feature_ids.shape)
                feature_ids = np.hstack((feature_ids, new_ids))

        # Draw tracked points
        show_features = True
        if show_features:
            vis = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)

            for p, id in zip(curr_pts, feature_ids):
                colort_tracked = (255, 255, 0)
                colort_untracked = (0, 0, 255)

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

            cv2.imshow("vis", vis)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                break

        # Update for next iteration
        prev_img = curr_img.copy()
        prev_pts = curr_pts.copy()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
