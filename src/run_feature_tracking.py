from pathlib import Path

import cv2

from .euroc_dataset import EurocDatasetReader
from .feature_tracker import FeatureTracker


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

    timestamp = 0.0
    for frame_idx in range(frames_to_process):
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

            cv2.imshow("Feature Tracking", vis)
            key = cv2.waitKey(0)
            if key & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
