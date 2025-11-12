#!/usr/bin/env python3
import sys
from pathlib import Path

from src.euroc_dataset import EurocDatasetReader


def run_dataset_loading() -> bool:
    """Test loading EuRoC dataset components."""
    dataset_path = Path(__file__).parent.parent / "dataset" / "mav0"

    print(f"Testing EuRoC dataset reader with path: {dataset_path}")
    print("=" * 60)

    try:
        reader = EurocDatasetReader(str(dataset_path))

        # Test cam0 loading
        print("Loading cam0 data...")
        cam0_data = reader.cam0
        print(f"  - Number of frames: {len(cam0_data.timestamps)}")
        print(f"  - Resolution: {cam0_data.resolution}")
        print(f"  - Intrinsics: {cam0_data.intrinsics}")
        print(f"  - Rate: {cam0_data.rate_hz} Hz")
        print("-> cam0 data loaded successfully")

        # Test cam1 loading
        print("\nLoading cam1 data...")
        cam1_data = reader.cam1
        print(f"  - Number of frames: {len(cam1_data.timestamps)}")
        print(f"  - Resolution: {cam1_data.resolution}")
        print(f"  - Intrinsics: {cam1_data.intrinsics}")
        print(f"  - Rate: {cam1_data.rate_hz} Hz")
        print("-> cam1 data loaded successfully")

        # Test IMU loading
        print("\nLoading IMU data...")
        imu_data = reader.imu
        print(f"  - Number of measurements: {len(imu_data.timestamps)}")
        print(f"  - Rate: {imu_data.rate_hz} Hz")
        print(f"  - Gyro noise density: {imu_data.gyro_noise_density}")
        print(f"  - Accel noise density: {imu_data.accel_noise_density}")
        print("-> IMU data loaded successfully")

        # Test ground truth loading
        print("\nLoading ground truth data...")
        gt_data = reader.ground_truth
        print(f"  - Number of poses: {len(gt_data.timestamps)}")
        print(
            f"  - Position range: [{gt_data.positions.min():.2f}, {gt_data.positions.max():.2f}]"
        )
        print("-> Ground truth data loaded successfully")

        # Test image path retrieval
        print("\nTesting image path retrieval...")
        first_timestamp = cam0_data.timestamps[0]
        image_path = reader.get_image_path("cam0", first_timestamp)
        print(f"  - First cam0 image path: {image_path}")
        print(f"  - Image exists: {image_path.exists()}")
        print("-> Image path retrieval works")

        print("\n" + "=" * 60)
        print("Dataset loaded successfully")
        return True

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        return False


if __name__ == "__main__":
    success = run_dataset_loading()
    sys.exit(0 if success else 1)
