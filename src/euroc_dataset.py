"""EuRoC dataset reader and parser for visual-inertial odometry."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from .validation_error import ProcessingError, ValidationError


@dataclass
class CameraData:
    """Container for camera data including timestamps, filenames, and calibration."""

    timestamps: np.ndarray  # Shape: (N,)
    filenames: List[str]  # Length: N
    intrinsics: np.ndarray  # Shape: (4,) - [fx, fy, cx, cy]
    distortion_coeffs: np.ndarray  # Shape: (4,) - [k1, k2, p1, p2]
    resolution: Tuple[int, int]  # (width, height)
    T_body_sensor: np.ndarray  # Shape: (4, 4) - transform from body to sensor frame
    rate_hz: float

    def get_camera_matrix(self) -> np.ndarray:
        """Get the camera matrix for the camera.

        Returns:
            np.ndarray: _description_
        """
        result: np.ndarray = np.array(
            [
                [self.intrinsics[0], 0, self.intrinsics[2]],
                [0, self.intrinsics[1], self.intrinsics[3]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return result


@dataclass
class IMUData:
    """Container for IMU data including measurements and calibration."""

    timestamps: np.ndarray  # Shape: (N,)
    angular_velocity: np.ndarray  # Shape: (N, 3) - [wx, wy, wz] in rad/s
    linear_acceleration: np.ndarray  # Shape: (N, 3) - [ax, ay, az] in m/s^2
    T_body_sensor: np.ndarray  # Shape: (4, 4) - transform from body to sensor frame
    rate_hz: float
    gyro_noise_density: float
    gyro_random_walk: float
    accel_noise_density: float
    accel_random_walk: float


@dataclass
class GroundTruthData:
    """Container for ground truth pose data."""

    timestamps: np.ndarray  # Shape: (N,)
    positions: np.ndarray  # Shape: (N, 3) - [x, y, z] in world frame
    orientations: np.ndarray  # Shape: (N, 4) - [qw, qx, qy, qz] quaternions
    velocities: np.ndarray  # Shape: (N, 3) - [vx, vy, vz] in world frame
    gyro_bias: np.ndarray  # Shape: (N, 3) - gyroscope bias
    accel_bias: np.ndarray  # Shape: (N, 3) - accelerometer bias


class EurocDatasetReader:
    """Reader for EuRoC MAV dataset files."""

    def __init__(self, dataset_path: str):
        """Initialize the dataset reader.

        Args:
            dataset_path: Path to the EuRoC dataset directory (e.g., 'mav0')

        Raises:
            ValidationError: If dataset path is invalid or required files are missing.
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise ValidationError(f"Dataset path does not exist: {dataset_path}")

        # Check for required directories
        required_dirs = ["cam0", "cam1", "imu0"]
        for dir_name in required_dirs:
            if not (self.dataset_path / dir_name).exists():
                raise ValidationError(
                    f"Required directory {dir_name} not found in dataset"
                )

        self._cam0_data: Optional[CameraData] = None
        self._cam1_data: Optional[CameraData] = None
        self._imu_data: Optional[IMUData] = None
        self._ground_truth_data: Optional[GroundTruthData] = None

    def load_camera_data(self, camera_name: str) -> CameraData:
        """Load camera data for specified camera (cam0 or cam1).

        Args:
            camera_name: Either 'cam0' or 'cam1'

        Returns:
            CameraData object containing timestamps, filenames, and calibration

        Raises:
            ValidationError: If camera data files are missing or malformed.
            ProcessingError: If data parsing fails.
        """
        if camera_name not in ["cam0", "cam1"]:
            raise ValidationError(
                f"Invalid camera name: {camera_name}. Must be 'cam0' or 'cam1'"
            )

        camera_dir = self.dataset_path / camera_name

        # Load sensor configuration
        sensor_yaml_path = camera_dir / "sensor.yaml"
        if not sensor_yaml_path.exists():
            raise ValidationError(
                f"Sensor configuration file not found: {sensor_yaml_path}"
            )

        try:
            with open(sensor_yaml_path) as f:
                sensor_config = yaml.safe_load(f)
        except Exception as e:
            raise ProcessingError(
                f"Failed to parse sensor config {sensor_yaml_path}: {e}"
            )

        # Load data CSV
        data_csv_path = camera_dir / "data.csv"
        if not data_csv_path.exists():
            raise ValidationError(f"Data file not found: {data_csv_path}")

        try:
            timestamps = []
            filenames = []
            with open(data_csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and not row[0].startswith("#"):
                        timestamps.append(int(row[0]))
                        filenames.append(row[1])
            timestamps = np.array(timestamps, dtype=np.int64)
        except Exception as e:
            raise ProcessingError(
                f"Failed to parse camera data CSV {data_csv_path}: {e}"
            )

        # Extract calibration parameters
        try:
            intrinsics = np.array(sensor_config["intrinsics"])  # [fx, fy, cx, cy]
            distortion_coeffs = np.array(
                sensor_config["distortion_coefficients"]
            )  # [k1, k2, p1, p2]
            resolution = tuple(sensor_config["resolution"])  # [width, height]
            rate_hz = sensor_config["rate_hz"]

            # Transform matrix (body to sensor)
            T_BS_data = sensor_config["T_BS"]["data"]
            T_body_sensor = np.array(T_BS_data).reshape(4, 4)

        except KeyError as e:
            raise ValidationError(
                f"Missing required calibration parameter in {sensor_yaml_path}: {e}"
            )

        return CameraData(
            timestamps=timestamps,
            filenames=filenames,
            intrinsics=intrinsics,
            distortion_coeffs=distortion_coeffs,
            resolution=resolution,
            T_body_sensor=T_body_sensor,
            rate_hz=rate_hz,
        )

    def load_imu_data(self) -> IMUData:
        """Load IMU data from imu0 directory.

        Returns:
            IMUData object containing measurements and calibration

        Raises:
            ValidationError: If IMU data files are missing or malformed.
            ProcessingError: If data parsing fails.
        """
        imu_dir = self.dataset_path / "imu0"

        # Load sensor configuration
        sensor_yaml_path = imu_dir / "sensor.yaml"
        if not sensor_yaml_path.exists():
            raise ValidationError(
                f"Sensor configuration file not found: {sensor_yaml_path}"
            )

        try:
            with open(sensor_yaml_path) as f:
                sensor_config = yaml.safe_load(f)
        except Exception as e:
            raise ProcessingError(
                f"Failed to parse sensor config {sensor_yaml_path}: {e}"
            )

        # Load data CSV
        data_csv_path = imu_dir / "data.csv"
        if not data_csv_path.exists():
            raise ValidationError(f"Data file not found: {data_csv_path}")

        try:
            timestamps = []
            angular_velocity_data = []
            linear_acceleration_data = []
            with open(data_csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and not row[0].startswith("#"):
                        timestamps.append(int(row[0]))
                        angular_velocity_data.append(
                            [float(row[1]), float(row[2]), float(row[3])]
                        )
                        linear_acceleration_data.append(
                            [float(row[4]), float(row[5]), float(row[6])]
                        )
            timestamps = np.array(timestamps, dtype=np.int64)
            angular_velocity = np.array(angular_velocity_data)
            linear_acceleration = np.array(linear_acceleration_data)
        except Exception as e:
            raise ProcessingError(f"Failed to parse IMU data CSV {data_csv_path}: {e}")

        # Extract calibration parameters
        try:
            rate_hz = sensor_config["rate_hz"]
            gyro_noise_density = sensor_config["gyroscope_noise_density"]
            gyro_random_walk = sensor_config["gyroscope_random_walk"]
            accel_noise_density = sensor_config["accelerometer_noise_density"]
            accel_random_walk = sensor_config["accelerometer_random_walk"]

            # Transform matrix (body to sensor)
            T_BS_data = sensor_config["T_BS"]["data"]
            T_body_sensor = np.array(T_BS_data).reshape(4, 4)

        except KeyError as e:
            raise ValidationError(
                f"Missing required calibration parameter in {sensor_yaml_path}: {e}"
            )

        return IMUData(
            timestamps=timestamps,
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
            T_body_sensor=T_body_sensor,
            rate_hz=rate_hz,
            gyro_noise_density=gyro_noise_density,
            gyro_random_walk=gyro_random_walk,
            accel_noise_density=accel_noise_density,
            accel_random_walk=accel_random_walk,
        )

    def load_ground_truth_data(self) -> GroundTruthData:
        """Load ground truth pose data from state_groundtruth_estimate0 directory.

        Returns:
            GroundTruthData object containing pose information

        Raises:
            ValidationError: If ground truth data files are missing or malformed.
            ProcessingError: If data parsing fails.
        """
        gt_dir = self.dataset_path / "state_groundtruth_estimate0"

        # Check if ground truth data exists
        if not gt_dir.exists():
            raise ValidationError(f"Ground truth directory not found: {gt_dir}")

        data_csv_path = gt_dir / "data.csv"
        if not data_csv_path.exists():
            raise ValidationError(f"Ground truth data file not found: {data_csv_path}")

        try:
            timestamps = []
            positions_data = []
            orientations_data = []
            velocities_data = []
            gyro_bias_data = []
            accel_bias_data = []
            with open(data_csv_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and not row[0].startswith("#"):
                        timestamps.append(int(row[0]))
                        positions_data.append(
                            [float(row[1]), float(row[2]), float(row[3])]
                        )
                        orientations_data.append(
                            [float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                        )
                        velocities_data.append(
                            [float(row[8]), float(row[9]), float(row[10])]
                        )
                        gyro_bias_data.append(
                            [float(row[11]), float(row[12]), float(row[13])]
                        )
                        accel_bias_data.append(
                            [float(row[14]), float(row[15]), float(row[16])]
                        )
            timestamps = np.array(timestamps, dtype=np.int64)
            positions = np.array(positions_data)
            orientations = np.array(orientations_data)
            velocities = np.array(velocities_data)
            gyro_bias = np.array(gyro_bias_data)
            accel_bias = np.array(accel_bias_data)
        except Exception as e:
            raise ProcessingError(
                f"Failed to parse ground truth data CSV {data_csv_path}: {e}"
            )

        return GroundTruthData(
            timestamps=timestamps,
            positions=positions,
            orientations=orientations,
            velocities=velocities,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
        )

    @property
    def cam0(self) -> CameraData:
        """Get cam0 data, loading it if necessary."""
        if self._cam0_data is None:
            self._cam0_data = self.load_camera_data("cam0")
        return self._cam0_data

    @property
    def cam1(self) -> CameraData:
        """Get cam1 data, loading it if necessary."""
        if self._cam1_data is None:
            self._cam1_data = self.load_camera_data("cam1")
        return self._cam1_data

    @property
    def imu(self) -> IMUData:
        """Get IMU data, loading it if necessary."""
        if self._imu_data is None:
            self._imu_data = self.load_imu_data()
        return self._imu_data

    @property
    def ground_truth(self) -> GroundTruthData:
        """Get ground truth data, loading it if necessary."""
        if self._ground_truth_data is None:
            self._ground_truth_data = self.load_ground_truth_data()
        return self._ground_truth_data

    def get_image_path(self, camera_name: str, timestamp: int) -> Path:
        """Get the full path to an image file given camera name and timestamp.

        Args:
            camera_name: Either 'cam0' or 'cam1'
            timestamp: Timestamp in nanoseconds

        Returns:
            Path to the image file

        Raises:
            ValidationError: If camera name is invalid or timestamp not found.
        """
        if camera_name not in ["cam0", "cam1"]:
            raise ValidationError(
                f"Invalid camera name: {camera_name}. Must be 'cam0' or 'cam1'"
            )

        camera_data = self.cam0 if camera_name == "cam0" else self.cam1

        # Find the index of the timestamp
        idx = np.where(camera_data.timestamps == timestamp)[0]
        if len(idx) == 0:
            raise ValidationError(
                f"Timestamp {timestamp} not found in {camera_name} data"
            )

        filename = camera_data.filenames[idx[0]]
        return self.dataset_path / camera_name / "data" / filename
