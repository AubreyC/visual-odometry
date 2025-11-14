"""Feature detection and tracking utilities."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import PinHoleCamera
from .validation_error import ValidationError
from .validation_helper import ValidationHelper


class Features2D:
    def __init__(self, points_2d: np.ndarray, ids: np.ndarray):
        """Initialize the features 2D.

        Args:
            points_2d (np.ndarray): Points 2D of the features 2D.
            ids (np.ndarray): IDs of the features 2D.
        """
        # Validate input: points_2d and ids must be numpy arrays
        ValidationHelper.validate_pts2d(points_2d)
        ValidationHelper.validate_ids(ids)

        self.points_2d = points_2d.copy()
        self.ids = ids.copy()

    def get_points_2d(self) -> np.ndarray:
        """Get the points 2D of the features 2D.

        Returns:
            np.ndarray: Points 2D of the features 2D.
        """
        return self.points_2d.copy()

    def get_ids(self) -> np.ndarray:
        """Get the IDs of the features 2D.

        Returns:
            np.ndarray: IDs of the features 2D.
        """
        return self.ids.copy()

    def get_selected_ids(self, ids: np.ndarray) -> "Features2D":
        """Get the features 2D with the selected IDs.

        Args:
            ids (np.ndarray): IDs of the features to select.

        Returns:
            Features2D: Features 2D with the selected IDs.
        """
        common_ids, common_indices_1, common_indices_2 = np.intersect1d(
            self.ids, ids, return_indices=True
        )
        return Features2D(self.points_2d[common_indices_1], common_ids)

    def copy(self) -> "Features2D":
        """Copy the features 2D.

        Returns:
            Features2D: Copy of the features 2D.
        """
        return Features2D(self.points_2d.copy(), self.ids.copy())

    def add_noise(self, noise_std: float) -> None:
        """Add noise to the feature points.

        Args:
            noise_std (float): Standard deviation of the noise.
        """
        if noise_std <= 0.0:
            return
        self.points_2d += np.random.randn(len(self.points_2d), 2) * noise_std


class Landmarks3D:
    def __init__(self, points_3d: np.ndarray, ids: np.ndarray):
        """Initialize the landmarks 3D.

        Args:
            points_3d (np.ndarray): Points 3D of the landmarks 3D.
            ids (np.ndarray): IDs of the landmarks 3D.
        """
        # Validate input: points_3d and ids must be numpy arrays
        ValidationHelper.validate_pts3d(points_3d)
        ValidationHelper.validate_ids(ids)

        self.points_3d = points_3d.copy()
        self.ids = ids.copy()

    def get_points_3d(self) -> np.ndarray:
        """Get the points 3D of the landmarks 3D.

        Returns:
            np.ndarray: Points 3D of the landmarks 3D.
        """
        return self.points_3d.copy()

    def get_ids(self) -> np.ndarray:
        """Get the IDs of the landmarks 3D.

        Returns:
            np.ndarray: IDs of the landmarks 3D.
        """
        return self.ids.copy()

    def get_selected_ids(self, ids: np.ndarray) -> "Landmarks3D":
        """Get the landmarks 3D with the selected IDs.

        Args:
            ids (np.ndarray): IDs of the landmarks to select.

        Returns:
            Landmarks3D: Landmarks 3D with the selected IDs.
        """
        common_ids, common_indices_1, common_indices_2 = np.intersect1d(
            self.ids, ids, return_indices=True
        )
        return Landmarks3D(self.points_3d[common_indices_1], common_ids)

    def copy(self) -> "Landmarks3D":
        """Copy the landmarks 3D.

        Returns:
            Landmarks3D: Copy of the landmarks 3D.
        """
        return Landmarks3D(self.points_3d.copy(), self.ids.copy())

    def add_noise(self, noise_std: float) -> None:
        """Add noise to the landmarks.

        Args:
            noise_std (float): Standard deviation of the noise.
        """
        if noise_std <= 0.0:
            return
        self.points_3d += np.random.randn(len(self.points_3d), 3) * noise_std


class ImageFeatures:
    """Image features."""

    def __init__(
        self,
        timestamp: float,
        camera_id: int,
        image_width: int,
        image_height: int,
        features_2d: Optional[Features2D] = None,
    ):
        """Initialize the image features.

        Args:
            timestamp (float): Timestamp of the image.
            camera_id (int): ID of the camera.
            image_width (int): Image width in pixels.
            image_height (int): Image height in pixels.
            features_2d (Optional[Features2D]): Features 2D of the image features.
        """
        self.timestamp = timestamp
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        self.features_2d = (
            features_2d
            if features_2d is not None
            else Features2D(np.empty((0, 2)), np.empty((0,), dtype=int))
        )

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_camera_id(self) -> int:
        return self.camera_id

    def get_points_2d(self) -> Features2D:
        return self.features_2d

    def to_opencv_image(
        self,
        image: Optional[np.ndarray] = None,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        feature_color: Tuple[int, int, int] = (0, 0, 255),
        feature_radius: int = 3,
        thickness: int = -1,
        show_landmark_ids: bool = True,
        text_color: Tuple[int, int, int] = (255, 0, 0),
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> Any:
        """Generate an OpenCV image from the ImageFeatures.

        Args:
            background_color (Tuple[int, int, int]): RGB color for background (0-255).
            feature_color (Tuple[int, int, int]): RGB color for feature points (0-255).
            feature_radius (int): Radius of feature point circles.
            thickness (int): Thickness of feature point circles (-1 for filled).
            show_landmark_ids (bool): Whether to display landmark IDs as text.
            text_color (Tuple[int, int, int]): RGB color for landmark ID text (0-255).
            font_scale (float): Font scale for landmark ID text.
            font_thickness (int): Thickness for landmark ID text.

        Returns:
            np.ndarray: OpenCV image as numpy array with shape (height, width, 3) and dtype uint8.
        """
        # Create blank image
        image = (
            np.full(
                (self.image_height, self.image_width, 3),
                background_color,
                dtype=np.uint8,
            )
            if image is None
            else image
        )

        # Validate image
        if image is not None:
            if image.shape != (self.image_height, self.image_width, 3):
                raise ValidationError(
                    f"Image must have shape (height, width, 3), got {image.shape}"
                )
            if image.dtype != np.uint8:
                raise ValidationError(f"Image must have dtype uint8, got {image.dtype}")

        # Draw feature points
        for point_2d, id in zip(
            self.features_2d.get_points_2d(), self.features_2d.get_ids()
        ):
            center = (int(round(point_2d[0])), int(round(point_2d[1])))
            # Draw feature point
            cv2.circle(image, center, feature_radius, feature_color, thickness)

            # Draw landmark ID if requested
            if show_landmark_ids:
                text = str(id)
                # Position text slightly above and to the right of the point
                text_pos = (
                    center[0] + feature_radius + 2,
                    center[1] - feature_radius - 2,
                )
                cv2.putText(
                    image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        return image

    @classmethod
    def from_points_3d(
        cls,
        timestamp: float,
        camera_id: int,
        image_width: int,
        image_height: int,
        camera: PinHoleCamera,
        points_3d: np.ndarray,
        ids: np.ndarray,
    ) -> "ImageFeatures":
        """Project points 3D to 2D image coordinates.

        Args:
            timestamp (float): Timestamp of the image.
            camera_id (int): ID of the camera.
            image_width (int): Image width in pixels.
            image_height (int): Image height in pixels.
            camera (PinHoleCamera): Camera model.
            points_3d (np.ndarray): 3D points to project.
            ids (np.ndarray): IDs of the points.

        Returns:
            ImageFeatures: Image features.
        """

        # Validate input
        ValidationHelper.validate_pts3d(points_3d)
        ValidationHelper.validate_ids(ids)

        # Project 3D points to 2D image coordinates and create features 2D
        points_2d = camera.project(points_3d)
        features_2d = Features2D(points_2d, ids)
        return cls(timestamp, camera_id, image_width, image_height, features_2d)


class FeatureObservation:
    """A feature observation in an image."""

    def __init__(
        self,
        landmark_id: int,
        image_coords: np.ndarray,
        camera_id: int,
        timestamp: float = 0.0,
    ):
        """Initialize feature observation.

        Args:
            landmark_id (int): ID of the landmark this feature corresponds to.
            image_coords (np.ndarray): 2D image coordinates [u, v].
            camera_id (int): ID of the camera that made this observation.
            timestamp (Optional[float]): Timestamp of observation.
        """
        self.landmark_id = landmark_id
        self.image_coords = image_coords.copy()
        self.camera_id = camera_id
        self.timestamp = timestamp


class FeatureTrack:
    """A track of feature observations for a single landmark."""

    def __init__(self, landmark_id: int):
        """Initialize feature track.

        Args:
            landmark_id (int): ID of the landmark this track represents.
        """
        self.landmark_id = landmark_id
        self.observations: List[FeatureObservation] = []

    def add_observation(self, observation: FeatureObservation) -> None:
        """Add an observation to this track.

        Args:
            observation (FeatureObservation): The observation to add.
        """
        if observation.landmark_id != self.landmark_id:
            raise ValueError(
                f"Observation landmark_id ({observation.landmark_id}) does not match track landmark_id ({self.landmark_id})"
            )
        self.observations.append(observation)

    def length(self) -> int:
        """Get the number of observations in this track.

        Returns:
            int: Number of observations.
        """
        return len(self.observations)

    def get_timestamps(self) -> List[float]:
        """Get the timestamps of the observations in this track.

        Returns:
            List[float]: List of timestamps.
        """
        return [obs.timestamp for obs in self.observations]


class Tracks:
    """Container for all feature tracks in a dataset."""

    def __init__(self) -> None:
        """Initialize tracks container."""
        self.tracks: Dict[int, FeatureTrack] = {}
        self.camera_ids: List[int] = []
        self.timestamps: List[float] = []

    def add_track(self, track: FeatureTrack) -> None:
        """Add a feature track.

        Args:
            track (FeatureTrack): The track to add.
        """
        self.tracks[track.landmark_id] = track

    def get_track(self, landmark_id: int) -> Optional[FeatureTrack]:
        """Get track for a specific landmark.

        Args:
            landmark_id (int): ID of the landmark.

        Returns:
            Optional[FeatureTrack]: The track if it exists, None otherwise.
        """
        return self.tracks.get(landmark_id)

    def get_all_tracks(self) -> List[FeatureTrack]:
        """Get all tracks.

        Returns:
            List[FeatureTrack]: List of all tracks.
        """
        return list(self.tracks.values())

    def get_observations_for_camera(self, camera_id: int) -> List[FeatureObservation]:
        """Get all observations for a specific camera.

        Args:
            camera_id (int): ID of the camera.

        Returns:
            List[FeatureObservation]: List of observations for this camera.
        """
        observations = []
        for track in self.tracks.values():
            for obs in track.observations:
                if obs.camera_id == camera_id:
                    observations.append(obs)
        return observations

    def __len__(self) -> int:
        """Number of tracks.

        Returns:
            int: Number of tracks.
        """
        return len(self.tracks)


class ImageObservations:
    def __init__(
        self,
        camera_id: int,
        timestamp: float = 0.0,
        feature_observations: Optional[List[FeatureObservation]] = None,
        image_width: int = 640,
        image_height: int = 480,
    ):
        """Initialize image features.

        Args:
            camera_id (int): Camera ID.
            timestamp (float): Timestamp.
            features (Optional[List[FeatureObservation]]): List of feature observations.
            image_width (int): Image width in pixels.
            image_height (int): Image height in pixels.
        """
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.feature_observations = (
            feature_observations if feature_observations is not None else []
        )
        self.image_width = image_width
        self.image_height = image_height

    def get_visible_landmarks(self) -> List[int]:
        """Get IDs of landmarks visible in this image.

        Returns:
            List[int]: List of landmark IDs.
        """
        result: List[int] = [obs.landmark_id for obs in self.feature_observations]
        return result

    def get_observations_for_landmark(
        self, landmark_id: int
    ) -> Optional[FeatureObservation]:
        """Get observation for specific landmark.

        Args:
            landmark_id (int): ID of landmark to find.

        Returns:
            Optional[FeatureObservation]: Observation if found, None otherwise.
        """
        for obs in self.feature_observations:
            if obs.landmark_id == landmark_id:
                return obs
        return None

    def add_feature(self, feature: FeatureObservation) -> None:
        """Add a feature to the image features.

        Args:
            feature (FeatureObservation): Feature observation.
        """
        self.feature_observations.append(feature)

    def get_features(self) -> List[FeatureObservation]:
        """Get the features of the image features.

        Returns:
            List[FeatureObservation]: List of feature observations.
        """
        return self.feature_observations

    def to_opencv_image(
        self,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        feature_color: Tuple[int, int, int] = (0, 0, 255),
        feature_radius: int = 3,
        thickness: int = -1,
        show_landmark_ids: bool = True,
        text_color: Tuple[int, int, int] = (255, 0, 0),
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> Any:
        """Generate an OpenCV image from the ImageFeatures.

        Args:
            background_color (Tuple[int, int, int]): RGB color for background (0-255).
            feature_color (Tuple[int, int, int]): RGB color for feature points (0-255).
            feature_radius (int): Radius of feature point circles.
            thickness (int): Thickness of feature point circles (-1 for filled).
            show_landmark_ids (bool): Whether to display landmark IDs as text.
            text_color (Tuple[int, int, int]): RGB color for landmark ID text (0-255).
            font_scale (float): Font scale for landmark ID text.
            font_thickness (int): Thickness for landmark ID text.

        Returns:
            np.ndarray: OpenCV image as numpy array with shape (height, width, 3) and dtype uint8.
        """
        # Create blank image
        image = np.full(
            (self.image_height, self.image_width, 3), background_color, dtype=np.uint8
        )

        # Draw feature points
        for observation in self.feature_observations:
            # Convert image coordinates to integer pixels
            u, v = observation.image_coords
            center = (int(round(u)), int(round(v)))

            # Draw feature point
            cv2.circle(image, center, feature_radius, feature_color, thickness)

            # Draw landmark ID if requested
            if show_landmark_ids:
                text = str(observation.landmark_id)
                # Position text slightly above and to the right of the point
                text_pos = (
                    center[0] + feature_radius + 2,
                    center[1] - feature_radius - 2,
                )
                cv2.putText(
                    image,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        return image
