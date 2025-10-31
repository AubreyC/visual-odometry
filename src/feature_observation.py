"""Feature detection and tracking utilities."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


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
