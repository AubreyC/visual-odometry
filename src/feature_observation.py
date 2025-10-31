from typing import List, Optional

import numpy as np

from .camera_pose import CameraPose


class FeatureObservation:
    """Represents a 2D feature observation in an image."""

    def __init__(
        self,
        landmark_id: int,
        image_coords: np.ndarray,
        landmark_3d: np.ndarray,
        camera_pose: CameraPose,
        timestamp: Optional[float] = None,
    ):
        """Initialize feature observation.

        Args:
            landmark_id (int): ID of the observed landmark.
            image_coords (np.ndarray): 2D coordinates in image plane [u, v].
            landmark_3d (np.ndarray): 3D position of the landmark in world coordinates.
            camera_pose (CameraPose): Camera pose when observation was made.
            timestamp (Optional[float]): Timestamp of observation.
        """
        self.landmark_id = landmark_id
        self.image_coords = image_coords.copy()
        self.landmark_3d = landmark_3d.copy()
        self.camera_pose = camera_pose
        self.timestamp = timestamp

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureObservation(landmark_id={self.landmark_id}, "
            f"coords=[{self.image_coords[0]:.1f}, {self.image_coords[1]:.1f}])"
        )


class FeatureTrack:
    """Represents a track of 2D feature observations of the same 3D landmark across multiple frames."""

    def __init__(self, landmark_id: int, landmark_3d: np.ndarray):
        """Initialize feature track.

        Args:
            landmark_id (int): ID of the landmark this track represents.
            landmark_3d (np.ndarray): 3D position of the landmark in world coordinates.
        """
        self.landmark_id = landmark_id
        self.landmark_3d = landmark_3d.copy()
        self.observations: List[FeatureObservation] = []

    def add_observation(self, observation: FeatureObservation) -> None:
        """Add an observation to this track.

        Args:
            observation (FeatureObservation): The observation to add.
        """
        from .camera import ValidationError

        if observation.landmark_id != self.landmark_id:
            raise ValidationError(
                f"Observation landmark_id ({observation.landmark_id}) does not match "
                f"track landmark_id ({self.landmark_id})"
            )
        self.observations.append(observation)

    def get_observations_at_time(
        self, timestamp: float
    ) -> Optional[FeatureObservation]:
        """Get observation at specific timestamp.

        Args:
            timestamp (float): Timestamp to search for.

        Returns:
            Optional[FeatureObservation]: Observation at timestamp, or None if not found.
        """
        for obs in self.observations:
            if obs.timestamp is not None and abs(obs.timestamp - timestamp) < 1e-6:
                return obs
        return None

    def get_image_coordinates(self) -> np.ndarray:
        """Get all image coordinates as Nx2 array.

        Returns:
            np.ndarray: Array of shape (N, 2) with image coordinates.
        """
        if not self.observations:
            return np.empty((0, 2))

        coords = np.array([obs.image_coords for obs in self.observations])
        return coords

    def get_timestamps(self) -> List[Optional[float]]:
        """Get timestamps of all observations.

        Returns:
            List[Optional[float]]: List of timestamps.
        """
        return [obs.timestamp for obs in self.observations]

    def length(self) -> int:
        """Get number of observations in track.

        Returns:
            int: Number of observations.
        """
        return len(self.observations)

    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureTrack(landmark_id={self.landmark_id}, length={self.length()})"


class SyntheticImage:
    """Represents a synthetic image with feature observations."""

    def __init__(
        self,
        camera_pose: CameraPose,
        observations: List[FeatureObservation],
        image_width: int = 640,
        image_height: int = 480,
    ):
        """Initialize synthetic image.

        Args:
            camera_pose (CameraPose): Camera pose for this image.
            observations (List[FeatureObservation]): List of feature observations in this image.
            image_width (int): Image width in pixels.
            image_height (int): Image height in pixels.
        """
        self.camera_pose = camera_pose
        self.observations = observations.copy()
        self.image_width = image_width
        self.image_height = image_height

    def get_visible_landmarks(self) -> List[int]:
        """Get IDs of landmarks visible in this image.

        Returns:
            List[int]: List of landmark IDs.
        """
        return [obs.landmark_id for obs in self.observations]

    def get_observations_for_landmark(
        self, landmark_id: int
    ) -> Optional[FeatureObservation]:
        """Get observation for specific landmark.

        Args:
            landmark_id (int): ID of landmark to find.

        Returns:
            Optional[FeatureObservation]: Observation if found, None otherwise.
        """
        for obs in self.observations:
            if obs.landmark_id == landmark_id:
                return obs
        return None

    def __len__(self) -> int:
        """Number of observations in image."""
        return len(self.observations)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SyntheticImage(pose={self.camera_pose}, "
            f"num_observations={len(self.observations)})"
        )
