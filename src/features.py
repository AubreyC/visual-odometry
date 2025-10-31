"""Feature detection and tracking utilities."""

from typing import Dict, List, Optional

import numpy as np


class FeatureObservation:
    """A feature observation in an image."""

    def __init__(
        self,
        landmark_id: int,
        image_coords: np.ndarray,
        camera_id: int,
        timestamp: Optional[float] = None,
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


class Tracks:
    """Container for all feature tracks in a dataset."""

    def __init__(self):
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
