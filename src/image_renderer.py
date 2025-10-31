from typing import Dict, List, Optional

import numpy as np

from .camera import PinHoleCamera, ValidationError
from .camera_pose import CameraPose
from .feature_observation import FeatureObservation, FeatureTrack, ImageObservations


class ImageRenderer:
    """Renders synthetic images and feature tracks from 3D landmarks and camera poses."""

    def __init__(
        self, camera: PinHoleCamera, image_width: int = 640, image_height: int = 480
    ):
        """Initialize image renderer.

        Args:
            camera (PinHoleCamera): Camera model with intrinsics.
            image_width (int): Image width in pixels.
            image_height (int): Image height in pixels.
        """
        self.camera = camera
        self.image_width = image_width
        self.image_height = image_height

    def project_landmarks_to_image(
        self,
        landmarks: np.ndarray,
        camera_pose: CameraPose,
        camera_id: int = 0,
        landmark_ids: Optional[List[int]] = None,
    ) -> List[FeatureObservation]:
        """Project 3D landmarks to 2D image coordinates for given camera pose.

        Args:
            landmarks (np.ndarray): 3D landmarks of shape (N, 3).
            camera_pose (CameraPose): Camera pose for projection.
            landmark_ids (Optional[List[int]]): IDs for landmarks. If None, uses indices.

        Returns:
            List[FeatureObservation]: List of valid feature observations.
        """

        # Validate landmarks
        if not isinstance(landmarks, np.ndarray):
            raise ValidationError(
                f"landmarks must be a numpy array, got {type(landmarks)}"
            )

        if landmarks.ndim != 2 or landmarks.shape[1] != 3:
            raise ValidationError(
                f"landmarks must have shape (N, 3), got {landmarks.shape}"
            )

        if landmarks.shape[0] == 0:
            raise ValidationError("landmarks cannot be empty")

        if not np.all(np.isfinite(landmarks)):
            raise ValidationError("landmarks must contain only finite values")

        if landmark_ids is None:
            landmark_ids = list(range(len(landmarks)))

        if len(landmark_ids) != len(landmarks):
            raise ValidationError(
                f"landmark_ids length ({len(landmark_ids)}) must match landmarks length ({len(landmarks)})"
            )

        observations = []

        # Transform landmarks to camera coordinates
        landmarks_camera = camera_pose.transform_points_world_to_camera(landmarks)

        # Project to image plane
        try:
            image_coords = self.camera.project(landmarks_camera)
        except Exception:
            # If projection fails for any landmark, return empty list
            return []

        if landmark_ids is None:
            landmark_ids = list(range(len(landmarks)))

        # Filter observations that are within image bounds and in front of camera
        for landmark_id, landmark_camera, img_coord in zip(
            landmark_ids, landmarks_camera, image_coords
        ):
            # Check if landmark is in front of camera (positive Z in camera frame)
            if landmark_camera[2] <= 0:
                continue

            # Check if image coordinates are within bounds
            u, v = img_coord
            if not (0 <= u < self.image_width and 0 <= v < self.image_height):
                continue

            observation = FeatureObservation(
                landmark_id=landmark_id,
                image_coords=img_coord,
                camera_id=camera_id,
                timestamp=camera_pose.timestamp,
            )
            observations.append(observation)

        return observations

    def render_image(
        self,
        landmarks: np.ndarray,
        camera_pose: CameraPose,
        camera_id: int = 0,
        landmark_ids: Optional[List[int]] = None,
    ) -> ImageObservations:
        """Render a synthetic image from landmarks and camera pose.

        Args:
            landmarks (np.ndarray): 3D landmarks of shape (N, 3).
            camera_pose (CameraPose): Camera pose.
            landmark_ids (Optional[List[int]]): IDs for landmarks.

        Returns:
            ImageObservations: Image features with observations.
        """
        observations = self.project_landmarks_to_image(
            landmarks, camera_pose, camera_id=camera_id, landmark_ids=landmark_ids
        )

        return ImageObservations(
            camera_id=0,  # Default camera ID
            timestamp=camera_pose.timestamp,
            feature_observations=observations,
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def generate_feature_tracks(
        self,
        landmarks: np.ndarray,
        camera_poses: List[CameraPose],
        landmark_ids: Optional[List[int]] = None,
        min_track_length: int = 2,
    ) -> List[FeatureTrack]:
        """Generate feature tracks from landmarks and camera pose sequence.

        Args:
            landmarks (np.ndarray): 3D landmarks of shape (N, 3).
            camera_poses (List[CameraPose]): Sequence of camera poses.
            landmark_ids (Optional[List[int]]): IDs for landmarks.
            min_track_length (int): Minimum track length to include.

        Returns:
            List[FeatureTrack]: List of feature tracks.
        """
        if landmark_ids is None:
            landmark_ids = list(range(len(landmarks)))

        # Initialize tracks for each landmark
        tracks = []
        for landmark_id in landmark_ids:
            track = FeatureTrack(landmark_id)
            tracks.append(track)

        # Generate observations for each pose
        for pose in camera_poses:
            observations = self.project_landmarks_to_image(
                landmarks, pose, camera_id=0, landmark_ids=landmark_ids
            )

            # Add observations to corresponding tracks
            for obs in observations:
                tracks[obs.landmark_id].add_observation(obs)

        # Filter tracks by minimum length
        valid_tracks = [track for track in tracks if track.length() >= min_track_length]

        return valid_tracks

    def render_image_sequence(
        self,
        landmarks: np.ndarray,
        camera_poses: List[CameraPose],
        landmark_ids: Optional[List[int]] = None,
    ) -> List[ImageObservations]:
        """Render a sequence of synthetic images.

        Args:
            landmarks (np.ndarray): 3D landmarks of shape (N, 3).
            camera_poses (List[CameraPose]): Sequence of camera poses.
            landmark_ids (Optional[List[int]]): IDs for landmarks.

        Returns:
            List[ImageObservations]: List of synthetic images.
        """
        images = []

        for pose in camera_poses:
            image = self.render_image(landmarks, pose, landmark_ids)
            images.append(image)

        return images

    def get_track_statistics(self, tracks: List[FeatureTrack]) -> Dict[str, float]:
        """Get statistics about feature tracks.

        Args:
            tracks (List[FeatureTrack]): List of feature tracks.

        Returns:
            Dict[str, float]: Statistics dictionary.
        """
        if not tracks:
            return {
                "num_tracks": 0,
                "avg_track_length": 0.0,
                "max_track_length": 0,
                "min_track_length": 0,
                "total_observations": 0,
            }

        track_lengths = [track.length() for track in tracks]
        total_observations = sum(track_lengths)

        return {
            "num_tracks": len(tracks),
            "avg_track_length": float(np.mean(track_lengths)),
            "max_track_length": int(np.max(track_lengths)),
            "min_track_length": int(np.min(track_lengths)),
            "total_observations": total_observations,
        }
