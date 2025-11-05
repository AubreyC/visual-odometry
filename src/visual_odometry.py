import warnings
from typing import List

import cv2
import numpy as np

from .camera_pose import CameraPose
from .feature_tracker import FeatureTracker
from .geometry import GeometryUtils


class VisualOdometry:
    def __init__(self, initial_pose: CameraPose) -> None:
        self.initial_pose = initial_pose
        self.current_pose = initial_pose

    def run_visual_odometry(
        self,
        timestamp: float,
        prev_features: np.ndarray,
        prev_features_ids: List[int],
        new_features: np.ndarray,
        new_features_ids: List[int],
        camera_matrix: np.ndarray,
    ) -> CameraPose:
        """_summary_

        Args:
            prev_features (np.ndarray): _description_
            prev_features_ids (np.ndarray): _description_
            new_features (np.ndarray): _description_
            new_features_ids (np.ndarray): _description_

        Returns:
            CameraPose: Estimated pose of the current camera
        """

        # Validate the inputs
        FeatureTracker.validate_features_input(prev_features)
        FeatureTracker.validate_features_input(new_features)
        FeatureTracker.validate_features_ids(prev_features_ids)
        FeatureTracker.validate_features_ids(new_features_ids)

        # Find common elements between prev_features and new_features
        prev_features_ids = np.array(prev_features_ids)
        new_features_ids = np.array(new_features_ids)

        common_ids, common_indices_prev, common_indices_new = np.intersect1d(
            prev_features_ids, new_features_ids, return_indices=True
        )

        prev_features = prev_features[common_indices_prev]
        new_features = new_features[common_indices_new]
        if len(prev_features) < 10:
            warnings.warn(
                "Not enough features to estimate pose", UserWarning, stacklevel=2
            )
            return self.initial_pose

        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            prev_features, new_features, camera_matrix, cv2.RANSAC, 0.999, 1.0
        )
        # If no motion (essential matrix is singular) do not initialize the pose
        if np.linalg.norm(E) < 1e-6:
            warnings.warn("Essential matrix is singular", UserWarning, stacklevel=2)
            return self.initial_pose

        # Filter out the features that are not in the mask
        prev_features = prev_features[mask.ravel() == 1]
        new_features = new_features[mask.ravel() == 1]

        # Frame details: Previous frame is F1 and current frame is F2
        # - R: Frame rotation matrix from the F1 (previous frame) to the F2 (current frame)
        # - t: Position of the center of the previous frame expressed in the current frame
        # We have following relation:
        # - X_F1 and X_F2 being the position of a point in the F1 and F2: X_F2 = R * X_F1 + t with .
        # - Center of current frame expressed in previous frame: C2_F1 = - R.transpose() @ t
        _, R, t, mask_pose = cv2.recoverPose(
            E, prev_features, new_features, camera_matrix
        )

        t = t / np.linalg.norm(t)
        R_total = R @ self.current_pose.rotation_matrix
        t_total = self.current_pose.position - (R_total.transpose() @ t).reshape(3)

        # Update the current pose
        self.current_pose = CameraPose(
            position=t_total,
            orientation=GeometryUtils.quaternion_from_rotation_matrix(R_total),
            timestamp=timestamp,
        )

        return self.current_pose
