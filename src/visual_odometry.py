import warnings
from typing import List, Tuple

import cv2
import numpy as np

from .camera_pose import CameraPose
from .feature_tracker import FeatureTracker
from .geometry import GeometryUtils


class VisualOdometry:
    def __init__(self, initial_pose: CameraPose) -> None:
        self.initial_pose = initial_pose
        self.current_pose = initial_pose

    @classmethod
    def get_common_features(
        cls,
        prev_features: np.ndarray,
        prev_features_ids: List[int],
        new_features: np.ndarray,
        new_features_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get common features between two frames.

        Args:
            prev_features (np.ndarray): _description_
            prev_features_ids (List[int]): _description_
            new_features (np.ndarray): _description_
            new_features_ids (List[int]): _description_

        Returns:
            Tuple[np.ndarray, List[int], List[int]]: _description_
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

        return prev_features, new_features, common_ids

    def run_visual_odometry(
        self,
        timestamp: float,
        prev_features: np.ndarray,
        prev_features_ids: List[int],
        new_features: np.ndarray,
        new_features_ids: List[int],
        camera_matrix: np.ndarray,
    ) -> CameraPose:
        """Run visual odometry to estimate the current pose of the camera.

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
        prev_features, new_features, _ = self.get_common_features(
            prev_features, prev_features_ids, new_features, new_features_ids
        )

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

    @classmethod
    def triangulate_points(
        cls,
        rot_F1_F: np.ndarray,
        originF1_F: np.ndarray,
        rot_F2_F: np.ndarray,
        originF2_F: np.ndarray,
        camera_matrix: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> np.ndarray:
        """Triangulate 2D image coordinates of points between two frames.

        Args:
            rot_F1_F (np.ndarray): Frame rotation matrix from F to the F1
            originF1_F (np.ndarray): Position of the center of the frame F1 expressed in frame F
            rot_F2_F (np.ndarray): Frame rotation matrix from F to the F2
            originF2_F (np.ndarray): Position of the center of the frame F2 expressed in frame F
            camera_matrix (np.ndarray): Camera matrix
            pts1 (np.ndarray): 2D image coordinates of points in on camera 1 (attached to frame F1)
            pts2 (np.ndarray): 2D image coordinates of points in on camera 2 (attached to frame F2)

        Returns:
            np.ndarray: 3D points expressed in frame F
        """

        # Validate the inputs
        GeometryUtils.validate_rotation_matrix(rot_F1_F)
        GeometryUtils.validate_rotation_matrix(rot_F2_F)
        GeometryUtils.validate_3d_point(originF1_F)
        GeometryUtils.validate_3d_point(originF2_F)

        FeatureTracker.validate_features_input(pts1)
        FeatureTracker.validate_features_input(pts2)

        # Need to convert position of the center of the frame to translation vector
        R1 = rot_F1_F
        R2 = rot_F2_F
        t1 = -rot_F1_F @ originF1_F
        t2 = -rot_F2_F @ originF2_F

        P1 = camera_matrix @ np.hstack((R1, t1))
        P2 = camera_matrix @ np.hstack((R2, t2))
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts4D /= pts4D[3]  # convert from homogeneous to 3D
        return pts4D[:3].T
