import warnings
from typing import List, Tuple

import cv2
import numpy as np

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .geometry import GeometryUtils
from .validation_helper import ValidationHelper


class Keyframe:
    def __init__(
        self,
        timestamp: float,
        pose: CameraPose,
        camera_matrix: np.ndarray,
        points_2d: np.ndarray,
        points_2d_ids: np.ndarray,
    ) -> None:
        self.timestamp = timestamp
        self.pose = pose
        self.points_2d = points_2d
        self.points_2d_ids = points_2d_ids


class VisualOdometry:
    def __init__(self, initial_pose: CameraPose) -> None:
        self.current_pose = initial_pose
        self.points_3d = np.empty((0, 3))
        self.points_3d_ids = np.empty((0,), dtype=int)
        self.initialized: bool = False

        self.keyframes: List[Keyframe] = []

    @classmethod
    def get_common_pts2d(
        cls,
        pts2d_1: np.ndarray,
        pts2d_ids_1: np.ndarray,
        pts2d_2: np.ndarray,
        pts2d_ids_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get common features between two frames.

        Args:
            pts2d_1 (np.ndarray): _description_
            pts2d_ids_1 (np.ndarray): _description_
            pts2d_2 (np.ndarray): _description_
            pts2d_ids_2 (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Common 2D points, common 2D points ids, common ids
        """

        # Validate the inputs
        ValidationHelper.validate_pts2d(pts2d_1)
        ValidationHelper.validate_pts2d(pts2d_2)
        ValidationHelper.validate_ids(pts2d_ids_1)
        ValidationHelper.validate_ids(pts2d_ids_2)

        # Find common elements between prev_features and new_features
        common_ids, common_indices_1, common_indices_2 = np.intersect1d(
            pts2d_ids_1, pts2d_ids_2, return_indices=True
        )

        pts2d_1_selected = pts2d_1[common_indices_1]
        pts2d_2_selected = pts2d_2[common_indices_2]

        return pts2d_1_selected, pts2d_2_selected, common_ids

    def plot_opencv_image(
        self, image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]
    ) -> None:
        """Plot a set of points on an image.

        Args:
            image (np.ndarray): Image
            points (np.ndarray): Points
            colors (np.ndarray): Colors
        """
        for point in points:
            center = (int(round(point[0])), int(round(point[1])))
            cv2.circle(image, center, 5, color, -1)
        return None

    @classmethod
    def get_common_pts2d_pts3d(
        cls,
        pts2d: np.ndarray,
        pts2d_ids: np.ndarray,
        pts3d: np.ndarray,
        pts3d_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get common 2D points and 3D points.

        Args:
            pts2d (np.ndarray): 2D points
            pts2d_ids (np.ndarray): 2D points ids
            pts3d (np.ndarray): 3D points
            pts3d_ids (np.ndarray): 3D points ids

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Common 2D points, common 3D points, common ids
        """

        # Validate the inputs
        ValidationHelper.validate_pts2d(pts2d)
        ValidationHelper.validate_ids(pts2d_ids)
        ValidationHelper.validate_pts3d(pts3d)
        ValidationHelper.validate_ids(pts3d_ids)

        # Find common elements between prev_features and new_features
        common_ids, common_indices_2d, common_indices_3d = np.intersect1d(
            pts2d_ids, pts3d_ids, return_indices=True
        )

        pts2d_selected = pts2d[common_indices_2d]
        pts3d_selected = pts3d[common_indices_3d]

        return pts2d_selected, pts3d_selected, common_ids

    def is_initialized(self) -> bool:
        """Check if the visual odometry is initialized.

        Returns:
            bool: True if the visual odometry is initialized, False otherwise
        """
        return self.initialized

    def get_pose(self) -> CameraPose:
        """Get the current pose of the camera.

        Returns:
            CameraPose: Current pose of the camera
        """
        return self.current_pose

    def init_visual_odometry(
        self,
        timestamp: float,
        prev_features: np.ndarray,
        prev_features_ids: np.ndarray,
        new_features: np.ndarray,
        new_features_ids: np.ndarray,
        camera_matrix: np.ndarray,
        reprojection_error_threshold: float = 1.0,
    ) -> bool:
        """Run visual odometry to estimate the current pose of the camera.

        Args:
            prev_features (np.ndarray): _description_
            prev_features_ids (np.ndarray): _description_
            new_features (np.ndarray): _description_
            new_features_ids (np.ndarray): _description_

        Returns:
            CameraPose: Estimated pose of the current camera
        """

        if self.initialized:
            warnings.warn(
                "Visual odometry is already initialized", UserWarning, stacklevel=2
            )
            return False

        # Validate the inputs
        ValidationHelper.validate_pts2d(prev_features)
        ValidationHelper.validate_pts2d(new_features)
        ValidationHelper.validate_ids(prev_features_ids)
        ValidationHelper.validate_ids(new_features_ids)

        # Find common elements between prev_features and new_features
        prev_features_selected, new_features_selected, features_ids_selected = (
            self.get_common_pts2d(
                prev_features, prev_features_ids, new_features, new_features_ids
            )
        )

        if len(prev_features_selected) < 5:
            warnings.warn(
                "Not enough features to estimate pose", UserWarning, stacklevel=2
            )
            return self.initialized

        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            prev_features_selected,
            new_features_selected,
            camera_matrix,
            cv2.RANSAC,
            0.999,
            1.0,
        )
        # If no motion (essential matrix is singular) do not initialize the pose
        if np.linalg.norm(E) < 1e-6:
            warnings.warn("Essential matrix is singular", UserWarning, stacklevel=2)
            return self.initialized

        # Filter out the features that are not in the mask
        prev_features_selected = prev_features_selected[mask.ravel() == 1]
        new_features_selected = new_features_selected[mask.ravel() == 1]
        features_ids_selected = features_ids_selected[mask.ravel() == 1]

        # Frame details: Previous frame is F1 and current frame is F2
        # - R: Frame rotation matrix from the F1 (previous frame) to the F2 (current frame)
        # - t: Position of the origin of F1 expressed in the F2
        # We have following relation:
        # - X_F1 and X_F2 being the position of a point in the F1 and F2: X_F2 = R * X_F1 + t with .
        # - Center of current frame expressed in previous frame: C2_F1 = - R.transpose() @ t
        _, R, t, mask_pose = cv2.recoverPose(
            E, prev_features_selected, new_features_selected, camera_matrix
        )

        t = t / np.linalg.norm(t)
        R_total = R @ self.current_pose.rotation_matrix.transpose()
        t_total = self.current_pose.position - (R_total.transpose() @ t).reshape(3)

        previous_pose = self.current_pose

        # Update the current pose
        self.current_pose = CameraPose(
            position=t_total,
            orientation=GeometryUtils.quaternion_from_rotation_matrix(
                R_total.transpose()
            ),
            timestamp=timestamp,
        )

        # Add initial 3D points to the map:
        points_3d, reprojection_error = VisualOdometry.triangulate_points(
            previous_pose.rotation_matrix.transpose(),
            previous_pose.position.reshape(3, 1),
            self.current_pose.rotation_matrix.transpose(),
            self.current_pose.position.reshape(3, 1),
            camera_matrix,
            prev_features_selected,
            new_features_selected,
        )

        # Filter out the points with high reprojection error
        points_3d_selected = points_3d[
            reprojection_error < reprojection_error_threshold
        ]
        points_3d_ids_selected = features_ids_selected[
            reprojection_error < reprojection_error_threshold
        ]

        self.points_3d = np.concatenate((self.points_3d, points_3d_selected))
        self.points_3d_ids = np.concatenate(
            (self.points_3d_ids, np.array(points_3d_ids_selected))
        )

        # Add keyframe to the list of keyframes
        print("new_features_ids:\n", new_features_ids)
        self.keyframes.append(
            Keyframe(
                timestamp,
                self.current_pose,
                camera_matrix,
                new_features,
                new_features_ids,
            )
        )

        self.initialized = True
        return self.initialized

    def update_visual_odometry(
        self,
        timestamp: float,
        prev_features: np.ndarray,
        prev_features_ids: np.ndarray,
        new_features: np.ndarray,
        new_features_ids: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> bool:
        """Update the visual odometry.

        Args:
            timestamp (float): Timestamp of the new frame
            prev_features (np.ndarray): Previous features
            prev_features_ids (np.ndarray): Previous features ids
            new_features (np.ndarray): New features
            new_features_ids (np.ndarray): New features ids
            camera_matrix (np.ndarray): Camera matrix
        """
        print("1")

        if not self.initialized:
            warnings.warn(
                "Visual odometry is not initialized", UserWarning, stacklevel=2
            )
            return False

        # Validate the inputs
        ValidationHelper.validate_pts2d(prev_features)
        ValidationHelper.validate_pts2d(new_features)
        ValidationHelper.validate_ids(prev_features_ids)
        ValidationHelper.validate_ids(new_features_ids)

        print("2")

        # Find common elements between new features and 3D points:
        pts2d_selected, points_3d_selected, selected_ids = self.get_common_pts2d_pts3d(
            new_features, new_features_ids, self.points_3d, self.points_3d_ids
        )

        print("new_features_ids:\n", new_features_ids)
        print("self.points_3d_ids:\n", self.points_3d_ids)
        print("selected_ids:\n", selected_ids)

        if len(pts2d_selected) < 4:
            warnings.warn(
                "Not enough pts2d to estimate pose with SolvePnP",
                UserWarning,
                stacklevel=2,
            )
            print("Not enough pts2d to estimate pose with SolvePnP")
            return False

        print("3")
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_selected, pts2d_selected, camera_matrix, None
        )
        if not success:
            warnings.warn("Failed to solve PnP", UserWarning, stacklevel=2)
            return False

        # Get the rotation and translation from the rotation vector:
        R, _ = cv2.Rodrigues(rvec)

        posCF_F = -R.transpose() @ tvec
        rot_CF_F = R

        self.current_pose = CameraPose(
            position=posCF_F.flatten(),
            orientation=GeometryUtils.quaternion_from_rotation_matrix(
                rot_CF_F.transpose()
            ),
            timestamp=timestamp,
        )

        print("4")
        # Check condition to add new keyframe:
        keyframe_last = self.keyframes[-1]
        quat_delta = GeometryUtils.quaternion_multiply(
            keyframe_last.pose.quaternion,
            GeometryUtils.quaternion_inverse(self.current_pose.quaternion),
        )
        _, angle = GeometryUtils.axis_angle_from_quaternion(quat_delta)

        if np.linalg.norm(
            keyframe_last.pose.position - self.current_pose.position
        ) > 0.1 and angle > np.deg2rad(2):
            print("Adding new keyframe")
            print("keyframe_last.points_2d_ids:\n", keyframe_last.points_2d_ids)

            self.add_pts3d_to_map(
                keyframe_last.points_2d,
                keyframe_last.points_2d_ids,
                keyframe_last.pose,
                new_features,
                new_features_ids,
                self.current_pose,
                camera_matrix,
            )

            self.keyframes.append(
                Keyframe(
                    timestamp,
                    self.current_pose,
                    camera_matrix,
                    new_features,
                    new_features_ids,
                )
            )
        else:
            print("Not adding new keyframe")

        return True

    def add_pts3d_to_map(
        self,
        pts2d_1: np.ndarray,
        pts2d_ids_1: np.ndarray,
        pose_1: CameraPose,
        pts2d_2: np.ndarray,
        pts2d_ids_2: np.ndarray,
        pose_2: CameraPose,
        camera_matrix: np.ndarray,
        reprojection_error_threshold: float = 1.0,
    ) -> None:
        """Add 3D points to the map.

        Args:
            pts2d_1 (np.ndarray): 2D points in frame 1
            pts2d_ids_1 (np.ndarray): 2D points ids in frame 1
            pose_1 (CameraPose): Pose of frame 1
            pts2d_2 (np.ndarray): 2D points in frame 2
            pts2d_ids_2 (np.ndarray): 2D points ids in frame 2
            pose_2 (CameraPose): Pose of frame 2
            camera_matrix (np.ndarray): Camera matrix
            reprojection_error_threshold (float): Reprojection error threshold
        """

        # Validate the inputs
        ValidationHelper.validate_pts2d(pts2d_1)
        ValidationHelper.validate_pts2d(pts2d_2)
        ValidationHelper.validate_ids(pts2d_ids_1)
        ValidationHelper.validate_ids(pts2d_ids_2)

        # Find common elements between new features points:
        pts2d_1_selected, pts2d_2_selected, selected_ids = self.get_common_pts2d(
            pts2d_1, pts2d_ids_1, pts2d_2, pts2d_ids_2
        )
        print("pts2d_ids_1:\n", pts2d_ids_1)
        print("pts2d_ids_2:\n", pts2d_ids_2)

        # Find ids that are not in the 3D points map:
        new_ids = np.setdiff1d(selected_ids, self.points_3d_ids)
        print("selected_ids:\n", selected_ids)
        print("New ids:\n", new_ids)

        if len(new_ids) > 0:
            # Add initial 3D points to the map:

            mask_pts2d_1 = np.isin(pts2d_ids_1, new_ids)
            mask_pts2d_2 = np.isin(pts2d_ids_2, new_ids)

            pts2d_1_selected = pts2d_1[mask_pts2d_1]
            pts2d_2_selected = pts2d_2[mask_pts2d_2]

            points_3d, reprojection_error = VisualOdometry.triangulate_points(
                pose_1.rotation_matrix.transpose(),
                pose_1.position.reshape(3, 1),
                pose_2.rotation_matrix.transpose(),
                pose_2.position.reshape(3, 1),
                camera_matrix,
                pts2d_1_selected,
                pts2d_2_selected,
            )

            # Filter out the points with high reprojection error
            points_3d_selected = points_3d[
                reprojection_error < reprojection_error_threshold
            ]
            points_3d_ids_selected = new_ids[
                reprojection_error < reprojection_error_threshold
            ]

            if len(points_3d_selected) > 0:
                print("Adding points to the map:\n", points_3d_ids_selected)
                self.points_3d = np.concatenate((self.points_3d, points_3d_selected))
                self.points_3d_ids = np.concatenate(
                    (self.points_3d_ids, np.array(points_3d_ids_selected))
                )

        return

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            Tuple[np.ndarray, np.ndarray]: 3D points expressed in frame F, Reprojection error
        """

        # Validate the inputs
        GeometryUtils.validate_rotation_matrix(rot_F1_F)
        GeometryUtils.validate_rotation_matrix(rot_F2_F)

        ValidationHelper.validate_pt3d(originF1_F)
        ValidationHelper.validate_pt3d(originF2_F)

        ValidationHelper.validate_pts2d(pts1)
        ValidationHelper.validate_pts2d(pts2)

        PinHoleCamera.is_valid_camera_matrix(camera_matrix)

        # Need to convert position of the center of the frame to translation vector
        R1 = rot_F1_F
        R2 = rot_F2_F
        t1 = -rot_F1_F @ originF1_F
        t2 = -rot_F2_F @ originF2_F

        P1 = camera_matrix @ np.hstack((R1, t1))
        P2 = camera_matrix @ np.hstack((R2, t2))
        pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts4D /= pts4D[3]  # convert from homogeneous to 3D
        result: np.ndarray = pts4D[:3].T

        # Add reprojection error to the points:
        print("result:\n", result.shape)
        pts1_reprojected, _ = cv2.projectPoints(result, R1, t1, camera_matrix, None)
        pts2_reprojected, _ = cv2.projectPoints(result, R2, t2, camera_matrix, None)

        pts1_reprojected = np.squeeze(pts1_reprojected, axis=1)
        pts2_reprojected = np.squeeze(pts2_reprojected, axis=1)

        reprojection_error = (
            np.linalg.norm(pts1_reprojected - pts1, axis=1)
            + np.linalg.norm(pts2_reprojected - pts2, axis=1)
        ) / 2

        return result, reprojection_error
