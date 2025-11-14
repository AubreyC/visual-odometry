from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .camera import PinHoleCamera
from .camera_pose import CameraPose
from .feature_observation import Features2D, ImageFeatures, Landmarks3D


class BundleAdjustment:
    """Bundle adjustment class for 3D landmarks and camera poses."""

    def __init__(self) -> None:
        pass

    # Define the residual function
    @staticmethod
    def residual_function(
        params: np.ndarray,
        camera_pose_initial: CameraPose,
        num_poses: int,
        landmark_ids: np.ndarray,
        observations: List[ImageFeatures],
        camera_model: PinHoleCamera,
    ) -> np.ndarray:
        """Residual function for bundle adjustment.

        Args:
            params (np.ndarray): Parameters to optimize: [poses_flat, landmarks_flat] with poses_flat = [pose_0, ..., pose_n] (pose_i = 3 position + 4 quaternion) and landmarks_flat = [landmark_0, ..., landmark_m] (landmark_i = [x, y, z] 3D position).
            camera_pose_initial (CameraPose): Initial camera pose (first camera pose not optimized).
            num_poses (int): Number of poses to optimize.
            landmark_ids (np.ndarray): Landmark IDs.
            observations (List[ImageFeatures]): Observations (image features) size num_poses + 1 (first camera pose not optimized).
            camera_model (PinHoleCamera): Camera model.

        Returns:
            np.ndarray: Residuals (flattened) of size 2 * num_observations [u_1 - u_1_proj, v_1 - v_1_proj, ..., u_n - u_n_proj, v_n - v_n_proj] with n = num_observations.
        """
        # 3 position + 4 quaternion
        pose_size = 7

        # Extract poses and landmarks from params
        landmarks = params[num_poses * pose_size :]
        points_3d = Landmarks3D(landmarks.reshape(-1, 3), landmark_ids)

        # Extract poses and landmarks from params
        residuals = []
        for pose_idx, image_features in enumerate[ImageFeatures](observations):
            if pose_idx == 0:
                camera_pose = camera_pose_initial
            else:
                # Extract pose parameters
                pose_start = (pose_idx - 1) * pose_size
                pose_end = pose_start + pose_size
                pose_data = params[pose_start:pose_end]

                position = pose_data[:3]
                quaternion = pose_data[3:7]

                # Normalize quaternion to ensure it's valid
                quaternion = quaternion / np.linalg.norm(quaternion)

                camera_pose = CameraPose(position, quaternion, pose_idx)

            # Extract landmarks
            points_3d_selected = points_3d.get_selected_ids(
                image_features.get_points_2d().get_ids()
            )

            landmark_cf = camera_pose.transform_points_world_to_camera(
                points_3d_selected.get_points_3d()
            )

            # Project to image plane
            projected = camera_model.project(landmark_cf)
            projected = projected.flatten()

            residual = (
                image_features.get_points_2d().get_points_2d().flatten()
                - projected.flatten()
            )
            residuals.append(residual)

        result: np.ndarray = np.array(residuals).flatten()
        return result

    def optimize(
        self,
        image_features_list: List[ImageFeatures],
        camera_pose_initial: CameraPose,
        camera_model: PinHoleCamera,
        camera_poses_guess: List[CameraPose],
        points3d_guess: Landmarks3D,
    ) -> Tuple[List[ImageFeatures], Landmarks3D, List[CameraPose]]:
        """
        Perform bundle adjustment optimization using scipy.optimize.least_squares.

        Args:
            image_features_list: List of ImageFeatures containing 2D observations
            camera_poses: List of CameraPose objects for each camera
            camera_intrinsics: PinHoleCamera with intrinsic parameters
            initial_points3d: Initial 3D landmark positions

        Returns:
            Tuple of (optimized_image_features, optimized_points3d)
        """

        # Extract data for optimization
        landmark_ids = points3d_guess.get_ids()
        landmark_positions = points3d_guess.get_points_3d()

        # Create parameter vector: [poses_flat, landmarks_flat]
        # Each pose: 3 position + 4 quaternion = 7 parameters
        pose_params: List[float] = []
        n_poses = len(camera_poses_guess)
        for pose in camera_poses_guess:
            pose_params.extend(pose.position)
            pose_params.extend(pose.quaternion)
        landmark_params = landmark_positions.flatten()

        image_width = image_features_list[0].image_width
        image_height = image_features_list[0].image_height

        # Initial parameters: [poses_flat, landmarks_flat]
        initial_params = np.concatenate([pose_params, landmark_params])

        # Perform optimization
        result = least_squares(
            self.residual_function,
            initial_params,
            method="lm",  # Levenberg-Marquardt
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=2000,  # Increase max iterations
            verbose=0,
            args=(
                camera_pose_initial,
                n_poses,
                landmark_ids,
                image_features_list,
                camera_model,
            ),
        )

        # Extract optimized parameters
        optimized_params = result.x
        pose_size = 7

        # Update camera poses
        optimized_camera_poses = [camera_pose_initial]
        for i in range(n_poses):
            pose_start = i * pose_size
            pose_end = pose_start + pose_size
            pose_data = optimized_params[pose_start:pose_end]

            position = pose_data[:3]
            quaternion = pose_data[3:7]
            quaternion = quaternion / np.linalg.norm(quaternion)  # Normalize

            # Keep original timestamp
            original_pose = camera_poses_guess[i]
            optimized_pose = CameraPose(position, quaternion, original_pose.timestamp)
            optimized_camera_poses.append(optimized_pose)

        # Update landmark positions
        landmark_start = n_poses * pose_size
        optimized_landmarks = optimized_params[landmark_start:].reshape(-1, 3)
        optimized_points3d = Landmarks3D(optimized_landmarks, landmark_ids)

        # Reproject to create optimized image features
        optimized_image_features = []
        for pose_idx, image_features in enumerate[ImageFeatures](image_features_list):
            timestamp = image_features.get_timestamp()
            camera_id = image_features.get_camera_id()
            pose = optimized_camera_poses[pose_idx]

            # Get observed landmark IDs for this image
            obs_ids = image_features.get_points_2d().get_ids()

            # Transform and project landmarks
            landmark_positions_opt = optimized_points3d.get_points_3d()
            landmark_camera_coords = pose.transform_points_world_to_camera(
                landmark_positions_opt
            )
            projected_points = camera_model.project(landmark_camera_coords)

            # Create new ImageFeatures with optimized projections
            # Only include landmarks that were observed in this image
            observed_indices = []
            projected_observed = []
            for i, landmark_id in enumerate(optimized_points3d.get_ids()):
                if landmark_id in obs_ids:
                    observed_indices.append(i)
                    projected_observed.append(projected_points[i])

            if projected_observed:
                # Create ImageFeatures with the projected 2D points directly
                # since we already have the projected points
                optimized_features = ImageFeatures(
                    timestamp,
                    camera_id,
                    image_width,
                    image_height,
                    Features2D(np.array(projected_observed), np.array(obs_ids)),
                )
            else:
                # If no valid projections, keep original
                optimized_features = image_features

            optimized_image_features.append(optimized_features)

        return optimized_image_features, optimized_points3d, optimized_camera_poses
