"""Feature detection and tracking algorithms for visual odometry."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .validation_error import ProcessingError, ValidationError

MIN_FEATURES = 70  # minimum features before re-detection
MAX_FEATURES = 150  # max ORB features per detection
MIN_DISTANCE = 10.0  # minimum distance to existing points


class FeatureTracker:
    """Feature detection and tracking algorithms for visual odometry."""

    def __init__(self) -> None:
        """Initialize the feature tracker."""

        self.initialized = False
        self.current_features_pts = np.empty((0, 2), dtype=np.float32)
        self.current_features_ids = np.empty((0,), dtype=int)

        self.prev_img = np.empty((0, 0), dtype=np.uint8)
        self.feature_ids_counter = 0

    def run_tracking(
        self,
        current_image: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        # Initialize the feature tracker
        if not self.initialized:
            self.current_features_pts, _, _ = self.detect_features(current_image)
            self.current_features_ids = np.arange(len(self.current_features_pts))
            self.feature_ids_counter += len(self.current_features_ids)
            self.prev_img = current_image.copy()
            self.initialized = True
            return self.current_features_pts, self.current_features_ids

        # Track existing features
        self.current_features_pts, status, error = self.track_features_klt(
            self.prev_img, current_image, self.current_features_pts
        )
        # Update the features ids and points
        self.current_features_ids = self.current_features_ids[status.ravel() == 1]
        self.current_features_pts = self.current_features_pts[status.ravel() == 1]

        # Remove features that are too close to each other
        _, feature_mask = self.filter_features_by_distance_with_mask(
            self.current_features_pts, MIN_DISTANCE
        )
        self.current_features_pts = self.current_features_pts[feature_mask]
        self.current_features_ids = self.current_features_ids[feature_mask]

        # If not enough features, detect new ones and merge
        if len(self.current_features_pts) < MIN_FEATURES:
            print(
                f"Not enough features, detecting new ones: {len(self.current_features_pts)}"
            )
            new_pts = self.add_new_features(current_image, self.current_features_pts)
            if len(new_pts) > 0:
                self.current_features_pts = np.vstack(
                    (self.current_features_pts, new_pts)
                )
                new_ids = np.arange(
                    self.feature_ids_counter, self.feature_ids_counter + len(new_pts)
                )
                self.feature_ids_counter += len(new_pts)
                self.current_features_ids = np.hstack(
                    (self.current_features_ids, new_ids)
                )

        self.prev_img = current_image.copy()
        return self.current_features_pts, self.current_features_ids

    def detect_features(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect features in the image.

        Returns:
            Tuple of (keypoints, descriptors, selection_mask) where:
            - keypoints: Selected feature coordinates, shape (N, 2)
            - descriptors: ORB descriptors for selected features, shape (N, 32)
            - selection_mask: Boolean mask indicating which detected features were selected
        """
        keypoints, descriptors = self.detect_orb_features(
            image,
            max_features=3 * MAX_FEATURES,
        )

        optimal_keypoints, selection_mask = self.select_optimal_features(
            keypoints,
            image.shape[1],
            image.shape[0],
        )
        return optimal_keypoints, descriptors, selection_mask

    def validate_image_input(self, image: np.ndarray) -> None:
        """Validate the image.

        Args:
            image: Image to validate.

        Raises:
            ValidationError: If image is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError("Image must be a numpy array")

        if image.ndim != 2:
            raise ValidationError(
                f"Image must be grayscale (2D), got shape {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValidationError(f"Image must be uint8, got dtype {image.dtype}")

    def detect_orb_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        max_features: int = MAX_FEATURES,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect ORB features in an image.

        Args:
            image: Grayscale input image.
            mask: Optional mask to restrict detection to certain regions.

        Returns:
            Tuple of (keypoints, descriptors) where keypoints is Nx2 array of [x,y] coordinates
            and descriptors is Nx32 array of ORB descriptors.

        Raises:
            ValidationError: If image is invalid.
            ProcessingError: If feature detection fails.
        """
        # Validate the image
        self.validate_image_input(image)

        try:
            orb_detector = cv2.ORB_create(nfeatures=max_features)
            keypoints, descriptors = orb_detector.detectAndCompute(image, mask)
        except Exception as e:
            raise ProcessingError(f"ORB feature detection failed: {e}") from e

        if keypoints is None or len(keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 32), dtype=np.uint8)

        # Convert keypoints to numpy array
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)

        return points, descriptors

    def track_features_klt(
        self,
        prev_image: np.ndarray,
        curr_image: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track features using Lucas-Kanade optical flow (KLT tracker).

        Args:
            prev_image: Previous grayscale image.
            curr_image: Current grayscale image.
            prev_points: Previous feature points, shape (N, 2).

        Returns:
            Tuple of (curr_points, status, err) where:
            - curr_points: Tracked points in current image, shape (N, 2)
            - status: Boolean array indicating successful tracking, shape (N,)
            - err: Tracking error for each point, shape (N,)

        Raises:
            ValidationError: If inputs are invalid.
            ProcessingError: If optical flow fails.
        """
        # Validate the images
        self.validate_image_input(prev_image)
        self.validate_image_input(curr_image)

        # Validate the previous points
        self.validate_features_input(prev_points)

        if len(prev_points) == 0:
            # No points to track
            return (
                np.empty((0, 2), dtype=np.float32),
                np.array([], dtype=bool),
                np.array([], dtype=np.float32),
            )

        try:
            curr_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_image, curr_image, prev_points, None
            )
        except Exception as e:
            raise ProcessingError(f"KLT optical flow tracking failed: {e}") from e

        if curr_points is None:
            curr_points = np.empty((0, 2), dtype=np.float32)
            status = np.array([], dtype=bool)
            err = np.array([], dtype=np.float32)

        return curr_points, np.asarray(status, dtype=bool), err

    def add_new_features(
        self,
        curr_img: np.ndarray,
        existing_pts: np.ndarray,
        min_distance: float = MIN_DISTANCE,
    ) -> np.ndarray:
        """Detect new features and add only those far enough from existing ones."""
        new_pts, _, _ = self.detect_features(curr_img)
        if len(existing_pts) == 0:
            return new_pts

        filtered_new_pts, _ = self.filter_new_features_by_distance_with_mask(
            new_pts, existing_pts, min_distance
        )

        return filtered_new_pts

    def validate_features_input(self, features: np.ndarray) -> None:
        """Validate the features.

        Args:
            features: Nx2 array of feature coordinates.

        Raises:
            ValidationError: If inputs are invalid.
        """
        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or features.shape[1] != 2
        ):
            raise ValidationError(
                f"features must be Nx2 array, got shape {features.shape}"
            )

    def filter_features_by_region_with_mask(
        self,
        features: np.ndarray,
        image_width: int,
        image_height: int,
        margin: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter features to remove those too close to image boundaries.

        Args:
            features: Nx2 array of feature coordinates.
            image_width: Width of the image.
            image_height: Height of the image.
            margin: Minimum distance from image boundaries.

        Returns:
            Tuple of (filtered_features, mask) where mask indicates which input features were kept.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate the features
        self.validate_features_input(features)

        if len(features) == 0:
            return features, np.array([], dtype=bool)

        # Filter by image boundaries
        valid_mask = (
            (features[:, 0] >= margin)
            & (features[:, 0] < image_width - margin)
            & (features[:, 1] >= margin)
            & (features[:, 1] < image_height - margin)
        )

        return features[valid_mask], valid_mask

    def filter_features_by_distance_with_mask(
        self,
        features: np.ndarray,
        min_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter features to ensure minimum distance between them.

        Args:
            features: Nx2 array of feature coordinates.
            min_distance: Minimum distance between features.

        Returns:
            Tuple of (filtered_features, mask) where mask indicates which input features were kept.

        Raises:
            ValidationError: If inputs are invalid.
        """

        # Validate the features
        self.validate_features_input(features)

        if len(features) == 0 or min_distance <= 0:
            return features, np.ones(len(features), dtype=bool)

        # Start with all features
        selected_mask = np.zeros(len(features), dtype=bool)

        for i, feature in enumerate(features):
            # Check distance to all previously selected features
            too_close = False
            for selected in features[selected_mask]:
                distance = np.linalg.norm(feature - selected)
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected_mask[i] = True

        return features[selected_mask], selected_mask

    def filter_new_features_by_distance_with_mask(
        self,
        new_features: np.ndarray,
        existing_features: np.ndarray,
        min_distance: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter features to ensure minimum distance between them.

        Args:
            features: Nx2 array of feature coordinates.
            min_distance: Minimum distance between features.

        Returns:
            Tuple of (filtered_features, mask) where mask indicates which input features were kept.

        Raises:
            ValidationError: If inputs are invalid.
        """

        # Validate the features
        self.validate_features_input(new_features)
        self.validate_features_input(existing_features)

        if len(new_features) == 0 or min_distance <= 0:
            return new_features, np.ones(len(new_features), dtype=bool)

        # Start with all features
        selected_mask = np.zeros(len(new_features), dtype=bool)

        for i, feature in enumerate(new_features):
            # Check distance to all previously selected features
            too_close = False
            for existing in existing_features:
                distance = np.linalg.norm(feature - existing)
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected_mask[i] = True

        return new_features[selected_mask], selected_mask

    def filter_features_by_distance_optimized_with_mask(
        self,
        features: np.ndarray,
        min_distance: float,
        max_features: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter features to ensure minimum distance using optimized selection.

        This method tries to select features that are well-distributed across the image
        by preferring features that are farthest from already selected ones.

        Args:
            features: Nx2 array of feature coordinates.
            min_distance: Minimum distance between features.
            max_features: Maximum number of features to return (None for no limit).

        Returns:
            Tuple of (filtered_features, mask) where mask indicates which input features were kept.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate features
        self.validate_features_input(features)

        if len(features) == 0 or min_distance <= 0:
            mask = np.ones(len(features), dtype=bool)
            if max_features is not None:
                mask[max_features:] = False
                features = features[:max_features]
            return features, mask

        if max_features is None:
            max_features = len(features)

        # Start with the feature closest to image center (assuming standard image coordinates)
        # This helps ensure central features are prioritized
        image_center = np.array([320.0, 240.0])  # Approximate center for 640x480 image
        distances_to_center = np.linalg.norm(features - image_center, axis=1)
        start_idx = np.argmin(distances_to_center)

        selected_indices = [int(start_idx)]
        remaining_indices = list(range(len(features)))
        remaining_indices.remove(int(start_idx))

        while len(remaining_indices) > 0 and len(selected_indices) < max_features:
            # Find the feature that is farthest from all already selected features
            max_min_distance = 0.0
            best_idx = -1

            for candidate_idx in remaining_indices:
                # Find minimum distance to any selected feature
                min_dist_to_selected = min(
                    np.linalg.norm(features[candidate_idx] - features[selected_idx])
                    for selected_idx in selected_indices
                )

                if min_dist_to_selected > max_min_distance:
                    max_min_distance = min_dist_to_selected  # type: ignore
                    best_idx = candidate_idx

            # If the best candidate is still too close to existing features, stop
            if max_min_distance < min_distance:
                break

            # Add the best candidate
            selected_indices.append(int(best_idx))
            remaining_indices.remove(int(best_idx))

        # Create mask
        selected_mask = np.zeros(len(features), dtype=bool)
        selected_mask[selected_indices] = True

        return features[selected_indices], selected_mask

    def distribute_features_evenly_with_mask(
        self,
        features: np.ndarray,
        image_width: int,
        image_height: int,
        grid_rows: int = 4,
        grid_cols: int = 6,
        max_per_cell: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute features evenly across the image using a grid-based approach.

        Args:
            features: Nx2 array of feature coordinates.
            image_width: Width of the image.
            image_height: Height of the image.
            grid_rows: Number of rows in the spatial grid.
            grid_cols: Number of columns in the spatial grid.
            max_per_cell: Maximum features per grid cell.

        Returns:
            Tuple of (selected_features, mask) where mask indicates which input features were kept.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate features
        self.validate_features_input(features)

        if len(features) == 0:
            return features, np.array([], dtype=bool)

        # Calculate grid cell dimensions
        cell_width = image_width / grid_cols
        cell_height = image_height / grid_rows

        # Group features by grid cell
        cell_features: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
        for i, feature in enumerate(features):
            x, y = feature
            cell_x = min(int(x / cell_width), grid_cols - 1)
            cell_y = min(int(y / cell_height), grid_rows - 1)
            cell_key = (cell_x, cell_y)

            if cell_key not in cell_features:
                cell_features[cell_key] = []
            cell_features[cell_key].append((i, feature))  # Store index and feature

        # Select features from each cell (prioritize by distance from cell center)
        selected_indices = []

        for cell_key, cell_feature_list in cell_features.items():
            if len(cell_feature_list) == 0:
                continue

            cell_x, cell_y = cell_key
            cell_center_x = (cell_x + 0.5) * cell_width
            cell_center_y = (cell_y + 0.5) * cell_height

            # Sort features in this cell by distance from cell center
            distances_from_center = []
            for idx, feature in cell_feature_list:
                distance = np.linalg.norm(
                    feature - np.array([cell_center_x, cell_center_y])
                )
                distances_from_center.append((distance, idx))

            # Sort by distance (closest to center first)
            distances_from_center.sort()

            # Take up to max_per_cell features from this cell
            num_to_take = min(max_per_cell, len(distances_from_center))
            for i in range(num_to_take):
                selected_indices.append(distances_from_center[i][1])

        # Create mask
        selected_mask = np.zeros(len(features), dtype=bool)
        selected_mask[selected_indices] = True

        return features[selected_mask], selected_mask

    def select_optimal_features(
        self,
        features: np.ndarray,
        image_width: int,
        image_height: int,
        max_features: int = 100,
        min_distance: float = 15.0,
        use_grid_distribution: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select optimal features with good spatial distribution and minimum distance constraints.

        This method combines multiple filtering strategies to select high-quality features:
        1. Boundary filtering to avoid edge features
        2. Grid-based distribution for even spatial coverage
        3. Distance-based filtering to prevent clustering
        4. Quality-based selection

        Args:
            features: Nx2 array of feature coordinates.
            image_width: Width of the image.
            image_height: Height of the image.
            max_features: Maximum number of features to return.
            min_distance: Minimum distance between features.
            use_grid_distribution: Whether to use grid-based distribution.

        Returns:
            Tuple of (selected_features, selection_mask) where:
            - selected_features: Optimized Nx2 array of feature coordinates.
            - selection_mask: Boolean array of same length as input features, indicating which were selected.

        Raises:
            ValidationError: If inputs are invalid.
        """
        # Validate features
        self.validate_features_input(features)

        if len(features) == 0:
            selection_mask = np.array([], dtype=bool)
            return features, selection_mask

        # Initialize selection mask (all features start as selected)
        selection_mask = np.ones(len(features), dtype=bool)

        # Step 1: Filter by boundaries (keep features away from edges)
        boundary_filtered, boundary_mask = self.filter_features_by_region_with_mask(
            features, image_width, image_height, margin=20
        )
        selection_mask &= boundary_mask

        if len(boundary_filtered) == 0:
            return boundary_filtered, selection_mask

        # Step 2: If requested, distribute evenly across image using grid
        if use_grid_distribution and len(boundary_filtered) > max_features:
            grid_filtered, grid_mask = self.distribute_features_evenly_with_mask(
                boundary_filtered,
                image_width,
                image_height,
                grid_rows=4,
                grid_cols=6,
                max_per_cell=4,
            )
            # Update the global mask - only features that passed boundary filtering can be in grid_mask
            temp_mask = np.zeros(len(features), dtype=bool)
            temp_mask[boundary_mask] = grid_mask
            selection_mask &= temp_mask
        else:
            grid_filtered = boundary_filtered
            # grid_mask would be all True for the boundary_filtered features
            # but we don't need to update selection_mask since boundary_mask already covers this

        # Step 3: Apply distance filtering with optimization
        if len(grid_filtered) > max_features:
            distance_filtered, distance_mask = (
                self.filter_features_by_distance_optimized_with_mask(
                    grid_filtered, min_distance, max_features
                )
            )
        else:
            # If we have fewer features than max, just apply basic distance filtering
            distance_filtered, distance_mask = (
                self.filter_features_by_distance_with_mask(grid_filtered, min_distance)
            )

        # Update the global mask
        if use_grid_distribution and len(boundary_filtered) > max_features:
            # distance_mask corresponds to grid_filtered indices
            temp_mask = np.zeros(len(features), dtype=bool)
            grid_indices = np.where(boundary_mask)[0][grid_mask]
            temp_mask[grid_indices] = distance_mask
            selection_mask &= temp_mask
        else:
            # distance_mask corresponds to boundary_filtered indices
            temp_mask = np.zeros(len(features), dtype=bool)
            boundary_indices = np.where(boundary_mask)[0]
            temp_mask[boundary_indices] = distance_mask
            selection_mask &= temp_mask

        # Step 4: Final limit to max_features
        if len(distance_filtered) > max_features:
            # If still too many, take the best distributed ones
            final_mask = np.ones(len(distance_filtered), dtype=bool)
            final_mask[max_features:] = False
            distance_filtered = distance_filtered[:max_features]

            # Update the global mask
            temp_mask = np.zeros(len(features), dtype=bool)
            selected_indices = np.where(selection_mask)[0]
            temp_mask[selected_indices] = final_mask
            selection_mask &= temp_mask

        return distance_filtered, selection_mask
