import numpy as np

from .validation_error import ValidationError


class ValidationHelper:
    """Helper class for validating inputs."""

    @staticmethod
    def validate_ids(ids: np.ndarray) -> None:
        if not isinstance(ids, np.ndarray):
            raise TypeError("ids must be a NumPy array.")
        if ids.ndim != 1:
            raise ValueError(f"ids must be 1D, but got {ids.ndim}D.")

    @staticmethod
    def validate_pts3d(pts3d: np.ndarray) -> None:
        if not isinstance(pts3d, np.ndarray) or pts3d.ndim != 2 or pts3d.shape[1] != 3:
            raise ValidationError(f"pts3d must be Nx3 array, got shape {pts3d.shape}")

    @staticmethod
    def validate_pts2d(pts2d: np.ndarray) -> None:
        if not isinstance(pts2d, np.ndarray) or pts2d.ndim != 2 or pts2d.shape[1] != 2:
            raise ValidationError(f"pts2d must be Nx2 array, got shape {pts2d.shape}")

    @staticmethod
    def validate_pt3d(pt3d: np.ndarray) -> None:
        """Validate single 3D point

        Args:
            pt3d (np.ndarray): (3,) array of a 3D point coordinates

        Raises:
            ValidationError: If pt_3d are invalid.
        """

        if not isinstance(pt3d, np.ndarray):
            raise ValidationError(f"pt3d must be a numpy array, got {type(pt3d)}")

        if pt3d.shape != (3,):
            raise ValidationError(f"pt3d must be (3,) array, got shape {pt3d.shape}")

        if not np.all(np.isfinite(pt3d)):
            raise ValidationError("pt3d must contain only finite values")

    @staticmethod
    def validate_pt2d(pt2d: np.ndarray) -> None:
        """Validate single 2D point

        Args:
            pt2d (np.ndarray): (2,) array of a 2D point coordinates

        Raises:
            ValidationError: If pt2d are invalid.
        """

        if not isinstance(pt2d, np.ndarray):
            raise ValidationError(f"pt2d must be a numpy array, got {type(pt2d)}")

        if pt2d.shape != (2,):
            raise ValidationError(f"pt2d must be (2,) array, got shape {pt2d.shape}")

        if not np.all(np.isfinite(pt2d)):
            raise ValidationError("pt2d must contain only finite values")
