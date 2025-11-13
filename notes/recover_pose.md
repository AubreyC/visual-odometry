# OpenCV `cv2.recoverPose()` Summary

## Purpose
`cv2.recoverPose()` recovers the **relative pose** (rotation and translation) between two calibrated camera views from an **essential matrix**.

---

## Inputs
- `E`: Essential matrix
- `points1`: matched points in the first image (Nx2)
- `points2`: matched points in the second image (Nx2)
- `K`: Camera intrinsic matrix  

---

## Outputs
- `R`: 3×3 rotation matrix (camera 2 relative to camera 1)
- `t`: 3×1 translation vector (up to scale)
- `mask`: inlier mask
- `retval`: Number of inliers used  

---

## Coordinate Frames

For a 3D point `X`:

$$
X_2 = R X_1 + t
$$

- `X_1` : point in **camera 1 coordinates**  
- `X_2` : point in **camera 2 coordinates**  
- **R** : rotates points from camera 1 → camera 2  
- **t** : translation vector **expressed in camera 2 frame** (direction only, unknown scale)  

Camera 2 center in camera 1 coordinates:
$$
C_2 = -R^\top t
$$

---

## Notes
- Translation `t` is **up to scale** (cannot recover real distances without extra information).  
- Chirality check ensures the physically correct solution (positive depth).  
- To get **camera trajectory** over multiple frames, accumulate relative `R` and `t` carefully, adjusting for frame coordinates.
