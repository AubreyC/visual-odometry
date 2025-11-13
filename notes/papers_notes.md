# Papers notes

## References
- Mourikis & Roumeliotis, *MSCKF (2007)*  
- Forster et al., *On-Manifold Preintegration (2015)*  
- Mur-Artal & Tardós, *Visual-Inertial Monocular SLAM (2017)*  
- Bloesch et al., *ROVIO (2015)*

## Comparative Summary

| **Method** | **Type** | **Optimization / Filter** | **State Variables** | **Visual Model** | **IMU Handling** | **Key Features / Notes** |
|-------------|-----------|---------------------------|---------------------|------------------|------------------|---------------------------|
| **MSCKF** | VIO | EKF-based Filter | IMU + recent camera poses | Feature-based (tracked features) | Propagation with IMU equations | Efficient but delayed feature inclusion |
| **On-Manifold Preintegration** | VIO | Non-linear Optimization | Keyframe pose, velocity, orientation | Structureless model | IMU preintegration on manifold | Clean formulation, widely used |
| **VIO SLAM with Map Reuse** | SLAM | Non-linear Optimization + Bundle Adjustment | Pose, velocity, biases | Reprojection of tracked features | IMU integrated and optimized | Real-time tracking + background BA |
| **ROVIO** | VIO | EKF-based Filter | Full IMU + limited visual landmarks | Patch-intensity based | Continuous IMU propagation | Robust to lighting, tightly coupled |

## MSCKF

### Methodology
- **Filter-based method**
- **State:** IMU + position of the camera at some image frames (a fixed number in the past)
  - **Process:** IMU equations  
  - **Measurements:**
    - Constraints from a visual feature observed in multiple frames  
    - Least-squares estimation to estimate this feature’s location in the inertial frame  
    - Use this to reproject onto the image frame for each frame  
    - **Residual:** Reprojected position vs. measured position  
      *(with some tricks to eliminate the fact that the estimated state is used to estimate the feature position in the inertial frame)*

### Notes
- Equation (13): Perform the math with the full state

  $$
  \dot{P} = F P + P F^T + G Q G^T
  $$  

  but with  

  $$
  F = 
  \begin{bmatrix}
  F_1 & 0 \\
  0 & 0
  \end{bmatrix}
  $$
  
  because the rest of the state represents the position and velocity of previous image frames (constant in time).

### Limitations
- Must wait until a feature is **no longer tracked** before adding it to the filter  
  (measurements mix all the observations into one after tracking ends)

## On-Manifold Preintegration for Real-Time Visual-Inertial Odometry

### Methodology
- Select **keyframes** where the state (position, velocity, orientation) will be estimated  
- **Non-linear optimization**
  - **Inertial constraint:** Define IMU preintegration on the manifold between keyframes  
  - **Vision:**
    - Structureless vision model  
    - Does **not** use the actual feature points

## Visual-Inertial Monocular SLAM with Map Reuse

> Based on *On-Manifold Preintegration for Real-Time Visual-Inertial Odometry*

### Methodology
- **Background thread:**  
  Compute bundle adjustment to find feature point positions in the inertial frame
- **Tracking (real-time tracker):**
  - Integrate the IMU data  
  - Perform non-linear optimization of the last frame position to minimize:
    - **IMU error**
    - **Feature point reprojection error**
  - **State:** Position, velocity, accel bias, gyro bias  
  - **Cost terms:**
    - **Visual:** Reprojection error of the feature point onto the image plane  
    - **IMU:** Error between the optimized state and the “predicted” state from IMU integration

## ROVIO — Robust Visual Inertial Odometry

### Methodology
- **EKF-based filter**
  - **State:** IMU (position, orientation, velocity, accel bias, gyro bias)
  - **Visual features:** Expressed in vehicle state with bearing and distance  
    (up to *N* to keep the state size manageable)
- **Visual feature selection:**
  - Based on gradient intensity (with heuristics)
  - A **patch region** is selected around each feature
- **Propagation:**
  - Propagate the state according to IMU input  
  - Propagate the visual features
- **Update:**
  - **Residual:** Observed patch intensities − Predicted patch intensities  
  - **Predicted patch:** Projected pixel location in the reference (or keyframe) patch,  
    warped according to the predicted transform from the estimated pose and landmark depth

