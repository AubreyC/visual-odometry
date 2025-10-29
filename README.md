# Visual Inertial Odometry and SLAM

This repository presents notes and material on the topic on visual intertial odometry.

**Most notable methodologies:**
- MSCKF
- ROVIO
- OKVIS
- VINS-Mono
- Visual-Inertial Monocular SLAM with Map Reuse uses “On-Manifold Preintegration for Real-Time Visual-Inertial Odometry"

**Implemented:**
- [] MSCKF
- [] Non-Linear bundle adjustment
- [] On-Manifold Preintegration for Real-Time Visual-Inertial Odometry

## Litterature review

### MSCKF 

[A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation Anastasios I. Mourikis and Stergios I. Roumeliotis](https://intra.ece.ucr.edu/~mourikis/papers/MourikisRoumeliotis-ICRA07.pdf)

**Methodology:**
- State: IMU + Position of the camera at image frames
- Process: IMU equations
- Measurements:
    - Constraints from a visual feature oberved in multiple frame
    - Least-Square estimation to estimate this feature location in inertial frame
    - Use this to reproject on image frame for each frame
    - Residual: reprojected position vs measurement position (with some tricks to eliminate the fact that the estimate state is used to estimate the feature position in inertial frame)

**Notes:**
- Eq: (13) Do the Math with the full state Pdot = F*P + P*F^T + G*Q*G^T but with F = [F1, 0; 0; 0] because the rest of the state are position and velocity of the previous image frames (constant in time)

**Limitations:**
- Need to wait until a feature is not tracked anymore to add it to the filter (measuremens mices all the observations in one measurement after it is done being observed)

**Related methods / Papers:**
- Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight

## On-Manifold Preintegration for Real-Time Visual-Inertial Odometry

[On-Manifold Preintegration for Real-Time Visual-Inertial Odometry](https://arxiv.org/pdf/1512.02363)

**Methodology:**
- IMU data and Images
- Select KeyFrame at which we want to estimate state [pos, vel, orientation]
- Inertial:
    - In between keyframe define IMU preintegration on Manifold
- Vision:
    - Structureless Vision Model
    - Does not uses the actual feature points
- Non-linear optimization of this

Questions:
- Feature point position in inertial frame not computed ???

## Visual-Inertial Monocular SLAM with Map Reuse

[Visual-Inertial Monocular SLAM with Map Reuse](https://arxiv.org/pdf/1610.05949v1)

**Methodology:**
- Background thread:
    - Compute Bundle Adjustment to find feature point position in intertial frame
- Tracking (real-time tracker):
    - Integrate the IMU data
    - Non linear optimization: the last frame position to minimize error from IMU & error from feature points (reprojected)
        - State (position, velocity, accel bias, gyro bias)
        - Cost:
            - Visual: Reprojection error of the feature point onto image plane
            - IMU: Error between state (variable to optimize) and "predicted" state from IMU integration

Related methods / Papers:
- `On-Manifold Preintegration for Real-Time Visual-Inertial Odometry`

## ROVIO

[M. Bloesch, S. Omari, M. Hutter, and R. Siegwart, “Robust visual inertial odometry using a direct ekf-based approach,” in Ieee/rsj International Conference on Intelligent Robots and Systems, 2015](https://github.com/MichaelBeechan/VO-SLAM-Review/blob/master/%5BVIO%5D%20Robust%20Visual%20Inertial%20Odometry%20Using%20a%20Direct%20EKF-Based%20Approach.pdf)

**Methodology:**
- EKF filter:
    - IMU state: position, orientation, velocity, accel bias, gyro bias
    - Visual features expressed in Vehicle state with bearing and distance (up to N to keep the size of the state manageable)
- Visual feature:
    - Selected based on gradient itensity (some heuristics to select then)
    - "Patch" region is selected around this feature
- Propagattion:
    - Propagate the state according to the IMU input
    - Propagate the visual feature 
- Update:
    - Residual is Observed patch intensities − Predicted patch intensities 
    - Predicted patch intensities: projected pixel location in the reference (or keyframe) patch, warped according to the predicted transform from the estimated pose and landmark depth.

Note: `Iterated EKF with direct photometric feedback, Bloesch et al. 2017` provides more details.

## OKVIS

[Stefan Leutenegger, Simon Lynen, Michael Bosse, Roland Siegwart and Paul Timothy Furgale. Keyframe-based visual–inertial odometry using nonlinear optimization. The International Journal of Robotics Research, 2015.](https://www.doc.ic.ac.uk/~sleutene/publications/ijrr2014_revision_1.pdf)

## Slambook


## Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age

Question:
- Since maximizing the posterior is the same as minimizing the negative log-posterior ?
- Note that the formulation (4) follows from the assumption of Normally distributed noise:
    - Other assumptions for the noise distribution lead to different cost functions;
    - For instance, if the noise follows a Laplace distribution, the squared l2 norm in (4) is replaced by the l1 norm

Graph optimization for pose estimation:
- Node: Pose of the camera
- Vertice: Measurement, external information (sensor, etc)

## VINS-Mono

This is a full system:
- Initialization
- Visualt Odometry based on sliding window method
- Global pose estimation based on graph optimization

Visual Inertial Odometry:
- bundle adjustment based on a sliding window method
- visual feature projected on a sphere tangent plane
- pre-integrated IMU data (?)

Global optimization:
- 

##

https://www.doc.ic.ac.uk/~sleutene/publications/ijrr2014_revision_1.pdf

## Visual-Inertial Monocular SLAM with Map Reuse

## Books

- STATE ESTIMATION FOR ROBOTICS - Timothy D. Barfoot (http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf)

## Reference

- A. I. Mourikis and S. I. Roumeliotis. A Multi-State Constraint Kalman
Filter for Vision-aided Inertial Navigation. In Proceedings of the IEEE
International Conference on Robotics and Automation (ICRA), pages
3565–3572. IEEE, 2007.

- S. Lynen, T. Sattler, M. Bosse, J. Hesch, M. Pollefeys, and R. Siegwart.
Get out of my lab: Large-scale, real-time visual-inertial localization.
In Proceedings of Robotics: Science and Systems Conference (RSS),
pages 338–347, 2015

Maximum A Posteriori:

- G. Grisetti, R. K¨ummerle, C. Stachniss, and W. Burgard. A Tutorial
on Graph-based SLAM. IEEE Intelligent Transportation Systems
Magazine, 2(4):31–43, 2010
- F. Dellaert. Factor Graphs and GTSAM: A Hands-on Introduction.
Technical Report GT-RIM-CP&R-2012-002, Georgia Institute of Tech-
nology, Sept. 2012


- M. Pizzoli, C. Forster, and D. Scaramuzza. REMODE: Probabilistic,
Monocular Dense Reconstruction in Real Time. In Proceedings of the
IEEE International Conference on Robotics and Automation (ICRA),
pages 2609–2616. IEEE, 2014.

- G. Dissanayake, S. Huang, Z. Wang, and R. Ranasinghe. A review
of recent developments in Simultaneous Localization and Mapping. In
International Conference on Industrial and Information Systems, pages
477–482. IEEE, 2011

- Z. Wang, S. Huang, and G. Dissanayake. Simultaneous Localization
and Mapping: Exactly Sparse Information Filters. World Scientific,
2011

- A. P´azman. Foundations of Optimum Experimental Design. Springer,
1986.

- Huang G (2019) Visual-inertial navigation: A concise review. In: IEEE Int. Conf. Robot. Autom.
(ICRA)

- Kottas DG, Hesch JA, Bowman SL, Roumeliotis SI (2012) On the consistency of vision-aided
inertial navigation. In: Int. Symp. Experimental Robotics 

- Kelly J, Sukhatme GS (2011) Visual-inertial sensor fusion: Localization, mapping and sensor-to-
sensor self-calibration. Int J Robot Research 30(1):56–79, DOI 10.1177/0278364910382802

- Kelly J, Sukhatme GS (2011) Visual-inertial sensor fusion: Localization, mapping and sensor-to-
sensor self-calibration. Int J Robot Research 30(1):56–79, DOI 10.1177/0278364910382802

- Furgale P, Rehder J, Siegwart R (2013) Unified temporal and spatial calibration for multi-sensor
systems. In: IEEE/RSJ Int. Conf. Intell. Robot. Syst. (IROS)

- F. C. Park, J. E. Bobrow, and S. R. Ploen, “A lie group formulation
of robot dynamics,” The International Journal of Robotics Research,
vol. 14, no. 6, pp. 609–618, 1995.

- F. Bullo and A. D. Lewis, Geometric control of mechanical systems:
modeling, analysis, and design for simple mechanical control systems,
vol. 49. Springer Science & Business Media, 2004

## Dataset

## Tools and Implementation

- https://gitlab.com/VladyslavUsenko/basalt
- https://github.com/daniilidis-group/msckf_mono
- https://github.com/KumarRobotics/msckf_vio?tab=readme-ov-file