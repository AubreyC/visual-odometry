# The Schur Complement Trick in Visual Odometry

## 1. Context: Nonlinear Least Squares in Visual Odometry

In visual odometry (VO), we estimate:
- **Camera poses** \( \mathbf{T}_i \) (or states)
- **Landmark positions** \( \mathbf{p}_j \) (3D points in the world)

using observed **image measurements** (features, reprojections, etc.).

We solve a **nonlinear least squares** problem:

\[
\min_{\mathbf{x}} \; \| \mathbf{r}(\mathbf{x}) \|^2
\]

where \( \mathbf{x} = [\mathbf{x}_c, \mathbf{x}_p] \) includes:
- \( \mathbf{x}_c \): camera pose parameters  
- \( \mathbf{x}_p \): landmark (point) parameters  

and \( \mathbf{r}(\mathbf{x}) \) are the residuals (e.g., reprojection errors).

---

## 2. Linearization and the Normal Equations

At each Gauss-Newton (or Levenberg–Marquardt) iteration, we linearize:

\[
\mathbf{r}(\mathbf{x} + \delta \mathbf{x}) \approx \mathbf{r} + J \, \delta \mathbf{x}
\]

Then we solve the **normal equations**:

\[
(J^\top J) \, \delta \mathbf{x} = - J^\top \mathbf{r}
\]

---

## 3. Block Structure of the Jacobian

Observations relate poses and landmarks, so the Jacobian \( J \) has a **sparse block structure**:

\[
\delta \mathbf{x} = 
\begin{bmatrix}
\delta \mathbf{x}_c \\
\delta \mathbf{x}_p
\end{bmatrix}
, \quad
H = J^\top J =
\begin{bmatrix}
H_{cc} & H_{cp} \\
H_{pc} & H_{pp}
\end{bmatrix}
, \quad
\mathbf{b} = 
\begin{bmatrix}
b_c \\
b_p
\end{bmatrix}
\]

The linear system is:

\[
\begin{bmatrix}
H_{cc} & H_{cp} \\
H_{pc} & H_{pp}
\end{bmatrix}
\begin{bmatrix}
\delta \mathbf{x}_c \\
\delta \mathbf{x}_p
\end{bmatrix}
=
\begin{bmatrix}
b_c \\
b_p
\end{bmatrix}
\]

---

## 4. The Schur Complement Trick

Directly solving the full system is expensive because there can be **thousands of landmarks**.  
However, \( H_{pp} \) (landmark-related) is **block diagonal**, making it easy to invert.

From the second block row:

\[
H_{pc} \, \delta \mathbf{x}_c + H_{pp} \, \delta \mathbf{x}_p = b_p
\]

Solve for landmarks:

\[
\delta \mathbf{x}_p = H_{pp}^{-1}(b_p - H_{pc} \, \delta \mathbf{x}_c)
\]

Substitute into the first row to get a **reduced camera-only system**:

\[
(H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}) \, \delta \mathbf{x}_c = b_c - H_{cp} H_{pp}^{-1} b_p
\]

This is the **Schur complement system**:

\[
H_{sc} \, \delta \mathbf{x}_c = b_{sc}
\]

where:

\[
H_{sc} = H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}, \quad
b_{sc} = b_c - H_{cp} H_{pp}^{-1} b_p
\]

Solve this for camera updates, then **back-substitute** for landmarks.

---

## 5. Solving Efficiently

1. Compute \( H_{pp}^{-1} \) (cheap, block-diagonal).  
2. Form the reduced system \( H_{sc} \, \delta \mathbf{x}_c = b_{sc} \).  
3. Solve for \( \delta \mathbf{x}_c \).  
4. Back-substitute for landmarks:

\[
\delta \mathbf{x}_p = H_{pp}^{-1}(b_p - H_{pc} \, \delta \mathbf{x}_c)
\]

---

## 6. Why It Matters in Visual Odometry

- **Faster optimization**: Only camera poses are solved in the reduced system.  
- **Numerically stable**: Reduces ill-conditioning from many points.  
- **Memory efficient**: Exploits sparsity of \( H_{pp} \).  
- Widely used in:
  - Bundle Adjustment (BA)
  - Keyframe-based VO (e.g., ORB-SLAM, VINS-Mono)

---

## 7. Back-Computing Landmark Positions

Once \( \delta \mathbf{x}_c \) is known, recover landmark updates:

\[
\delta \mathbf{x}_p = H_{pp}^{-1}(b_p - H_{pc} \, \delta \mathbf{x}_c)
\]

- Each landmark is independent (block-diagonal).  
- Update landmarks:

\[
\mathbf{x}_{p_j}^{\text{new}} = \mathbf{x}_{p_j}^{\text{old}} + \delta \mathbf{x}_{p_j}
\]

Optionally, landmarks can be **re-triangulated geometrically** after camera pose updates for better numerical stability.

---

## 8. Full Optimization Cycle

| Step | Description | Variables |
|------|-------------|-----------|
| 1 | Linearize residuals | \( J, H, b \) |
| 2 | Apply Schur complement | Eliminate \( \delta \mathbf{x}_p \) |
| 3 | Solve reduced system | \( \delta \mathbf{x}_c \) |
| 4 | Back-substitute | \( \delta \mathbf{x}_p \) |
| 5 | Update parameters | \( \mathbf{x}_c, \mathbf{x}_p \) |
| 6 | Iterate | Until convergence |

---

## 9. Summary

- The **Schur complement** allows **efficient optimization** by eliminating landmarks.  
- Landmarks can be **recovered after camera optimization** via back-substitution or geometric re-triangulation.  
- Central to modern **bundle adjustment**, **keyframe VO**, and **SLAM**.

