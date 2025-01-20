# MiniCheetah State Estimator (Offline Kalman Filter)

This repository contains Python code for implementing an offline Kalman filter-based state estimator for the Mini Cheetah robot. The filter processes logged data to estimate the robot's center of mass (COM) position and velocity, as well as individual leg positions, velocities, and heights. It is designed for offline analysis and debugging of state estimation algorithms. This code is based on the [Cheetah Software](https://github.com/mit-biomimetics/Cheetah-Software) published by the MIT Biomimetic Robotics Lab.

## Mathematical Overview

The Linear Kalman filter operates in two main steps:

### 1. Predict Step
```math
 \hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B a_k
```
```math
P_{k|k-1} = A P_{k-1|k-1} A^T + Q
```

### 2. Update Step


- Observation:
```math
yModel_k = C \hat{x}_{k|k-1}
```
- Kalman Gain:
```math
K_k = P_{k|k-1} C^T (C P_{k|k-1} C^T + R)^{-1}
```
 
- State Update:
```math
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (y_k - yModel_k)
```
- Covariance Update:
```math
P_{k|k} = (I - K_k C) P_{k|k-1}
```

---

### Details of Variables

#### **State Vector ($\hat{x}$)**
The state vector contains the following components:
1. **COM Position (3 values)**: $[x, y, z]$
2. **COM Velocity (3 values)**: $[\dot{x}, \dot{y}, \dot{z}]$
3. **Leg Positions (4 legs × 3 values each)**:
```math
   \text{Leg} 0: [x_0, y_0, z_0], \text{ Leg } 1: [x_1, y_1, z_1], \ldots, \text{ Leg } 3: [x_3, y_3, z_3]
```
Total size of $\hat{x}$: $18 \times 1$


---

#### **State Transition Matrix (\(A\))**
The matrix $(A)$ represents how the state evolves over time, incorporating the effect of time step $(dt)$:
1. The top-left $6 \times 6$ block handles the COM position and velocity dynamics.
2. The bottom-right $12 \times 12$  block is an identity matrix for the leg positions.

$$
A =
\begin{bmatrix}
I_{3} & dt \cdot I_{3} & 0 \\
0 & I_{3} & 0 \\
0 & 0 & I_{12}
\end{bmatrix}
$$

#### **Control Input ($a_k$)**
The control input vector $a_k$ represents accelerations:
1. **Control Input Matrix ($B$)**:

$$
B =
\begin{bmatrix}
0 \\
dt \cdot I_{3} \\
0
\end{bmatrix}
$$

3. **Input Vector ($a_k$)**:
   Acceleration input: $[a_x, a_y, a_z]$
---

#### **Measurement Vector ($y_k$)**
The measurement vector contains:
1. **Leg Positions (4 legs × 3 values each)**:
   
$$
\text{Leg } 0: [x_0, y_0, z_0], \text{ Leg } 1: [x_1, y_1, z_1], \ldots
$$

3. **Leg Velocities (4 legs × 3 values each)**:

$$
\text{Leg } 0: [\dot{x}_0, \dot{y}_0, \dot{z}_0], \ldots
$$

4. **Leg Heights (4 legs × 1 value each)**:

$$
\text{Leg } 0: z_0, \ldots
$$

Total size of $y_k$: $28 \times 1$


---

#### **Observation Matrix ($C$)**
The matrix $(C)$ maps the state vector to the measurement space. It includes:
- Direct mapping of leg positions from $\hat{x}$ to $y_k$.
- Negative identity blocks for relative positions and velocities.

```math
\mathbf{C} =
\begin{bmatrix}
\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & -\mathbf{I}_{12} \\
\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
\mathbf{0}_3 & \mathbf{I}_3 & \mathbf{0}_3 & \mathbf{0}_3 & 0 \\
0 & 0 & 0 & 0 & 0 & \dots & 1 & \dots & 0 \\
0 & 0 & 0 & 0 & 0 & \dots & 0 & 1 & \dots & 0 \\
0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & \dots & 0 & 0 & 0 & 1
\end{bmatrix}
```

---

#### **Noise Matrices**
1. **Process Noise Covariance ($Q$)**:
   Represents uncertainty in the model:
   - If you want to rely more on the model, which uses states calculated from IMU data, you should decrease the values in $Q$. A better IMU means more confidence in the model, so you can lower the noise values.
   
2. **Measurement Noise Covariance ($R$)**:
   Represents sensor noise:
   - If you want to rely more on the inverse kinematics (measurements), you should decrease the values in $R$. This means putting more trust in the sensor data (leg positions, foot tip positions) rather than the model.


---

### Workflow

1. **Log Parsing**: The `parse_log_file` function is responsible for extracting data from the logged file which are computed using inverse kinematics. The data contains real-world measurements recorded from the robot during operations. These measurements will be used as inputs for the Kalman filter to estimate the robot's state:

   * **Leg Positions**: These are collected from the robot’s legs, representing their positions in space.

   * **Z Foot Tip**: The foot tip height of the legs. 

   These two (leg positions and foot tip positions) will serve as the **measurements $(y_k)$** in the Kalman filter’s observation update step, where they are compared with the predicted values from the model.

2. **Kalman Filtering**: The `LinearKalmanFilter` function applies the Kalman filter algorithm:

   * **Prediction**: The state vector is predicted based on the previous state and accelaration inputs, using the system's state transition matrix $A$.
   
   * **Observation**: The observed measurements $y_k$, which include leg positions and foot tip positions, are compared with the predicted state values $yModel$ calculated from the accelerometer data of the IMU.
   
   * **Error Calculation**: The difference between the observed measurements $y_k$ and the predicted values $yModel$ generates an error vector $e_y$.

3. **State Estimation**:

   * The error vector $e_y$ is multiplied by the **Kalman Gain** ($K_k$) to compute how much influence the measurement error will have on correcting the state estimate.
   * This correction is then added to the previous state estimate $\hat{x}_{k|k-1}$ to update the state estimate to a new value $\hat{x}_{k|k}$.

---

### Logged Data
A logged file (in `.txt` format) from real robot tests in the **Recovery Stand** and **Squat Down** modes of the Mini Cheetah is provided. You can use this data to run offline tests, debug the Kalman filter, and analyze the accuracy and performance of the state estimator.


- **P.S**: Additionally, there is another Python script that implements the mathematical model of KF symbolically. 




