from sympy import symbols, Matrix, eye, zeros, pprint, Transpose

dt = 0.002

p_x, p_y, p_z = symbols('p_x p_y p_z')  # IMU position
v_x, v_y, v_z = symbols('v_x v_y v_z')  # IMU velocity

l1_x, l1_y, l1_z = symbols('l1_x l1_y l1_z')  # Leg 1 position
l2_x, l2_y, l2_z = symbols('l2_x l2_y l2_z')  # Leg 2 position
l3_x, l3_y, l3_z = symbols('l3_x l3_y l3_z')  # Leg 3 position
l4_x, l4_y, l4_z = symbols('l4_x l4_y l4_z')  # Leg 4 position


_xhat = Matrix([
    p_x, p_y, p_z,  # IMU position
    v_x, v_y, v_z,  # IMU velocity
    l1_x, l1_y, l1_z,  # Leg 1 position
    l2_x, l2_y, l2_z,  # Leg 2 position
    l3_x, l3_y, l3_z,  # Leg 3 position
    l4_x, l4_y, l4_z   # Leg 4 position
])


# Initialize symbolic matrices with corrected dimensions
_A = Matrix.zeros(18, 18)  # System matrix
_B = Matrix.zeros(18, 3)   # Control matrix
_C = Matrix.zeros(28, 18)  # Observation matrix
_P = eye(18) * 100         # Covariance matrix
_Q0 = Matrix.zeros(18, 18) # Process noise covariance
_R0 = eye(28)              # Measurement noise covariance


# Set _A matrix
_A[:3, :3] = eye(3)  
_A[:3, 3:6] = dt * eye(3)
_A[3:6, 3:6] = eye(3)
_A[6:, 6:] = eye(12)

# Set _B matrix
_B[3:6, :3] = dt * eye(3)

# Define submatrices C1 and C2
C1 = Matrix.hstack(eye(3), Matrix.zeros(3, 3))
C2 = Matrix.hstack(Matrix.zeros(3, 3), eye(3))

# Set _C matrix
_C[:3, :6] = C1
_C[3:6, :6] = C1
_C[6:9, :6] = C1
_C[9:12, :6] = C1
_C[:12, 6:18] = -eye(12)
_C[12:15, :6] = C2
_C[15:18, :6] = C2
_C[18:21, :6] = C2
_C[21:24, :6] = C2
_C[24, 8] = 1
_C[25, 11] = 1
_C[26, 14] = 1
_C[27, 17] = 1

# Set _Q0 matrix
_Q0[:3, :3] = (dt / 20.0) * eye(3)
_Q0[3:6, 3:6] = (dt * 9.8 / 20.0) * eye(3)
_Q0[6:, 6:] = dt * eye(12)

# Define symbolic variables
process_noise_pimu, process_noise_vimu, process_noise_pfoot = symbols('process_noise_pimu process_noise_vimu process_noise_pfoot')
sensor_noise_pimu_rel_foot, sensor_noise_vimu_rel_foot, sensor_noise_zfoot = symbols('sensor_noise_pimu_rel_foot sensor_noise_vimu_rel_foot sensor_noise_zfoot')


Q0 = eye(18) 
R0 = eye(28)  

# Define Q0 with symbolic dt
Q0[0:3, 0:3] = (dt / 20) * eye(3)
Q0[3:6, 3:6] = (dt * 9.8 / 20) * eye(3)
Q0[6:18, 6:18] = dt * eye(12)


Q = eye(18)  # Process noise covariance
R = eye(28)  # Measurement noise covariance

# Update Q matrix symbolically
# Q[0:3, 0:3] = Q0[0:3, 0:3] * process_noise_pimu
# Q[3:6, 3:6] = Q0[3:6, 3:6] * process_noise_vimu
# Q[6:18, 6:18] = Q0[6:18, 6:18] * process_noise_pfoot

# # Update R matrix symbolically
# R[0:12, 0:12] = R0[0:12, 0:12] * sensor_noise_pimu_rel_foot
# R[12:24, 12:24] = R0[12:24, 12:24] * sensor_noise_vimu_rel_foot
# R[24:28, 24:28] = R0[24:28, 24:28] * sensor_noise_zfoot

# Making the Code ligther 
# Update Q matrix symbolically
Q[0:3, 0:3] = Q0[0:3, 0:3] 
Q[3:6, 3:6] = Q0[3:6, 3:6] 
Q[6:18, 6:18] = Q0[6:18, 6:18]

# Update R matrix symbolically
R[0:12, 0:12] = R0[0:12, 0:12] 
R[12:24, 12:24] = R0[12:24, 12:24] 
R[24:28, 24:28] = R0[24:28, 24:28] 

# Define symbolic variables for position measurements (ps) for each leg (4 legs, xyz each)
ps = Matrix([
    symbols('ps1_x ps1_y ps1_z'),
    symbols('ps2_x ps2_y ps2_z'),
    symbols('ps3_x ps3_y ps3_z'),
    symbols('ps4_x ps4_y ps4_z'),
])  # 12x1 (4 legs * xyz)

# Define symbolic variables for velocity measurements (vs) for each leg (4 legs, xyz each)
vs = Matrix([
    symbols('vs1_x vs1_y vs1_z'),
    symbols('vs2_x vs2_y vs2_z'),
    symbols('vs3_x vs3_y vs3_z'),
    symbols('vs4_x vs4_y vs4_z'),
])  # 12x1 (4 legs * xyz)

# Reshape ps and vs to 12x1
ps = ps.reshape(12, 1)
vs = vs.reshape(12, 1)

# Define symbolic variables for foot positions (pzs) for each leg (z component only)
pzs = Matrix(symbols('pzs1 pzs2 pzs3 pzs4'))  # 4x1 (1 per leg)

y = Matrix.vstack(ps, vs, pzs)  # y = [ps; vs; pzs] (28x1)

# Define symbolic variables for control input (acceleration)
a_x, a_y, a_z = symbols('a_x a_y a_z')  # Acceleration in x, y, z
a = Matrix([a_x, a_y, a_z])  

# KALMAN FILTER
# Update _xhat symbolically
_xhat = _A * _xhat + _B * a 
At = _A.T 
Pm = _A * _P * At + Q  
yModel = _C * _xhat
ey = y - yModel
Ct = _C.T 
S = _C * Pm * Ct + R 
S_ey = S.LUsolve(ey)  # Efficient LU decomposition to solve for S_ey

# Update state vector _xhat
_xhat = _xhat + Pm * Ct * S_ey  # Kalman gain update

print(f'Updated States (IMU X POSITION): ')
pprint(_xhat[0])



