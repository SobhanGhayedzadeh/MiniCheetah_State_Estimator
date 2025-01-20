import re
import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(file_path):

    iterations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Regular expressions for identifying sections
    position_pattern = re.compile(r'^---------------------------- positions ------------------------------$')
    velocity_pattern = re.compile(r'^---------------------------- velocities -----------------------------$')
    a_pattern = re.compile(r'^-------------------------------- a ---------------------------------$')
    z_height_pattern = re.compile(r'^---------------------------- z height -------------------------------$')

    positions = []
    velocities = []
    a_data = []
    z_height = []

    # Flags to determine the current section being parsed
    in_positions = False
    in_velocities = False
    in_a = False
    in_z_height  = False

    # Parse the file line by line
    for line in lines:
        line = line.strip()

        # Check for section headers
        if position_pattern.match(line):
            in_positions = True
            in_velocities = False
            in_a = False
            in_z_height  = False
            positions = []
            # print('pos')
            continue
        elif velocity_pattern.match(line):
            in_positions = False
            in_velocities = True
            in_a = False
            in_z_height  = False
            velocities = []
            # print('vel')
            continue
        elif a_pattern.match(line):
            in_positions = False
            in_velocities = False
            in_a = True
            in_z_height  = False
            a_data = []
            # print('a')
            continue
        elif z_height_pattern.match(line):
            in_positions = False
            in_velocities = False
            in_a = False
            in_z_height  = True
            z_height = []
            continue

        elif line.startswith('--------'): 
            in_positions = False
            in_velocities = False
            in_a = False
            in_z_height  = False
            continue

        # Skip other non-target sections
        if line.startswith('-------------') and not (position_pattern.match(line) or velocity_pattern.match(line) or a_pattern.match(line) or z_height_pattern.match(line)):
            print('fuck')
            continue

        # Collect data for the current section
        if in_positions:
            positions.extend(map(float, line.split()))
        elif in_velocities:
            # print(line)
            velocities.extend(map(float, line.split()))
        elif in_a:
            a_data.extend(map(float, line.split()))
        elif in_z_height: 
            z_height.extend(map(float, line.split()))

        # End of an iteration when the next 'a' section starts or file ends
        if in_a and len(a_data) == 3:
            iterations.append({
                '_ps': positions,
                '_vs': velocities,
                'a': a_data,
                'psz': z_height
            })
            in_a = False

    return iterations

file_path = "log/estimator_log.txt" 
parsed_data = parse_log_file(file_path)

# Time step
dt = 0.002  # Assuming `controller_dt` is defined elsewhere

process_noise_pimu = 0.8               #    1           # 0.02      #1              0.8       DOWN:  NOISY and DRIFT
process_noise_vimu =  0.5              #    0.1         # 0.02      # 0.5           0.5       no difference
process_noise_pfoot =  0.0002          #    0.00002     # 0.02      # 0.00002      0.0002     Up: NOSIY and DRIFT
sensor_noise_pimu_rel_foot = 0.0001    #    0.0001      # 0.02      #  0.00001     0.00001    Up: NOISY DATA 
sensor_noise_vimu_rel_foot = 0.0001    #    0.0001      # 0.1       # 0.00001      0.00001    no difference 
sensor_noise_zfoot = 0.0001            #    0.001       # 0.001     # 0.00001      0.00001    no difference 

# Initialize matrices with corrected dimensions
_xhat = np.zeros((18, 1))  # State vector
_ps = np.zeros((3, 1))
_vs = np.zeros((3, 1))
_A = np.zeros((18, 18))  # System matrix
_B = np.zeros((18, 3))   # Control matrix
_C = np.zeros((28, 18))  # Observation matrix
_P = np.eye(18) * 100    # Covariance matrix
_Q0 = np.zeros((18, 18)) # Process noise covariance
_R0 = np.eye(28)         # Measurement noise covariance

# Set _A matrix
_A[:3, :3] = np.eye(3)  
_A[:3, 3:6] = dt * np.eye(3)
_A[3:6, 3:6] = np.eye(3)
_A[6:, 6:] = np.eye(12)

# Set _B matrix
_B[3:6, :3] = dt * np.eye(3)

C1 = np.block([
    [np.eye(3), np.zeros((3, 3))]
])
C2 = np.block([
    [np.zeros((3, 3)), np.eye(3)]
])

# Set _C matrix
_C[:3, :6] = C1
_C[3:6, :6] = C1
_C[6:9, :6] = C1
_C[9:12, :6] = C1
_C[:12, 6:18] = -np.eye(12)  # -1 times Identity matrix
_C[12:15, :6] = C2
_C[15:18, :6] = C2
_C[18:21, :6] = C2
_C[21:24, :6] = C2
_C[24, 8] = 1
_C[25, 11] = 1
_C[26, 14] = 1
_C[27, 17] = 1

# Set _Q0 matrix
_Q0[:3, :3] = (dt / 20.0) * np.eye(3)
_Q0[3:6, 3:6] = (dt * 9.8 / 20.0) * np.eye(3)
_Q0[6:, 6:] = dt * np.eye(12)


# Initialize Q and R matrices
Q = np.eye(18)  # Process noise covariance
R = np.eye(28)  # Measurement noise covariance


Q_scale = 1
R_scale =  1

# Update Q matrix
Q[:3, :3] = _Q0[:3, :3] * process_noise_pimu * Q_scale
Q[3:6, 3:6] = _Q0[3:6, 3:6] * process_noise_vimu  * Q_scale
Q[6:, 6:] = _Q0[6:, 6:] * process_noise_pfoot * Q_scale

# Update R matrix
R[:12, :12] = _R0[:12, :12] * sensor_noise_pimu_rel_foot * R_scale
R[12:24, 12:24] = _R0[12:24, 12:24] * sensor_noise_vimu_rel_foot * R_scale 
R[24:, 24:] = _R0[24:, 24:] * sensor_noise_zfoot * R_scale


def LinearKalmanFilter(_xhat, y, _P, a, Q, R):

    _xhat = _A @ _xhat + _B @ a 

    At = _A.T 
    Pm = _A @ _P @ At + Q  
    Ct = _C.T 
    yModel = _C @ _xhat  
    ey = y - yModel 
    S = _C @ Pm @ Ct + R  

    S_ey = np.linalg.solve(S, ey) # LU

    adjust = Pm @ Ct @ S_ey
    # _xhat_before_adjust = _xhat

    _xhat += adjust  # Update
    # print((Pm @ Ct @ S_ey)[0])

    S_C = np.linalg.solve(S, _C) 

    # Update the covariance matrix _P
    identity_matrix = np.eye(_P.shape[0])
    _P = (identity_matrix - Pm @ Ct @ S_C) @ Pm

    # Symmetrize _P to avoid numerical instability
    Pt = _P.T
    _P = (_P + Pt) / 2.0

    if np.linalg.det(_P[:2, :2]) > 1e-6:  # Equivalent to C++ `T(0.000001)`
        # Zero out blocks to maintain stability
        _P[:2, 2:] = 0
        _P[2:, :2] = 0

        # Scale the top-left 2x2 block
        _P[:2, :2] /= 10.0


    return _xhat , _P, yModel, ey, adjust


trust = 1
high_suspect_number = 100

first_Q = Q
first_R = R

_xhat_for_plot = []
_yModel_for_plot = []
_ey_for_plot = []
_y_for_plot = []
_iteration_for_plot = []

for i, iteration in enumerate(parsed_data):

    y = []
    y.extend(iteration["_ps"])
    y.extend(iteration["_vs"])
    y.extend(iteration["psz"])
    y = np.array(y) 
    y = y.reshape(28, 1)  
    
 
    a = np.array(iteration["a"])
    a = a.reshape(3 , 1) 

    for j in range(4):
        i1 = 3 * j
        qindex = 6 + i1
        rindex1 = i1
        rindex2 = 12 + i1
        rindex3 = 24 + j

        # Update Q block
        Q[qindex:qindex + 3, qindex:qindex + 3] *= (1 + (1 - trust) * high_suspect_number)

        # Update R blocks
        R[rindex1:rindex1 + 3, rindex1:rindex1 + 3] *= 1
        R[rindex2:rindex2 + 3, rindex2:rindex2 + 3] *= (1 + (1 - trust) * high_suspect_number)
        R[rindex3, rindex3] *= (1 + (1 - trust) * high_suspect_number)

    _xhat , _P, yModel, ey, adjust = LinearKalmanFilter(_xhat, y, _P, a, Q, R)

    _xhat_for_plot.append(_xhat)
    _yModel_for_plot.append(yModel)
    _ey_for_plot.append(ey)
    _y_for_plot.append(y)
    _iteration_for_plot.append(i)

    print(f'Iteration: {i}')
    print(f'COM POSITIONS X Y Z: ')
    print(_xhat[0], _xhat[1], _xhat[2])
    # print(f'Adjusting: {adjust[0]}')

    # sp.pprint(f'Q: {Q}')
    # sp.pprint(f'R: {R}')

    Q = first_Q
    R = first_R
    # if i == 4: 
    #     break


import matplotlib.pyplot as plt

def plot_data(_xhat_for_plot, _yModel_for_plot, _ey_for_plot, _y_for_plot, _iteration_for_plot):
    # Extract components from the data for plotting
    com_position = [xhat[:3] for xhat in _xhat_for_plot]
    com_velocity = [xhat[3:6] for xhat in _xhat_for_plot]
    leg_positions_xhat = [xhat[6:] for xhat in _xhat_for_plot]

    yModel_positions = [[ym[0:3], ym[3:6], ym[6:9], ym[9:12]] for ym in _yModel_for_plot]
    yModel_velocities = [[ym[12:15], ym[15:18], ym[18:21], ym[21:24]] for ym in _yModel_for_plot]
    yModel_heights = [ym[24:28] for ym in _yModel_for_plot]

    y_positions = [[y[0:3], y[3:6], y[6:9], y[9:12]] for y in _y_for_plot]
    y_velocities = [[y[12:15], y[15:18], y[18:21], y[21:24]] for y in _y_for_plot]
    y_heights = [y[24:28] for y in _y_for_plot]

    ey_positions = [[ey[0:3], ey[3:6], ey[6:9], ey[9:12]] for ey in _ey_for_plot]
    ey_velocities = [[ey[12:15], ey[15:18], ey[18:21], ey[21:24]] for ey in _ey_for_plot]
    ey_heights = [ey[24:28] for ey in _ey_for_plot]

    # Plot 1: COM Position
    plt.figure(figsize=(12, 8))
    plt.plot(_iteration_for_plot, [cp[0] for cp in com_position], label='X')
    plt.plot(_iteration_for_plot, [cp[1] for cp in com_position], label='Y')
    plt.plot(_iteration_for_plot, [cp[2] for cp in com_position], label='Z')
    plt.title("COM Position")
    plt.legend()
    # plt.show()

    # Plot 2: COM Velocity
    plt.figure(figsize=(12, 8))
    plt.plot(_iteration_for_plot, [cv[0] for cv in com_velocity], label='Vx')
    plt.plot(_iteration_for_plot, [cv[1] for cv in com_velocity], label='Vy')
    plt.plot(_iteration_for_plot, [cv[2] for cv in com_velocity], label='Vz')
    plt.title("COM Velocity")
    plt.legend()
    # plt.show()

    # Plot 3: Leg Positions (xhat)
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [lp[leg_idx * 3] for lp in leg_positions_xhat], label=f'Leg {leg_idx} X')
        plt.plot(_iteration_for_plot, [lp[leg_idx * 3 + 1] for lp in leg_positions_xhat], label=f'Leg {leg_idx} Y')
        plt.plot(_iteration_for_plot, [lp[leg_idx * 3 + 2] for lp in leg_positions_xhat], label=f'Leg {leg_idx} Z')
    plt.title("Leg Positions (xhat)")
    plt.legend()
    # plt.show()

    # Plot 4: yModel Positions
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [ymp[leg_idx][0] for ymp in yModel_positions], label=f'Leg {leg_idx} X')
        plt.plot(_iteration_for_plot, [ymp[leg_idx][1] for ymp in yModel_positions], label=f'Leg {leg_idx} Y')
        plt.plot(_iteration_for_plot, [ymp[leg_idx][2] for ymp in yModel_positions], label=f'Leg {leg_idx} Z')
    plt.title("yModel Positions")
    plt.legend()
    # plt.show()

    # Plot 5: yModel Velocities
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [ymv[leg_idx][0] for ymv in yModel_velocities], label=f'Leg {leg_idx} Vx')
        plt.plot(_iteration_for_plot, [ymv[leg_idx][1] for ymv in yModel_velocities], label=f'Leg {leg_idx} Vy')
        plt.plot(_iteration_for_plot, [ymv[leg_idx][2] for ymv in yModel_velocities], label=f'Leg {leg_idx} Vz')
    plt.title("yModel Velocities")
    plt.legend()
    # plt.show()

    # Plot 6: yModel Heights
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [ymh[leg_idx] for ymh in yModel_heights], label=f'Leg {leg_idx} Height')
    plt.title("yModel Heights")
    plt.legend()
    # plt.show()

    # Plot 7: y Positions
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [yp[leg_idx][0] for yp in y_positions], label=f'Leg {leg_idx} X')
        plt.plot(_iteration_for_plot, [yp[leg_idx][1] for yp in y_positions], label=f'Leg {leg_idx} Y')
        plt.plot(_iteration_for_plot, [yp[leg_idx][2] for yp in y_positions], label=f'Leg {leg_idx} Z')
    plt.title("y Positions")
    plt.legend()
    # plt.show()

    # Plot 8: y Velocities
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [yv[leg_idx][0] for yv in y_velocities], label=f'Leg {leg_idx} Vx')
        plt.plot(_iteration_for_plot, [yv[leg_idx][1] for yv in y_velocities], label=f'Leg {leg_idx} Vy')
        plt.plot(_iteration_for_plot, [yv[leg_idx][2] for yv in y_velocities], label=f'Leg {leg_idx} Vz')
    plt.title("y Velocities")
    plt.legend()
    # plt.show()

    # Plot 9: y Heights
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [yh[leg_idx] for yh in y_heights], label=f'Leg {leg_idx} Height')
    plt.title("y Heights")
    plt.legend()
    # plt.show()

    # Plot 10: ey Data
    plt.figure(figsize=(12, 8))
    for leg_idx in range(4):
        plt.plot(_iteration_for_plot, [ep[leg_idx][0] for ep in ey_positions], label=f'Leg {leg_idx} X')
        plt.plot(_iteration_for_plot, [ep[leg_idx][1] for ep in ey_positions], label=f'Leg {leg_idx} Y')
        plt.plot(_iteration_for_plot, [ep[leg_idx][2] for ep in ey_positions], label=f'Leg {leg_idx} Z')
        plt.plot(_iteration_for_plot, [ev[leg_idx][0] for ev in ey_velocities], linestyle='--', label=f'Leg {leg_idx} Vx')
        plt.plot(_iteration_for_plot, [ev[leg_idx][1] for ev in ey_velocities], linestyle='--', label=f'Leg {leg_idx} Vy')
        plt.plot(_iteration_for_plot, [ev[leg_idx][2] for ev in ey_velocities], linestyle='--', label=f'Leg {leg_idx} Vz')
        plt.plot(_iteration_for_plot, [eh[leg_idx] for eh in ey_heights], linestyle=':', label=f'Leg {leg_idx} Height')
    plt.title("ey Data")
    plt.legend()
    plt.show()


# Call the function
plot_data(_xhat_for_plot, _yModel_for_plot, _ey_for_plot, _y_for_plot, _iteration_for_plot)
