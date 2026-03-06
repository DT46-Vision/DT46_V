import numpy as np
from numba import njit

@njit(fastmath=True)
def _fast_ekf_predict(X, P, F, Q):
    # 将复杂的矩阵运算挪到这里，Numba 会将其编译为机器码
    X_new = F @ X
    P_new = F @ P @ F.T + Q
    return X_new, P_new

@njit(fastmath=True)
def _fast_ekf_update(X, P, H, R, Y, I):
    # 计算卡尔曼增益 K
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T

    # 更新状态 X
    X_new = X + K @ Y

    # Joseph form 更新 P (保证正定性)
    I_KH = I - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T

    return X_new, P_new

class ExtendedKalmanFilter:
    def __init__(self):
        # -----------------------------------------------------------
        # C++ 标准 9 维状态向量 X:
        # [0] xc: 车中心 x
        # [1] v_xc: 车中心 vx
        # [2] yc: 车中心 y
        # [3] v_yc: 车中心 vy
        # [4] za: 装甲板 z
        # [5] v_za: 装甲板 vz
        # [6] yaw: 偏航角 (连续)
        # [7] v_yaw: 角速度
        # [8] r: 车辆半径
        # -----------------------------------------------------------
        self.X = np.zeros(9)

        # 初始协方差 P (对角矩阵)
        self.P = np.eye(9)
        self.P[6, 6] = 1.0
        self.P[8, 8] = 1.0

        # 过程噪声参数
        self.s2qxyz = 20.0
        self.s2qyaw = 100.0
        self.s2qr   = 800.0

        # 观测噪声参数
        self.r_xyz_factor = 0.05
        self.r_yaw = 0.02

        # ==========================================================
        # 【性能优化】预分配矩阵内存，避免高频循环中 np.zeros 产生碎片
        # ==========================================================
        self.F = np.eye(9)           # 状态转移矩阵
        self.Q = np.zeros((9, 9))    # 过程噪声矩阵
        self.H = np.zeros((4, 9))    # 观测雅可比矩阵
        self.R = np.zeros((4, 4))    # 观测噪声矩阵
        self.I = np.eye(9)           # 单位矩阵

        # 缓存一些不需要重复计算的索引切片，稍微提升一点访问速度
        self.idx_pos_vel = [0, 2, 4, 6] # x, y, z, yaw 的索引
        self.dt = 0.0

        # ==========================================================
        # 【Numba 预热】在初始化时空跑一次，触发 JIT 编译，防止实战第一帧卡顿
        # ==========================================================
        dummy_Y = np.zeros(4)
        dummy_R = np.eye(4)  # 使用单位阵作为占位，避免 S 矩阵全为 0 导致不可逆
        
        _fast_ekf_predict(self.X, self.P, self.F, self.Q)
        
        # 注意下面这行：第四个参数必须是 dummy_R，不能是 self.R
        _fast_ekf_update(self.X, self.P, self.H, dummy_R, dummy_Y, self.I)


    def init_QR(self, q_xyz=20.0, q_yaw=100.0, q_r=800.0, r_xyz_factor=0.05, r_yaw=0.02):
        self.s2qxyz = q_xyz
        self.s2qyaw = q_yaw
        self.s2qr   = q_r
        self.r_xyz_factor = r_xyz_factor
        self.r_yaw = r_yaw

    def init_state(self, xa, ya, za, yaw, r0):
        xc = xa + r0 * np.cos(yaw)
        yc = ya + r0 * np.sin(yaw)

        # 使用 [:] 原位赋值，不改变内存地址
        self.X[:] = [xc, 0.0, yc, 0.0, za, 0.0, yaw, 0.0, r0]

        # 同样原位重置 P
        self.P.fill(0.0)
        np.fill_diagonal(self.P, 1.0)

    def predict(self, dt):
        """
        预测步 (Predict)
        """
        # 1. 更新 F 矩阵 (仅更新动态部分)
        # F 是单位矩阵，只需要修改对角线偏移位置的时间项
        self.F[0, 1] = dt
        self.F[2, 3] = dt
        self.F[4, 5] = dt
        self.F[6, 7] = dt

        # 2. 动态构建过程噪声矩阵 Q
        # 【优化】原地修改 self.Q，不创建新对象
        self.Q.fill(0) # 重置为0

        t2 = dt**2
        t3 = dt**3
        t4 = dt**4

        # XYZ 的噪声系数
        q_xyz_x = t4 / 4 * self.s2qxyz
        q_xyz_vx = t3 / 2 * self.s2qxyz
        q_xyz_vv = t2 * self.s2qxyz

        # 直接赋值给预分配的矩阵
        self.Q[0,0] = q_xyz_x; self.Q[0,1] = q_xyz_vx
        self.Q[1,0] = q_xyz_vx; self.Q[1,1] = q_xyz_vv

        self.Q[2,2] = q_xyz_x; self.Q[2,3] = q_xyz_vx
        self.Q[3,2] = q_xyz_vx; self.Q[3,3] = q_xyz_vv

        self.Q[4,4] = q_xyz_x; self.Q[4,5] = q_xyz_vx
        self.Q[5,4] = q_xyz_vx; self.Q[5,5] = q_xyz_vv

        # Yaw 的噪声
        q_yaw_x = t4 / 4 * self.s2qyaw
        q_yaw_vx = t3 / 2 * self.s2qyaw
        q_yaw_vv = t2 * self.s2qyaw
        self.Q[6,6] = q_yaw_x; self.Q[6,7] = q_yaw_vx
        self.Q[7,6] = q_yaw_vx; self.Q[7,7] = q_yaw_vv

        # 改为 t2 (dt的平方) 或直接使用 dt，让半径具备适应跳变的灵活性
        self.Q[8,8] = t2 * self.s2qr

        # 3. 执行预测
        # 注意：这里矩阵乘法 @ 仍然会产生临时的中间大矩阵，
        # 但相比反复 malloc Q和F，性能开销已经大幅降低。
        self.X, self.P = _fast_ekf_predict(self.X, self.P, self.F, self.Q)

        return self.X

    def update(self, measurement):
        """
        更新步 (Update)
        """
        Z = np.array(measurement)

        yaw = self.X[6]
        r = self.X[8]

        # cache trig values
        s_yaw = np.sin(yaw)
        c_yaw = np.cos(yaw)

        # 1. 更新观测雅可比矩阵 H
        # 【优化】原地修改
        self.H.fill(0)

        # Row 0: xa
        self.H[0, 0] = 1
        self.H[0, 6] = r * s_yaw
        self.H[0, 8] = -c_yaw

        # Row 1: ya
        self.H[1, 2] = 1
        self.H[1, 6] = -r * c_yaw
        self.H[1, 8] = -s_yaw

        # Row 2: za
        self.H[2, 4] = 1

        # Row 3: yaw
        self.H[3, 6] = 1

        # 2. 更新观测噪声 R
        # 【优化】原地修改
        self.R.fill(0)

        obs_x, obs_y, obs_z = Z[0], Z[1], Z[2]

        base_noise = 0.05  # 设置一个保底噪声
        self.R[0,0] = abs(self.r_xyz_factor * obs_x) + base_noise
        self.R[1,1] = abs(self.r_xyz_factor * obs_y) + base_noise
        self.R[2,2] = abs(self.r_xyz_factor * obs_z) + base_noise
        self.R[3,3] = self.r_yaw

        # 3. 计算预计观测值 h(x)
        xc, yc, za_state = self.X[0], self.X[2], self.X[4]

        # 预测的观测向量
        # 这里创建一个长度为4的小array开销可以接受，
        # 如果非要优化，可以 self.Z_pred[:] = ...
        Z_pred = np.array([
            xc - r * c_yaw,
            yc - r * s_yaw,
            za_state,
            yaw
        ])

        # 4. 计算残差 Y
        Y = Z - Z_pred

        # 5. 标准卡尔曼更新 (调用 Numba 加速函数)
        self.X, self.P = _fast_ekf_update(self.X, self.P, self.H, self.R, Y, self.I)

        return self.X
