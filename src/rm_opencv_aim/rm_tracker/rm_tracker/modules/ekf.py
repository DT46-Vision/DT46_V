import numpy as np

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

    def init_QR(self, q_xyz=20.0, q_yaw=100.0, q_r=800.0, r_xyz_factor=0.05, r_yaw=0.02):
        self.s2qxyz = q_xyz
        self.s2qyaw = q_yaw
        self.s2qr   = q_r
        self.r_xyz_factor = r_xyz_factor
        self.r_yaw = r_yaw

    def init_state(self, xa, ya, za, yaw, r0):
        xc = xa + r0 * np.cos(yaw)
        yc = ya + r0 * np.sin(yaw)

        self.X = np.array([
            xc, 0,    # x, vx
            yc, 0,    # y, vy
            za, 0,    # z, vz
            yaw, 0,   # yaw, v_yaw
            r0        # r
        ])

        # 重置协方差
        self.P = np.eye(9)

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

        # Radius 的噪声
        self.Q[8,8] = t4 / 4 * self.s2qr

        # 3. 执行预测
        # 注意：这里矩阵乘法 @ 仍然会产生临时的中间大矩阵，
        # 但相比反复 malloc Q和F，性能开销已经大幅降低。
        self.X = self.F @ self.X
        self.P = self.F @ self.P @ self.F.T + self.Q

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

        self.R[0,0] = abs(self.r_xyz_factor * obs_x)
        self.R[1,1] = abs(self.r_xyz_factor * obs_y)
        self.R[2,2] = abs(self.r_xyz_factor * obs_z)
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

        # 5. 标准卡尔曼更新
        # S = H P H^T + R
        S = self.H @ self.P @ self.H.T + self.R

        try:
            # 使用 solve 代替 inv，数值上更稳定且通常更快
            # K = P H^T S^-1
            # 等价于 K = (S^-1 H P)^T ... 写法较多，这里保持原逻辑
            # K = self.P @ self.H.T @ np.linalg.inv(S)

            # 使用 lstsq 或者 solve
            # K = (np.linalg.solve(S.T, (self.P @ self.H.T).T)).T
            # 为了保持代码简单且不出错，这里还是用 inv，但在嵌入式上 solve 更优
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((9, 4))

        self.X = self.X + K @ Y

        # Joseph form 更新 P (数值稳定性)
        # P = (I - K H) P (I - K H)^T + K R K^T
        I_KH = self.I - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.X
