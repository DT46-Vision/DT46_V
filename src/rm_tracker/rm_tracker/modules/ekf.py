import numpy as np
from numba import njit
import math

@njit(fastmath=True)
def _fast_ekf_predict(X, P, F, Q):
    # 将复杂的矩阵运算挪到这里，Numba 会将其编译为机器码
    X_new = F @ X
    P_new = F @ P @ F.T + Q
    return X_new, P_new

@njit(fastmath=True)
def _fast_ekf_update(X, P, H, R, Y, I):
    # 计算 Innovation 协方差 S
    S = H @ P @ H.T + R
    
    # --- 【新增：马氏距离计算与野值剔除】 ---
    # 计算马氏距离的平方: D^2 = Y^T * S^-1 * Y
    # 使用 np.linalg.solve 求解 S^-1 * Y
    S_inv_Y = np.linalg.solve(S, Y)
    mahalanobis_sq = np.dot(Y, S_inv_Y)
    
    # 自由度为 4 (x, y, z, yaw) 的卡方分布，99% 置信区间的临界值约为 13.28
    # 如果马氏距离平方大于该阈值，判定为野值，直接拒绝更新，返回纯预测值
    if mahalanobis_sq > 15.0: 
        # 【关键修复】必须返回 .copy()，保证返回的数组内存布局与新建数组一致
        # 防止 Numba 因内存特征变化而每一帧都在疯狂重编译
        return X.copy(), P.copy()
    # ----------------------------------------

    # 计算卡尔曼增益 K
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
        dummy_X = np.zeros(9, dtype=np.float64)
        dummy_P = np.eye(9, dtype=np.float64)
        dummy_F = np.eye(9, dtype=np.float64)
        dummy_Q = np.zeros((9, 9), dtype=np.float64)
        dummy_H = np.zeros((4, 9), dtype=np.float64)
        dummy_R = np.eye(4, dtype=np.float64)
        dummy_Y = np.zeros(4, dtype=np.float64)
        dummy_I = np.eye(9, dtype=np.float64)

        _fast_ekf_predict(dummy_X, dummy_P, dummy_F, dummy_Q)

        _fast_ekf_update(dummy_X, dummy_P, dummy_H, dummy_R, dummy_Y, dummy_I)

    def init_QR(self, q_xyz=20.0, q_yaw=100.0, q_r=800.0, r_xyz_factor=0.05, r_yaw=0.02, stable_dist = 1.5):
        self.s2qxyz = q_xyz
        self.s2qyaw = q_yaw
        self.s2qr   = q_r
        self.r_xyz_factor = r_xyz_factor
        self.r_yaw = r_yaw
        self.stable_dist = stable_dist

    def init_state(self, xa, ya, za, yaw, r0):
        offset_x = r0 * np.cos(yaw)
        offset_y = r0 * np.sin(yaw)

        # 默认使用加法公式计算
        xc = xa + offset_x
        yc = ya + offset_y

        norm_a_sq = xa**2 + ya**2
        norm_c_sq = xc**2 + yc**2

        # 如果发现装甲板比车体中心还远，说明 yaw 的法向量反了
        if norm_a_sq > norm_c_sq:
            # 直接翻转 yaw 角度，并重新计算真正的中心点
            # 引入 math 模块中的 pi，或者传入 numpy 的 np.pi
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi 
            
            # 使用修正后的 yaw 重新计算 (此时相当于减法，但 yaw 的状态也被正确更新了)
            offset_x = r0 * np.cos(yaw)
            offset_y = r0 * np.sin(yaw)
            xc = xa + offset_x
            yc = ya + offset_y

        # 初始化状态向量，此时传入的 yaw 已经是修正过后的了
        self.X[:] = [xc, 0.0, yc, 0.0, za, 0.0, yaw, 0.0, r0]

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

        # 计算当前观测点到云台中心的水平距离 (假设世界坐标系原点在云台)
        # 如果你的世界系原点就是当前相机/云台投影点，可以直接用模长
        dist_h = math.sqrt(obs_x**2 + obs_y**2)

        base_noise = 0.05  # 设置一个保底噪声
        self.R[0,0] = abs(self.r_xyz_factor * obs_x) + base_noise
        self.R[1,1] = abs(self.r_xyz_factor * obs_y) + base_noise
        self.R[2,2] = abs(self.r_xyz_factor * obs_z) + base_noise
        
        # 【新增：远距离 Yaw 角信任降级】
        # 设定 1.5 米为稳定阈值。超过此距离，Yaw 噪声迅速放大
        if dist_h > self.stable_dist:
            # 使用二次方放大噪声，使其在 3 米时几乎完全不信任 PnP 的 Yaw
            dynamic_r_yaw = self.r_yaw * ((dist_h / 1.5) ** 2) 
            self.R[3,3] = min(dynamic_r_yaw, 10.0) # 设置一个上限防止矩阵奇异
        else:
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
