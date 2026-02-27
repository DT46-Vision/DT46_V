"""
ArmorEKF — 扩展卡尔曼滤波器

状态向量 x (9):
    [0] xc      — 机器人中心 x (世界/odom 系)
    [1] v_xc    — 中心 x 方向速度
    [2] yc      — 机器人中心 y
    [3] v_yc    — 中心 y 方向速度
    [4] za      — 当前可见装甲板高度
    [5] v_za    — 装甲板高度变化率
    [6] yaw     — 底盘偏航角 (连续展开，不取模)
    [7] v_yaw   — 偏航角角速度
    [8] r       — 中心到当前装甲板的半径

观测量 z (4):
    [0] xa      — 装甲板 x (世界/odom 系)
    [1] ya      — 装甲板 y
    [2] za      — 装甲板 z
    [3] yaw     — 底盘偏航角

过程模型 f (匀速模型，CV):
    xc'  = xc  + v_xc * dt
    yc'  = yc  + v_yc * dt
    za'  = za  + v_za * dt
    yaw' = yaw + v_yaw * dt
    v_*' = v_* (保持不变)
    r'   = r   (保持不变)

观测模型 h (几何约束):
    xa = xc - r * cos(yaw)
    ya = yc - r * sin(yaw)
    za = za
    yaw = yaw

过程噪声 Q:
    sigma2_q_xyz  — 作用于 (xc,v_xc)、(yc,v_yc)、(za,v_za)
    sigma2_q_yaw  — 作用于 (yaw,v_yaw)
    sigma2_q_r    — 作用于 r

观测噪声 R:
    diag( |r_xyz_factor * xa|, |r_xyz_factor * ya|, |r_xyz_factor * za|, r_yaw )

Python 使用示例 (接口保持和 C++ 类似):
    ekf = ArmorEKF()
    ekf.update_dt(dt)
    ekf.set_params(sigma2_q_xyz=20.0, sigma2_q_yaw=100.0, sigma2_q_r=800.0,
                    r_xyz_factor=0.05, r_yaw=0.02)
    ekf.init_from_armor(xa, ya, za, yaw, r_init=0.26)
    x_pred = ekf.kf_predict()
    x_post = ekf.kf_update((xa, ya, za, yaw))  # 传入观测
    x      = ekf.get_state()

说明:
- 半径 r 会被限制在 [min_r, max_r] 内 (默认 [0.12, 0.40])。
- 偏航角 yaw 假定在外部已经展开处理，不需要在滤波内部 wrap。
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

# ---------------- 工具函数 ----------------

def _safe_inv(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)

def _wrap_state(x: np.ndarray) -> np.ndarray:
    # yaw 在这里不 wrap，保持连续展开
    return x

# ---------------- EKF 主体 ----------------

@dataclass
class _NoiseParams:
    sigma2_q_xyz: float = 20.0   # 平移噪声
    sigma2_q_yaw: float = 100.0  # 偏航角噪声
    sigma2_q_r:   float = 800.0  # 半径噪声
    r_xyz_factor: float = 0.05   # 观测噪声因子 (位置)
    r_yaw:        float = 0.02   # 观测噪声 (yaw)

class ArmorEKF:
    def __init__(self):
        self.n = 9   # 状态维度
        self.m = 4   # 观测维度
        self.dt: float = 1.0/60.0  # 默认频率 ~60Hz
        self.noise = _NoiseParams()
        self.P = np.eye(self.n)    # 初始协方差矩阵
        self.x = np.zeros((self.n, 1))  # 状态向量
        self.I = np.eye(self.n)

        # 半径约束 (同 C++ tracker)
        self.min_r = 0.12
        self.max_r = 0.40

        # 缓存矩阵
        self.F = np.eye(self.n)
        self.Q = np.zeros((self.n, self.n))
        self.H = np.zeros((self.m, self.n))

        self._build_F()
        self._build_H()

        # 预测缓存
        self.x_pri = self.x.copy()
        self.P_pri = self.P.copy()

    # -------- 公共接口 --------

    def update_dt(self, dt: float):
        """更新时间步长 dt"""
        self.dt = float(dt)
        self._build_F()

    def set_params(self, sigma2_q_xyz=None, sigma2_q_yaw=None, sigma2_q_r=None,
                    r_xyz_factor=None, r_yaw=None):
        """更新滤波器参数"""
        if sigma2_q_xyz is not None: self.noise.sigma2_q_xyz = float(sigma2_q_xyz)
        if sigma2_q_yaw is not None: self.noise.sigma2_q_yaw = float(sigma2_q_yaw)
        if sigma2_q_r   is not None: self.noise.sigma2_q_r   = float(sigma2_q_r)
        if r_xyz_factor is not None: self.noise.r_xyz_factor = float(r_xyz_factor)
        if r_yaw        is not None: self.noise.r_yaw        = float(r_yaw)

    def set_radius_bounds(self, r_min: float, r_max: float):
        """设置半径范围"""
        self.min_r = float(r_min); self.max_r = float(r_max)

    def set_state(self, x0):
        """直接设置状态"""
        x0 = np.array(x0, dtype=float).reshape(self.n, 1)
        self.x = x0.copy()
        self.P = np.eye(self.n)
        self.x_pri = self.x.copy()
        self.P_pri = self.P.copy()

    def get_state(self):
        """获取当前状态"""
        return self.x.copy().reshape(self.n)

    def init_from_armor(self, xa: float, ya: float, za: float, yaw: float, r_init: float = 0.26):
        """根据装甲板反算机器人中心并初始化状态"""
        r = float(r_init)
        xc = xa + r * np.cos(yaw)
        yc = ya + r * np.sin(yaw)
        x0 = np.zeros((self.n, 1))
        x0[0] = xc;  x0[1] = 0.0
        x0[2] = yc;  x0[3] = 0.0
        x0[4] = za;  x0[5] = 0.0
        x0[6] = yaw; x0[7] = 0.0
        x0[8] = r
        self.set_state(x0)

    def kf_predict(self):
        """预测步骤"""
        self._build_F()
        self._build_Q()

        self.x_pri = self._f(self.x)
        self.P_pri = self.F @ self.P @ self.F.T + self.Q

        # 和 C++ 一样，预测值直接作为当前状态
        self.x = self.x_pri.copy()
        self.P = self.P_pri.copy()
        self._clamp_r_in_state()
        return self.get_state()

    def kf_update(self, z):
        """更新步骤 (传入观测 z = (xa, ya, za, yaw))"""
        z = np.array(z, dtype=float).reshape(self.m, 1)

        H = self._jacobian_h(self.x_pri)
        R = self._build_R(z)

        S = H @ self.P_pri @ H.T + R
        K = self.P_pri @ H.T @ _safe_inv(S)

        y = z - self._h(self.x_pri)

        self.x = self.x_pri + K @ y
        self.P = (self.I - K @ H) @ self.P_pri

        self._clamp_r_in_state()
        self.x = _wrap_state(self.x)
        return self.get_state()

    # -------- 内部函数 --------

    def _f(self, x):
        """过程模型 f(x)"""
        x_new = x.copy()
        dt = self.dt
        x_new[0] = x[0] + x[1] * dt
        x_new[2] = x[2] + x[3] * dt
        x_new[4] = x[4] + x[5] * dt
        x_new[6] = x[6] + x[7] * dt
        return x_new

    def _h(self, x):
        """观测模型 h(x)"""
        z = np.zeros((self.m, 1))
        xc, yc, za, yaw, r = float(x[0]), float(x[2]), float(x[4]), float(x[6]), float(x[8])
        z[0] = xc - r * np.cos(yaw)
        z[1] = yc - r * np.sin(yaw)
        z[2] = za
        z[3] = yaw
        return z

    def _build_F(self):
        """构建过程模型的雅可比 F"""
        dt = self.dt
        F = np.eye(self.n)
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        F[6, 7] = dt
        self.F = F

    def _build_Q(self):
        """构建过程噪声 Q"""
        dt = self.dt
        t2 = dt * dt; t3 = t2 * dt; t4 = t2 * t2
        q_xyz = float(self.noise.sigma2_q_xyz)
        q_yaw = float(self.noise.sigma2_q_yaw)
        q_r   = float(self.noise.sigma2_q_r)

        Q = np.zeros((self.n, self.n))

        def _cv_block(i_pos, i_vel, qacc):
            Q[i_pos, i_pos] += t4/4.0 * qacc
            Q[i_pos, i_vel] += t3/2.0 * qacc
            Q[i_vel, i_pos] += t3/2.0 * qacc
            Q[i_vel, i_vel] += t2      * qacc

        _cv_block(0, 1, q_xyz)
        _cv_block(2, 3, q_xyz)
        _cv_block(4, 5, q_xyz)
        _cv_block(6, 7, q_yaw)

        Q[8, 8] += t4/4.0 * q_r

        self.Q = Q

    def _build_H(self):
        """初始化 H (具体由 _jacobian_h 动态生成)"""
        self.H = np.zeros((self.m, self.n))

    def _jacobian_h(self, x):
        """观测模型 h(x) 的雅可比 H"""
        H = np.zeros((self.m, self.n))
        yaw = float(x[6]); r = float(x[8])
        H[0, 0] = 1.0
        H[0, 6] =  r * np.sin(yaw)
        H[0, 8] = -np.cos(yaw)
        H[1, 2] = 1.0
        H[1, 6] = -r * np.cos(yaw)
        H[1, 8] = -np.sin(yaw)
        H[2, 4] = 1.0
        H[3, 6] = 1.0
        return H

    def _build_R(self, z):
        """构建观测噪声 R"""
        r = np.zeros((self.m, self.m))
        f = float(self.noise.r_xyz_factor)
        r[0, 0] = abs(f * z[0, 0])
        r[1, 1] = abs(f * z[1, 0])
        r[2, 2] = abs(f * z[2, 0])
        r[3, 3] = float(self.noise.r_yaw)
        return r

    def _clamp_r_in_state(self):
        """限制半径 r 在[min_r, max_r]范围内"""
        if self.x[8, 0] < self.min_r:
            self.x[8, 0] = self.min_r
        elif self.x[8, 0] > self.max_r:
            self.x[8, 0] = self.max_r
