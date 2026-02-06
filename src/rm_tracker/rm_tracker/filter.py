"""
Filter.py (Hybrid NumPy + SciPy)
--------------------------------------
统一位姿滤波模块（简洁工程版 + NumPy/SciPy混合）

结构：
- 坐标: CV Kalman + 滑动窗口拟合预测 (抗抖 + 减滞后)
- 姿态: SO(3) 误差态 EKF (旋转滤波)
- 外部接口与原 rm_tracker 保持一致:
    update_dt(dt)
    kf_predict()
    kf_update(tvec, rvec)
    get_kf_state()
    reset_kf()
    set_params()
"""

import numpy as np
import cv2
import math
from collections import deque
from scipy import linalg, sparse
from scipy.sparse import linalg as spla


# ========== 通用数值工具 ==========
def safe_inv(A):
    A = np.array(A, dtype=float)
    if A.shape[0] <= 20:
        return np.linalg.inv(A)
    elif sparse.issparse(A):
        I = sparse.eye(A.shape[0], format="csr")
        return spla.spsolve(A, I)
    else:
        return linalg.inv(A)


def safe_svd(A):
    A = np.array(A, dtype=float)
    if A.shape[0] <= 50:
        return np.linalg.svd(A)
    else:
        return linalg.svd(A)


def finite(v):
    return np.isfinite(v).all()


# ========== SO(3) 工具 ==========
def so3_exp(phi):
    R, _ = cv2.Rodrigues(np.array(phi, dtype=float).reshape(3, 1))
    return R


def so3_log(R):
    r, _ = cv2.Rodrigues(np.array(R, dtype=float))
    return r.flatten()


# ========== 位置滤波 ==========
class PositionFilterHybrid:
    def __init__(self, dt=0.01, q_acc=3e-2, r_meas=1e-2,
                 win_size=8, poly_degree=2, alpha=0.7, future_scale=1.0):
        self.dt = float(dt)
        self.q_acc = q_acc
        self.r_meas = r_meas
        self.win_size = win_size
        self.poly_degree = poly_degree
        self.alpha = alpha
        self.future_scale = future_scale

        self.x = np.zeros((6, 1))      # 状态 [p(3), v(3)]
        self.P = np.eye(6) * 1e-2
        self._build_mats()

        self.t = 0.0
        self.hist = deque(maxlen=win_size)
        self.initialized = False

    def _build_mats(self):
        dt = self.dt
        I = np.eye(3)
        Z = np.zeros((3, 3))

        self.F = np.block([[I, dt * I], [Z, I]])
        self.H = np.block([I, Z])

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.q_acc
        Q11 = (dt4 / 4.0) * I
        Q12 = (dt3 / 2.0) * I
        Q22 = dt2 * I
        self.Q = q * np.block([[Q11, Q12], [Q12, Q22]])

        self.R = np.eye(3) * self.r_meas
        self.I6 = np.eye(6)

    def set_dt(self, dt):
        self.dt = float(dt)
        self._build_mats()

    def set_params(self, q_acc=None, r_meas=None,
                   win_size=None, poly_degree=None,
                   alpha=None, future_scale=None):
        if q_acc is not None:
            self.q_acc = float(q_acc)
        if r_meas is not None:
            self.r_meas = float(r_meas)
            self.R = np.eye(3) * self.r_meas
        if win_size is not None:
            self.win_size = int(win_size)
            self.hist = deque(maxlen=self.win_size)
        if poly_degree is not None:
            self.poly_degree = int(poly_degree)
        if alpha is not None:
            self.alpha = float(alpha)
        if future_scale is not None:
            self.future_scale = float(future_scale)
        self._build_mats()

    def reset(self):
        self.__init__(self.dt, self.q_acc, self.r_meas,
                      self.win_size, self.poly_degree,
                      self.alpha, self.future_scale)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.t += self.dt
        p = self.x[0:3, 0]
        v = self.x[3:6, 0]
        self._push_hist(self.t, p)
        return p.copy(), v.copy()

    def update(self, p_meas):
        z = np.array(p_meas, dtype=float).reshape(3, 1)
        if not self.initialized:
            self.x[0:3, 0] = z.flatten()
            self.initialized = True

        self.predict()

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ safe_inv(S)
        self.x += K @ y
        self.P = (self.I6 - K @ self.H) @ self.P

        p_kf = self.x[0:3, 0].copy()
        v_kf = self.x[3:6, 0].copy()

        p_fit = self._fit_predict(self.dt * self.future_scale)
        if p_fit is not None and finite(p_fit):
            p_fused = self.alpha * p_kf + (1.0 - self.alpha) * p_fit
        else:
            p_fused = p_kf

        self._push_hist(self.t, p_fused)
        self.x[0:3, 0] = p_fused
        return p_fused.copy(), v_kf.copy()

    def _push_hist(self, t, p3):
        p3 = np.array(p3, dtype=float).reshape(3)
        if finite(p3):
            self.hist.append((float(t), p3.copy()))

    def _fit_predict(self, dt_future):
        if len(self.hist) < 5:
            return None
        t_ref = self.hist[-1][0]
        ts = np.array([t - t_ref for t, _ in self.hist])
        xs = np.array([p[0] for _, p in self.hist])
        ys = np.array([p[1] for _, p in self.hist])
        zs = np.array([p[2] for _, p in self.hist])
        try:
            cx = np.polyfit(ts, xs, self.poly_degree)
            cy = np.polyfit(ts, ys, self.poly_degree)
            cz = np.polyfit(ts, zs, self.poly_degree)
        except Exception:
            return None
        t_future = float(dt_future)
        px = np.polyval(cx, t_future)
        py = np.polyval(cy, t_future)
        pz = np.polyval(cz, t_future)
        return np.array([px, py, pz], dtype=float)


# ========== 姿态滤波 ==========
class RotationEKF:
    def __init__(self, dt=0.01, q_rot=2e-3, r_rot=1e-2):
        self.dt = float(dt)
        self.Q = float(q_rot)
        self.Rm = np.eye(3) * float(r_rot)
        self.P = np.eye(3) * 1e-2
        self.R = np.eye(3)
        self.initialized = False

    def set_dt(self, dt):
        self.dt = float(dt)

    def set_params(self, q_rot=None, r_rot=None):
        if q_rot is not None:
            self.Q = float(q_rot)
        if r_rot is not None:
            self.Rm = np.eye(3) * float(r_rot)

    def reset(self):
        self.__init__(self.dt, self.Q, float(self.Rm[0, 0]))

    def predict(self):
        self.P += self.Q * self.dt * np.eye(3)
        return so3_log(self.R)

    def update(self, r_meas):
        r_meas = np.array(r_meas, dtype=float).reshape(3)
        Rz = so3_exp(r_meas)
        if not self.initialized:
            self.R = Rz
            self.initialized = True
            return so3_log(self.R)

        self.predict()
        y = so3_log(Rz @ self.R.T)
        S = self.P + self.Rm
        K = self.P @ safe_inv(S)
        delta = K @ y.reshape(3, 1)
        self.R = so3_exp(delta.flatten()) @ self.R
        self.R = self._fix_R(self.R)
        I = np.eye(3)
        self.P = (I - K) @ self.P
        return so3_log(self.R)

    def _fix_R(self, R):
        U, _, Vt = safe_svd(R)[:3]
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1
            Rn = U @ Vt
        return Rn


# ========== 总封装 ==========
class ArmorFilter:
    def __init__(self, dt=0.01,
                 q_acc=3e-2, r_meas=1e-2,
                 q_rot=2e-3, r_rot=1e-2):
        self.dt = float(dt)
        self.q_acc = q_acc
        self.r_meas = r_meas
        self.q_rot = q_rot
        self.r_rot = r_rot

        self.pos_filter = PositionFilterHybrid(dt=self.dt,
                                               q_acc=self.q_acc,
                                               r_meas=self.r_meas)
        self.rot_filter = RotationEKF(dt=self.dt,
                                      q_rot=self.q_rot,
                                      r_rot=self.r_rot)

    def set_params(self, q_acc=None, r_meas=None,
                   q_rot=None, r_rot=None,
                   win_size=None, poly_degree=None,
                   alpha=None, future_scale=None):
        if q_acc is not None:
            self.q_acc = q_acc
        if r_meas is not None:
            self.r_meas = r_meas
        if q_rot is not None:
            self.q_rot = q_rot
        if r_rot is not None:
            self.r_rot = r_rot

        self.pos_filter.set_params(q_acc=q_acc, r_meas=r_meas,
                                   win_size=win_size,
                                   poly_degree=poly_degree,
                                   alpha=alpha,
                                   future_scale=future_scale)
        self.rot_filter.set_params(q_rot=q_rot, r_rot=r_rot)

    def update_dt(self, dt: float) -> None:
        self.dt = float(dt)
        self.pos_filter.set_dt(self.dt)
        self.rot_filter.set_dt(self.dt)

    def kf_predict(self) -> None:
        self.pos_filter.predict()
        self.rot_filter.predict()

    def kf_update(self, tvec, rvec) -> None:
        self.pos_filter.update(tvec)
        self.rot_filter.update(rvec)

    def get_kf_state(self):
        p = self.pos_filter.x[0:3, 0]
        v = self.pos_filter.x[3:6, 0]
        r = so3_log(self.rot_filter.R)

        tvec3 = (float(p[0]), float(p[1]), float(p[2]))
        vel3  = (float(v[0]), float(v[1]), float(v[2]))
        rvec3 = (float(r[0]), float(r[1]), float(r[2]))
        return tvec3, vel3, rvec3

    def reset_kf(self) -> None:
        self.pos_filter.reset()
        self.rot_filter.reset()
