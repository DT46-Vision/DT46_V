"""
Armor Tracker (IMM-6D, 位姿滤波：tvec+rvec)
--------------------------------------
- 订阅:  /detector/armors_info（每个 armor: armor_id, dx, dy, dz, rx, ry, rz）
- 过程:  选目标 → IMM/KF 融合(位姿+速度) → 相机→枪管 → 弹道解算 → 射击判定
- 输出:  (shoot_flag, yaw_sent, pitch_sent, msg_color)
- 坐标:  统一为 OpenCV 相机坐标 (X→右, Y→下, Z→前)

说明：
- 和原版一致，不改对外接口函数名；仅把内部 KF 升级为新方案（坐标CV+滑窗拟合，姿态EKF）。
- 兼容：kf_update(x,y,z) 仍存在（仅位置观测），主链路直接传 tvec/rvec 给滤波器。
- 可视化：坐标轴箭头使用滤波后的 rvec/tvec；蓝点=原始PnP，绿点=KF后，红点=弹道预测。
"""

import math
import cv2
import numpy as np
from rm_tracker.filter import ArmorFilter  # 你若保存为 filter.py，用这一行
from scipy.spatial.transform import Rotation as R

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0
EPS = 1e-9


class Object:
    def __init__(self, object_id=-1, x=0.0, y=0.0, z=0.0,
                 rx=0.0, ry=0.0, rz=0.0, vel=None):
        self.object_id = int(object_id)
        self.tvec = np.array([x, y, z], dtype=float).reshape(3,)
        self.rvec = np.array([rx, ry, rz], dtype=float).reshape(3,)
        self.vel  = np.zeros(3, dtype=float) if vel is None else np.array(vel, dtype=float).reshape(3,) # 加速度/速度向量，默认全 0

    @property
    def quat(self):
        R_obj = R.from_rotvec(self.rvec)
        return R_obj.as_quat()

    @property
    def rotation_matrix(self):
        return R.from_rotvec(self.rvec).as_matrix()

class Robot:
    def __init__(self, robot: Object, armor_0: Object, armor_1: Object, armor_2: Object, armor_3: Object):
        self.robot = robot
        self.armor_0 = armor_0
        self.armor_1 = armor_1
        self.armor_2 = armor_2
        self.armor_3 = armor_3

class ColorPrint:
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"


class Tracker:
    """
    输入: ArmorsCppMsg（每个 armor: armor_id, dx, dy, dz, rx, ry, rz）
    过程: 选目标 → KF/IMM 融合(位姿+速度) → 相机→枪管 → 弹道解算 → 射击判定
    输出: (shoot_flag, yaw_sent, pitch_sent, msg_color)
    坐标系统一为 OpenCV 相机坐标 (X→右, Y→下, Z→前)
    """
    def __init__(self, logger=None):
        self.logger = logger
        self.color = ColorPrint()

        # =================== 预封装旋转矩阵 ===================
        self.ROT_Y_180   = R.from_euler("y", 180, degrees=True)
        self.ROT_Y_POS90 = R.from_euler("y", 90, degrees=True)
        self.ROT_Y_NEG90 = R.from_euler("y", -90, degrees=True)
        self.ROT_X_POS15 = R.from_euler("x", 15, degrees=True)
        self.ROT_X_NEG15 = R.from_euler("x", -15, degrees=True)

        # 常用的组合旋转也提前算好
        self.OFFSET_BACK  = self.ROT_X_NEG15
        self.OFFSET_FRONT = self.ROT_X_POS15
        self.OFFSET_RIGHT = self.ROT_Y_NEG90 * self.ROT_X_POS15
        self.OFFSET_LEFT  = self.ROT_Y_POS90 * self.ROT_X_POS15

        # 四个装甲板相对位置（米）
        self.positions = np.array([
            [0.0, 0.0,  0.25],   # 后
            [0.0, 0.0, -0.25],   # 前
            [0.25, 0.0,  0.0],   # 右
            [-0.25, 0.0,  0.0]   # 左
        ])

        # ======== 相机参数 ==========
        self.fx = self.fy = self.cx = self.cy = 0.0
        self.dist = np.zeros(5, dtype=float)
        self.has_camera_info = False

        # ======== 外参 ==========
        self.camera_tx_m = 0.0
        self.camera_ty_m = -0.03
        self.camera_tz_m = 0.0
        self.camera_roll_deg = 0.0
        self.camera_pitch_deg = 0.0
        self.camera_yaw_deg = -4.0

        # ======== 追踪配置 ==========
        self.tracking_color = 0
        self.track_deep_tol = 10.0
        self.shoot_yaw_max = 1.5
        self.shoot_pitch_max = 1.0
        self.target_pos = (None, None)

        # ======== 滤波参数（直接暴露）==========
        # --- 坐标滤波 ---
        self.dt = 0.01
        self.q_acc = 0.05
        self.r_meas = 0.02
        self.win_size = 8
        self.poly_degree = 2
        self.alpha = 0.7
        self.future_scale = 1.0
        # --- 姿态滤波 ---
        self.q_rot = 0.002
        self.r_rot = 0.01

        # ======== 创建滤波器实例 ==========
        self.kf = ArmorFilter(dt=0.01)


        # ======== KF 状态控制 ==========
        self.use_kf = True
        self.max_lost_ms = 200.0
        self.lost_time_s = 0.0
        self.predict = False
        self.if_find = False

        # ======== 追踪目标 ==========
        self.tracking_armor = None
        self.tracking_armor_filtered = None
        self.tracking_robot = None

        # ======== 弹道参数 ==========
        self.use_ballistics = True
        self.bullet_speed_mps = 30.0
        self.gravity = 9.81
        self.extra_latency_s = 0.02
        self.max_iters_ballistic = 3
        self.max_target_speed_mps = 8.0

        # ======== 弹道稳定性 ==========
        self.ballistic_min_dist_m = 0.25
        self.ballistic_max_pitch_deg = 40.0
        self.ballistic_conv_tol_deg = 0.05
        self.ballistic_time_damping = 0.5

    # -------------------- 单位一致性工具 --------------------
    def _normalize_xyz_to_m(self, x: float, y: float, z: float):
        abs_max = max(abs(x), abs(y), abs(z))
        converted = False
        if abs_max > 50.0:
            x_m, y_m, z_m = x / 1000.0, y / 1000.0, z / 1000.0
            converted = True
        elif abs_max < 0.5:
            x_m, y_m, z_m = x, y, z
        else:
            x_m, y_m, z_m = x, y, z
        return float(x_m), float(y_m), float(z_m), converted

    def _normalize_vel_to_mps(self, vx: float, vy: float, vz: float):
        v_abs_max = max(abs(vx), abs(vy), abs(vz))
        converted = False
        if v_abs_max > 500.0:
            vx_m, vy_m, vz_m = vx / 1000.0, vy / 1000.0, vz / 1000.0
            converted = True
        else:
            vx_m, vy_m, vz_m = vx, vy, vz
        return float(vx_m), float(vy_m), float(vz_m), converted

    def update_dt(self, dt: float):
        dt = float(dt)
        if dt > 0.0:
            self.dt = dt
            self.kf.update_dt(dt)  # 直接同步到 ArmorFilter

    # -------------------- 主入口 --------------------
    def track_armor(self, msg):
        tracking_armor = self.select_tracking_armor(msg)
        tracking_armor_filtered, msg_color = self.filter(tracking_armor)   # 这里是 m，且 Y↓

        if not self.if_find:
            self.target_pos = None
            self.tracking_armor_filtered = None
            # self.tracking_robot = None
            return 0, 0.0, 0.0, msg_color

        if self.use_ballistics and self.bullet_speed_mps > 1e-3:
            yaw_sent, pitch_sent = self.solve_ballistic_angles(tracking_armor_filtered)
        else:
            yaw_sent, pitch_sent = self.tf_to_gun_angles_from_cam_xyz(tracking_armor_filtered.tvec)

        self.target_pos = (yaw_sent, pitch_sent)
        shoot_flag = self.if_shoot(yaw_sent, pitch_sent)
        return shoot_flag, yaw_sent, pitch_sent, msg_color

    # -------------------- 目标选择 --------------------
    def select_tracking_armor(self, msg):
        # armor_info = [Object(object_id = a.armor_id, x = a.dx, y = a.dy, z = a.dz, rx = a.rx, ry = a.ry, rz = a.rz) for a in msg.armors]
        armor_info = [Object(object_id = a.armor_id, x = a.dx, y = a.dy, z = a.dz, rz = a.rz) for a in msg.armors]
        tracking_armor = None
        if not armor_info:
            return None

        if self.tracking_color == 1:
            filtered = [ar for ar in armor_info if ar.object_id < 6]
        elif self.tracking_color == 0:
            filtered = [ar for ar in armor_info if ar.object_id > 5]
        else:
            return None

        if not filtered:
            return None

        if len(filtered) == 1:
            self.tracking_armor = filtered[0]
            return filtered[0]

        top_two = sorted(filtered, key=lambda ar: ar.z, reverse=True)[:2]
        tracking_armor = top_two[0]

        if (top_two[0].z - top_two[1].z) <= self.track_deep_tol:
            if abs(top_two[0].x) < abs(top_two[1].x):
                tracking_armor = top_two[0]
            else:
                tracking_armor = top_two[1]
        self.tracking_armor = tracking_armor
        return tracking_armor

    # -------------------- 滤波主链 --------------------
    def filter(self, tracking_armor):
        """
        主滤波函数：
        - 输入: 当前检测的 Armor（含 tvec/rvec）
        - 输出: (tvec3, vel3, rvec3, msg_color)
        """
        if not tracking_armor:
            self.tracking_armor = tracking_armor
            # 无目标时预测或重置
            if self.use_kf:
                self.lost_time_s += self.dt
                if self.lost_time_s * 1000.0 <= self.max_lost_ms and self.predict:
                    # 推进滤波
                    self.kf.kf_predict()
                    tvec3, vel3, rvec3 = self.kf.get_kf_state()
                    self.if_find = True
                else:
                    # 超时丢失，重置滤波器
                    self.kf.reset_kf()
                    tvec3, vel3, rvec3 = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                    self.predict = False
                    self.if_find = False
            else:
                # 不用滤波，直接置零
                tvec3, vel3, rvec3 = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
                self.if_find = False

        else:
            # 有目标：正常观测更新
            self.predict = True
            self.if_find = True
            armor = tracking_armor

            # 新版 Armor 存储 tvec / rvec
            tvec = np.asarray(armor.tvec, dtype=float)
            rvec = np.asarray(armor.rvec, dtype=float)

            # 自动单位判断 (mm -> m)
            if np.max(np.abs(tvec)) > 50.0:
                tvec = tvec / 1000.0

            if self.use_kf:
                self.lost_time_s = 0.0
                # 更新滤波器
                self.kf.kf_update(tvec, rvec)
                tvec3, vel3, rvec3 = self.kf.get_kf_state()
            else:
                # 无滤波，直接赋值
                tvec3 = tuple(tvec)
                vel3  = (0.0, 0.0, 0.0)
                rvec3 = tuple(rvec)

        # 颜色提示
        if self.tracking_color == 0:
            msg_color = f"追踪{self.color.RED}红色{self.color.RESET}装甲板"
        elif self.tracking_color == 1:
            msg_color = f"追踪{self.color.BLUE}蓝色{self.color.RESET}装甲板"
        else:
            msg_color = "追踪未知颜色装甲板"

        tracking_armor_filtered = self.pack_armor_filtered(tvec3, vel3, rvec3)
        self.tracking_armor_filtered = tracking_armor_filtered

        return tracking_armor_filtered, msg_color

    def pack_armor_filtered(self, tvec3, vel3, rvec3):
        """把滤波结果打包成 Object 对象并存储"""
        tracking_armor_filtered = Object()

        tracking_armor_filtered.tvec = np.asarray(tvec3, dtype=float).reshape(3,)
        tracking_armor_filtered.vel  = np.asarray(vel3, dtype=float).reshape(3,)
        tracking_armor_filtered.rvec = np.asarray(rvec3, dtype=float).reshape(3,)

        return tracking_armor_filtered

    # -------------------- 机器人中心估算 --------------------
    def estimate_robot_center(self, armor: Object):
        """
        根据装甲板的位置和姿态估算整个机器人的位置和姿态
        基于装甲板(自身)坐标系做位移：先旋转到世界系，再沿局部Z轴(前方)后退0.25m
        """
        # rvec -> Rotation 对象
        R_armor = R.from_rotvec(armor.rvec)

        # 加上固定的 Pitch +15°（注意：Rotation 相乘为左乘：先 R_armor，后 ROT_X_POS15）
        R_adjusted = R_armor * self.ROT_X_POS15

        # 在装甲板局部坐标系里的偏移向量：沿 -Z 方向后退 0.25 m
        offset_local = np.array([0.0, 0.0, -0.25], dtype=float)

        # 变换到世界系并叠加到装甲板位置（基于物体坐标轴的移动）
        robot_position = armor.tvec + R_adjusted.apply(offset_local)

        # 姿态仍采用调整后的旋转
        rvec_new = R_adjusted.as_rotvec()

        # 打包结果（保留 rvec；需要四元数可用 Object.quat）
        robot_pose = Object(
            object_id = armor.object_id,
            x=float(robot_position[0]), y=float(robot_position[1]), z=float(robot_position[2]),
            rx=float(rvec_new[0]), ry=float(rvec_new[1]), rz=float(rvec_new[2])
        )
        return robot_pose

    # -------------------- 四块装甲板预测 --------------------
    def estimate_robot(self, tracking_robot: Object):
        """
        根据机器人位置和姿态预测四个装甲板的位置和姿态
        """
        robot_position = tracking_robot.tvec
        R_robot = R.from_rotvec(tracking_robot.rvec)

        # 四个方向的旋转
        rotations = [
            R_robot * self.OFFSET_BACK,   # 后
            R_robot * self.OFFSET_FRONT,  # 前
            R_robot * self.OFFSET_RIGHT,  # 右
            R_robot * self.OFFSET_LEFT    # 左
        ]

        other_armors = []
        for i, (pos, R_adj) in enumerate(zip(self.positions, rotations)):
            armor_position = robot_position + R_robot.apply(pos)
            rvec_new = R_adj.as_rotvec()
            armor_obj = Object(
                object_id=tracking_robot.object_id,
                x=armor_position[0], y=armor_position[1], z=armor_position[2],
                rx=rvec_new[0], ry=rvec_new[1], rz=rvec_new[2]
            )
            other_armors.append(armor_obj)

        # 打包成 Robot 对象
        robot = Robot(tracking_robot,
                      other_armors[0], other_armors[1],
                      other_armors[2], other_armors[3])
        return robot

    # -------------------- 相机 xyz → 枪管 yaw/pitch（几何） --------------------
    def tf_to_gun_angles_from_cam_xyz(self, tvec):
        # 全程 Y 向下
        P_gun, _ = self.transform_cam_to_gun(tvec, None)
        X_gun, Y_gun, Z_gun = P_gun.flatten().tolist()
        if abs(Z_gun) < EPS:
            Z_gun = EPS

        yaw_deg   = math.degrees(math.atan2(X_gun, Z_gun))                         # 右偏为正，符合直觉
        pitch_deg = math.degrees(math.atan2(-Y_gun, math.hypot(X_gun, Z_gun)))     # 负号把“向上”映射为正
        return yaw_deg, pitch_deg

    # -------------------- 相机位置/速度 -> 枪管系 --------------------
    def _rotation_cam_to_gun(self):
        """
        计算从相机坐标系 (X→右, Y→下, Z→前)
        到枪管坐标系的旋转。
        intrinsic ZYX 顺序（先 Roll, 再 Pitch, 最后 Yaw）。
        """
        roll = self.camera_roll_deg
        pitch = self.camera_pitch_deg
        yaw = self.camera_yaw_deg

        return R.from_euler("ZYX", [yaw, pitch, roll], degrees=True)  # 返回 Rotation 对象

    def transform_cam_to_gun(self, P_cam: np.ndarray, V_cam: np.ndarray | None):
        """
        输入/输出都采用 OpenCV 坐标系 (X→右, Y→下, Z→前)。
        枪管系与相机系同向，仅存在固定外参位姿差 (R_extr, T)。
        """
        T = np.array([self.camera_tx_m, self.camera_ty_m, self.camera_tz_m], dtype=float).reshape(3,)

        R_extr = self._rotation_cam_to_gun()  # 这里返回一个 Rotation 对象
        P_gun = R_extr.apply(P_cam) + T
        V_gun = R_extr.apply(V_cam) if V_cam is not None else None

        # --- 位置 ---
        P_cam = np.asarray(P_cam, dtype=float).reshape(3,)
        P_gun = R_extr.apply(P_cam) + T

        # --- 速度 ---
        V_gun = None
        if V_cam is not None:
            V_cam = np.asarray(V_cam, dtype=float).reshape(3,)
            V_gun = R_extr.apply(V_cam)

        return P_gun, V_gun

    # -------------------- 弹道解算 --------------------
    def solve_ballistic_angles(self, target):
        """
        弹道解算 (OpenCV 坐标系版)
        ---------------------------------
        坐标系：X→右, Y→下, Z→前
        g > 0 表示重力方向 +Y (向下)
        pitch>0 代表“向上抬枪”，因此 pitch 为负时炮口朝下
        """
        v0 = max(float(self.bullet_speed_mps), 1e-3)
        g = float(self.gravity)             # 9.81, 向下为正
        tol_deg = float(self.ballistic_conv_tol_deg)
        max_pitch = float(self.ballistic_max_pitch_deg)
        t_damp = float(self.ballistic_time_damping)
        min_dist = float(self.ballistic_min_dist_m)

        P_cam = np.array(target.tvec)
        V_cam = np.array(target.vel)

        v_norm = float(np.linalg.norm(V_cam))
        if v_norm > self.max_target_speed_mps:
            V_cam *= self.max_target_speed_mps / (v_norm + EPS)

        P_gun, V_gun = self.transform_cam_to_gun(P_cam, V_cam)
        X0, Y0, Z0 = float(P_gun[0]), float(P_gun[1]), float(P_gun[2])

        dist0 = math.sqrt(X0 * X0 + Y0 * Y0 + Z0 * Z0)
        if dist0 < min_dist:
            yaw_deg   = math.degrees(math.atan2(X0, Z0))
            pitch_deg = -math.degrees(math.atan2(Y0, math.hypot(X0, Z0)))  # Y↓ -> pitch 向上为正
            return yaw_deg, pitch_deg

        t = dist0 / v0
        prev_yaw_deg = prev_pitch_deg = None
        yaw_deg = pitch_deg = 0.0

        for _ in range(int(self.max_iters_ballistic)):
            t_lead = max(0.0, t + float(self.extra_latency_s))

            if V_gun is None:
                X, Y, Z = X0, Y0, Z0
            else:
                X = X0 + float(V_gun[0]) * t_lead
                Y = Y0 + float(V_gun[1]) * t_lead
                Z = Z0 + float(V_gun[2]) * t_lead

            R_h = math.hypot(X, Z)
            if R_h < 1e-6:
                yaw = 0.0
                theta = -math.atan2(Y, max(R_h, 1e-6))  # 向上为负
            else:
                # 判别式: v0^4 + g*(g*R_h^2 - 2*Y*v0^2)
                disc = v0**4 + g * (g * R_h**2 - 2.0 * Y * v0**2)
                if disc >= 0.0:
                    sqrt_disc = math.sqrt(disc)
                    denom = g * R_h + EPS
                    tan_theta = (v0**2 - sqrt_disc) / denom
                    tan_theta = max(-1e3, min(1e3, tan_theta))
                    theta = -math.atan(tan_theta)  # 负号：pitch>0 表示抬头
                else:
                    theta = -math.atan2(Y, R_h)
                yaw = math.atan2(X, Z)

            cos_theta = math.cos(theta)
            if abs(cos_theta) < 1e-3:
                cos_theta = 1e-3
            t_calc = R_h / (v0 * cos_theta)
            t = (1.0 - t_damp) * t + t_damp * t_calc

            yaw_deg = yaw * RAD2DEG
            pitch_deg = theta * RAD2DEG
            pitch_deg = max(-max_pitch, min(max_pitch, pitch_deg))

            if prev_yaw_deg is not None and prev_pitch_deg is not None:
                if (abs(yaw_deg - prev_yaw_deg) <= tol_deg and
                    abs(pitch_deg - prev_pitch_deg) <= tol_deg):
                    break
            prev_yaw_deg, prev_pitch_deg = yaw_deg, pitch_deg

        if yaw_deg > 180.0:
            yaw_deg -= 360.0
        if yaw_deg <= -180.0:
            yaw_deg += 360.0

        return float(yaw_deg), float(pitch_deg)

    # def _draw_axes(self, cv_img, tvec, rvec, K, D):
    #     """
    #     在图像上绘制坐标轴箭头 (X=红, Y=绿, Z=蓝)
    #     """
    #     axis_len = 0.06 if np.linalg.norm(tvec) < 5.0 else 60.0
    #     axes_local = np.float64([
    #         [axis_len, 0, 0],
    #         [0, axis_len, 0],
    #         [0, 0, axis_len]
    #     ])
    #     origin_img, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, D)
    #     axes_img, _ = cv2.projectPoints(axes_local, rvec, tvec, K, D)

    #     # reshape 并检查 finite
    #     origin = origin_img.reshape(-1, 2)[0]
    #     axes   = axes_img.reshape(-1, 2)

    #     if not np.isfinite(origin).all() or not np.isfinite(axes).all():
    #         return  # 丢弃非法数据

    #     origin_tuple = (int(round(origin[0])), int(round(origin[1])))
    #     colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=红, Y=绿, Z=蓝

    #     for i, color in enumerate(colors):
    #         pt = axes[i]
    #         if not np.isfinite(pt).all():
    #             continue
    #         pt_tuple = (int(round(pt[0])), int(round(pt[1])))
    #         cv2.arrowedLine(cv_img, origin_tuple, pt_tuple, color, 2, tipLength=0.3)



    def _draw_armor_rect(self, cv_img, tvec, rvec, object_id, K, D):
        """
        在图像上绘制装甲板矩形
        """
        if object_id in [0, 6]:  # 英雄/哨兵
            armor_w, armor_h = 0.23, 0.127
        else:
            armor_w, armor_h = 0.135, 0.06

        half_w, half_h = armor_w / 2.0, armor_h / 2.0
        armor_pts_local = np.float64([
            [-half_w, -half_h, 0],
            [ half_w, -half_h, 0],
            [ half_w,  half_h, 0],
            [-half_w,  half_h, 0]
        ])
        armor_img, _ = cv2.projectPoints(armor_pts_local, rvec, tvec, K, D)
        armor_img = armor_img.reshape(-1, 2).astype(int)

        if np.isfinite(armor_img).all() and np.abs(armor_img).max() < 8000:
            for i in range(4):
                pt1 = tuple(armor_img[i])
                pt2 = tuple(armor_img[(i + 1) % 4])
                cv2.line(cv_img, pt1, pt2, (0, 255, 255), 2)  # 黄色矩形


    def display_result(self, cv_img: np.ndarray):
        """
        绘制追踪结果：
        - 蓝点=原始PnP
        - 绿点=KF输出
        - 红点=弹道预测
        - 黄框=预测的其他装甲板
        - 箭头=检测到的装甲板姿态
        """
        # --- 蓝点：原始PnP ---
        if self.tracking_armor is not None and self.has_camera_info:
            tvec_raw = self.tracking_armor.tvec
            if tvec_raw[2] > 1e-6:
                u = int(self.fx * tvec_raw[0] / tvec_raw[2] + self.cx)
                v = int(self.fy * tvec_raw[1] / tvec_raw[2] + self.cy)
                cv2.circle(cv_img, (u, v), 5, (255, 0, 0), -1)

        if self.tracking_armor_filtered is not None:
            armor_f = self.tracking_armor_filtered
            tvec_f, rvec_f, id = armor_f.tvec, armor_f.rvec, armor_f.object_id

            if np.isfinite(np.hstack([tvec_f, rvec_f])).all() and self.has_camera_info and tvec_f[2] > 1e-6:
                # 绿点（KF输出）
                u_f = int(self.fx * tvec_f[0] / tvec_f[2] + self.cx)
                v_f = int(self.fy * tvec_f[1] / tvec_f[2] + self.cy)
                cv2.circle(cv_img, (u_f, v_f), 5, (0, 255, 0), -1)

                K = np.array([[self.fx, 0.0, self.cx],
                            [0.0, self.fy, self.cy],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
                D = np.asarray(self.dist, dtype=np.float64)
                if D.size < 4:
                    D = np.zeros(5, dtype=np.float64)

                tvec = tvec_f.reshape(3, 1).astype(np.float64)
                rvec = rvec_f.reshape(3, 1).astype(np.float64)

                # # 检测到的装甲板：只画箭头
                # self._draw_axes(cv_img, tvec, rvec, K, D)
                self._draw_armor_rect(cv_img, tvec, rvec, id, K, D)
            # --- 红点：弹道预测 ---
            if self.target_pos is not None and tvec_f[2] > 1e-6 and self.has_camera_info:
                yaw_sent, pitch_sent = self.target_pos
                Z_pred = float(tvec_f[2])
                X_pred = math.tan(math.radians(yaw_sent)) * Z_pred
                Y_up = math.tan(math.radians(pitch_sent)) * math.hypot(Z_pred, X_pred)
                Y_pred = -Y_up
                u_pred = int(self.fx * X_pred / Z_pred + self.cx)
                v_pred = int(self.fy * Y_pred / Z_pred + self.cy)
                if abs(u_pred) < 5000 and abs(v_pred) < 5000:
                    cv2.circle(cv_img, (u_pred, v_pred), 5, (0, 0, 255), -1)

        # # --- 额外绘制 Robot 的预测四个装甲板 ---
        # if self.tracking_robot is not None and self.has_camera_info:
        #     K = np.array([[self.fx, 0.0, self.cx],
        #                 [0.0, self.fy, self.cy],
        #                 [0.0, 0.0, 1.0]], dtype=np.float64)
        #     D = np.asarray(self.dist, dtype=np.float64)
        #     if D.size < 4:
        #         D = np.zeros(5, dtype=np.float64)

        #     for armor in [self.tracking_robot.armor_0,
        #                 self.tracking_robot.armor_1,
        #                 self.tracking_robot.armor_2,
        #                 self.tracking_robot.armor_3]:
        #         tvec = armor.tvec.reshape(3, 1).astype(np.float64)
        #         rvec = armor.rvec.reshape(3, 1).astype(np.float64)

        #         if np.isfinite(np.hstack([tvec, rvec])).all() and tvec[2] > 1e-6:
        #             # 预测的装甲板：只画黄色矩形，不画箭头
        #             self._draw_armor_rect(cv_img, tvec, rvec, armor.object_id, K, D)

        return cv_img

    # -------------------- 射击判定 --------------------
    def if_shoot(self, yaw: float, pitch: float):
        return int(abs(yaw) <= self.shoot_yaw_max and
                   abs(pitch) <= self.shoot_pitch_max and
                   self.if_find)
