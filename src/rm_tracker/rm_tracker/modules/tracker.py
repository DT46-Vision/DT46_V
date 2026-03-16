import math
import numpy as np
from .ekf import ExtendedKalmanFilter
import cv2
from dataclasses import dataclass
from typing import List

# 常量定义
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

@dataclass
class RobotAppearance:
    """
    装甲板外观数据类
    """
    __slots__ = ['id', 'armor_width', 'armor_height', 'armor_diagonal', 'robot_r1', 'robot_r2']

    def __init__(self, id, armor_width, armor_height, robot_r1, robot_r2):
        self.id = id
        self.armor_width = armor_width
        self.armor_height = armor_height
        self.armor_diagonal: float = math.sqrt(self.armor_width ** 2 + self.armor_height ** 2)
        self.robot_r1 = robot_r1
        self.robot_r2 = robot_r2

# 使用列表存储所有装甲板外观实例
robot_list: List[RobotAppearance] = [
    RobotAppearance(id=0, armor_width=230.0, armor_height=125.0, robot_r1=0.3, robot_r2=0.285),
    RobotAppearance(id=1, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=2, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=3, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=4, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=5, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=6, armor_width=230.0, armor_height=125.0, robot_r1=0.3, robot_r2=0.285),
    RobotAppearance(id=7, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=8, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=9, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=10, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21),
    RobotAppearance(id=11, armor_width=135.0, armor_height=125.0, robot_r1=0.23, robot_r2=0.21)
]

def normalize_angle(angle):
    """
    将角度限制在 [-pi, pi] 之间
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

def shortest_angular_distance(from_rad, to_rad):
    """
    计算传入的两个角度之间的最短角距离 (对应 ROS angles::shortest_angular_distance)
    返回值的范围是 [-pi, pi]
    """
    return normalize_angle(to_rad - from_rad)

class ColorPrint:
    __slots__ = ['PINK', 'CYAN', 'GREEN', 'RED', 'BLUE', 'RESET']
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"

class Armor:
    # 限制允许的属性，省掉 __dict__ 的内存开销
    __slots__ = ['id', 'pos', 'yaw', 'index', 'dist', 'angle_diff', 'target_to_armor_dist']

    def __init__(self, id, x, y, z, yaw, index=None):
        self.id = id
        self.pos = np.array([x, y, z])
        self.yaw = yaw

        # 之前在字典里乱塞的属性，现在正式给名分
        self.index = index      # 0,1,2,3 (装甲板序号)
        self.dist = np.linalg.norm(self.pos)
        self.angle_diff = 0.0   # 枪口偏离目标的角度
        self.target_to_armor_dist = 0.0
    def clone(self):
        """
        【关键修复】手写极速克隆，替代耗时百倍的 copy.deepcopy
        """
        a = Armor(self.id, self.pos[0], self.pos[1], self.pos[2], self.yaw, self.index)
        a.angle_diff = self.angle_diff
        a.dist = self.dist
        a.target_to_armor_dist = self.target_to_armor_dist
        return a

class Tracker:
    # 状态机枚举
    LOST = 0        # 丢失
    DETECTING = 1   # 检测中
    TRACKING = 2    # 跟踪中
    TEMP_LOST = 3   # 短暂丢失

    def __init__(self):
        self.c = ColorPrint()

        self.target_color = 0                               # 目标敌方阵营 (0: RED, 1: BLUE)

        self.cam_to_gun_pos = np.array([0.0, -0.05, 0.0])   # [外参] 相机相对于枪口的平移向量 (x:右, y:下, z:前)
        self.cam_to_gun_rpy = np.array([0.0, 0.0, 0.0])     # [外参] 相机相对于枪口的旋转欧拉角 (Roll, Pitch, Yaw)

        self.ekf = ExtendedKalmanFilter()                   # 扩展卡尔曼滤波器 (EKF) 核心实例

        self.dt = 0.01
        self.last_time = None                               # 上一帧的时间戳，用于计算帧间时间差 dt

        self.max_match_distance = 0.2                       # [匹配] EKF 预测值与观测值的最大欧氏距离阈值 (m)
        self.max_match_yaw_diff = 1.0                       # [匹配] 判定装甲板切换 (Armor Jump) 的 Yaw 角度差阈值 (rad)

        self.jump_cooldown = 0                              # [新增] 跳变冷却帧数计数器
        self.jump_cooldown_max = 20                         # [新增] 发生跳变后，10帧内拒绝再次跳变

        self.tracking_thres = 5                             # 进入 TRACKING 状态所需的连续检测帧数
        self.lost_thres = 10                                # 进入 LOST 状态所需的连续丢失帧数
        self.tracker_state = self.LOST                      # 跟踪器 FSM 当前状态 (LOST/DETECTING/TRACKING/TEMP_LOST)
        self.detect_count = 0                               # 连续检测帧计数器 (用于状态确认)
        self.lost_count = 0                                 # 连续丢失帧计数器 (用于状态确认)

        self.dist_tol = 0.15                                # [预处理] 同一装甲板不同观测点的距离容差 (防止误判)
        self.tracked_id = None                              # 当前锁定追踪的装甲板 ID
        self.tracked_armor = None                           # 上一帧匹配成功的装甲板对象 (用于时序关联)

        self.debug_yaw_armors = []                          # 当前帧所有有效的装甲板观测数据

        self.target_state = np.zeros(9)                     # EKF 预测的 9 维系统状态向量
        self.ekf_QR_params = None                           # EKF QR 参数
        self.radius_params = None                           # 旋转半径参数

        self.last_yaw = 0.0                                 # 上一帧的连续化 Yaw 角 (用于处理角度跳变与去卷绕)
        self.dz = 0.0                                       # [几何] 当前板与另一组板的高度差 (z轴偏移)
        self.another_r = 0.23                               # [几何] 另一组装甲板的旋转半径

        self.system_delay = 0.1

        self.spin = False                                   # [几何] 小陀螺旋转标准位

        self.min_spinning_frame = 10                        # 小陀螺旋转的最小帧数
        self.spinning_frame_lost = 5                        # 小陀螺帧数丢失阈值

        self.min_spinning_frame_count = 0                   # 小陀螺帧数计数器
        self.spinning_frame_lost_count = 0                  # 小陀螺帧数丢失计数器
        self.min_spinning_vel = 5.0                         # 小陀螺最低门限

        self.target = None                                  # 最终解算出的目标状态
        self.bullet_speed = 28.0                            # 弹道速度 (m/s)
        self.muzzle_target = None                           # 弹道解算后的目标点

        self.shootable_dist = 3.0                           # [弹道] 允许发射的弹道距离阈值 (m)
        self.distance_decress_ratio = 0.60                  # [弹道] 距离递减比例
        self.yaw_tolerance_deg = 5.0                        # 允许发射的 yaw 角度阈值
        self.pitch_tolerance_deg = 2.0                      # 允许发射的 pitch 角度阈值
        self.yaw_tolerance_deg_mix = self.yaw_tolerance_deg # 混合模式下的 yaw 角度阈值
        self.pitch_tolerance_deg_mix = self.pitch_tolerance_deg # 混合模式下的 pitch 角度阈值

        self.gimbal_control = None                          # 云台控制信息

        self.log_buffer = []                                # 系统日志缓存队列 (用于调试/可视化)
        self.text_size = 1                                  # 文字大小系数

    def _log(self, log_type, msg):
        """
        log_type: 用于在 Node 中决定是否节流，例如 "state", "jump", "debug"
        msg: 具体内容
        """
        self.log_buffer.append((log_type, msg))

    def update_dt(self, current_ros_time):
        if self.last_time is None:
            self.last_time = current_ros_time
            return 0.01

        duration = (current_ros_time - self.last_time).nanoseconds / 1e9

        # 严格限制最大 dt 为 0.03s (约 33fps 的底线)
        # 如果卡顿超过此时间，宁可让滤波器认为时间流逝得慢，也不能让 Q 矩阵爆炸
        dt = max(0.001, min(duration, 0.03))

        self.last_time = current_ros_time
        return dt

    def cam_to_world(self, tf, raw_xyz, imu_rpy):
        """
        将相机系坐标转换为惯性系(世界系)坐标
        保留您原来的 tf 接口逻辑
        """
        normal_axis_xyz = tf.rotate_pos_axis(raw_xyz, [-90, 0, -90], 'xyz') # 相机系 -> IMU系 (假设 rotate_pos_axis 处理了外参旋转)
        world_xyz = tf.rotate_pos_axis(normal_axis_xyz, imu_rpy, 'xyz')     # IMU系 -> 世界系 (根据陀螺仪 RPY)
        return world_xyz

    def world_to_cam(self, tf, world_xyz, imu_rpy):
        """
        world_xyz: [x, y, z] - 世界惯性系下的坐标
        imu_rpy: [roll, pitch, yaw] - 电控 IMU 反馈的实时欧拉角 (单位: 度)
        返回: 原始相机系下的 [x, y, z]
        """
        inv_imu_rpy = [-imu_rpy[2], -imu_rpy[1], -imu_rpy[0]] # 逆向动态转换：从世界系转回法向轴系 (先逆转 IMU)
        normal_axis_xyz = tf.rotate_pos_axis(world_xyz, inv_imu_rpy, 'zyx') # 必须使用 'zyx' 顺序来抵消正向的 'xyz'
        raw_xyz = tf.rotate_pos_axis(normal_axis_xyz, [90, 0, 90], 'zyx')

        return raw_xyz

    def process_armors(self, tf, msg, imu_rpy):
        """
        【预处理】
        1. 颜色过滤 (完全保留您的逻辑：1追<6, 0追>5)
        2. 坐标转换 (Cam -> World)
        3. Yaw角修正 (180 - yaw)
        4. 打包返回所有候选列表
        """
        armors = []
        debug_yaw_armors = []

        if not msg.armors:
            return armors

        gimbal_yaw_deg = imu_rpy[2]    # 获取当前云台在世界系下的 Yaw (来自 IMU), imu_rpy[2] 是 yaw (角度制)

        for a in msg.armors:
            aid = int(a.armor_id)

            if self.target_color == 1 and aid >= 6:    # BLUE TRACKING
                continue
            if self.target_color == 0 and aid < 6:    # RED TRACKING
                continue

            raw_pos = [a.dx / 1000.0, a.dy / 1000.0, a.dz / 1000.0] # [x, y, z] (mm -> m)

            world_pos = self.cam_to_world(tf, raw_pos, imu_rpy)     # 使用 tf 接口转到世界系

            pnp_yaw_deg = -float(a.yaw) # PnP 算出的相对角度, 这里简单做翻转保证数值方向和世界系对齐
            debug_yaw_armors.append(Armor(aid, world_pos[0], world_pos[1], world_pos[2], pnp_yaw_deg)) # 用于 debug 视觉 PnP 数据

            world_yaw_deg = gimbal_yaw_deg + pnp_yaw_deg # Yaw 角处理：叠加 IMU -> 世界系 Yaw = 云台 Yaw + 相对 Yaw

            yaw_rad = world_yaw_deg * DEG2RAD # 转弧度并归一化(滤波器需要弧度制)
            raw_yaw = normalize_angle(yaw_rad)    # 弧度制归一化 -180 ~ 180

            armors.append(Armor(aid, world_pos[0], world_pos[1], world_pos[2], raw_yaw))

        self.debug_yaw_armors = debug_yaw_armors    # 用于 debug 视觉 PnP 数据
        return armors

    def try_init_tracker(self, armors):
        """
        【初始化/恢复追踪】
        逻辑：深度排序(最近优先) -> 中心优先(容差范围内) -> 重新锁定
        """
        if not armors:
            return

        sorted_armors = sorted(armors, key=lambda ar: ar.pos[2], reverse=False)     # 排序：按 Z 轴深度升序排列 (从小到大，选最近的)

        chosen_armor = sorted_armors[0]

        # 如果存在第二个目标，且两者深度差在 dist_tol 范围内，则优选 X 轴靠近中心的
        if len(sorted_armors) >= 2:
            first = sorted_armors[0]
            second = sorted_armors[1]

            if abs(first.pos[2] - second.pos[2]) <= self.dist_tol:
                # 比较 X 轴绝对值 (横向偏移)
                if abs(first.pos[0]) > abs(second.pos[0]):
                    chosen_armor = second

        # 只要进入 try_init_tracker，说明已经处于 LOST 状态。
        # 无论 ID 是否改变，都必须强制重新锁定并初始化 EKF。
        self.lock_target(chosen_armor)

    def init_ekf(self, armor):
        """
        初始化 EKF (DETECTING -> TRACKING 时调用)
        """
        xa, ya, za = armor.pos
        # 重置连续角度为当前观测角度
        self.last_yaw = armor.yaw
        yaw = self.last_yaw

        # 初始化 EKF 状态
        # 这里初始化QR参数
        self.ekf.init_QR(**self.ekf_QR_params)
        # 注意：这里我们还没有 r，先给个默认值 0.26 (步兵一般是0.2-0.3之间)
        self.ekf.init_state(xa, ya, za, yaw, robot_list[armor.id].robot_r1)

        # 初始化物理参数
        self.dz = 0.0
        self.another_r = robot_list[armor.id].robot_r2
        
        self.target_state = self.ekf.X.copy()

        # 【新增】必须将 tracker 的记忆同步为 EKF 内部最终确定的 yaw，防止下一帧计算偏差发散
        self.last_yaw = self.target_state[6]
    
        # 【修改】 print -> _log
        self._log("sys", f"[Tracker] {self.c.GREEN}Init EKF with ID {self.c.CYAN}{armor.id}{self.c.RESET}")

    def lock_target(self, armor):
        """
        【执行重置】只有在确定换ID变化时才调用
        """
        self._log("sys", f"[Tracker] {self.c.RED}ID SWITCH: {self.c.BLUE}{self.tracked_id}{self.c.RESET} {self.c.PINK}-> {self.c.CYAN}{armor.id} | RESET EKF")

        # 1. 更新 ID
        self.tracked_id = armor.id

        # 2. 重置滤波器 (初始化 QR 参数 + 初始化状态向量 + 重置协方差 P)
        self.init_ekf(armor)

        # 3. 重置状态机计数
        self.tracker_state = self.DETECTING
        self.detect_count = 1

    def get_armor_position_from_state(self, x):
        """
        【辅助函数】从 9 维状态向量推算当前追踪装甲板的预测位置
        用于和实际观测做距离匹配
        x: 9维状态向量
        """
        xc, yc, za = x[0], x[2], x[4]
        yaw = x[6]
        r = x[8]

        # 逆向推算：从车中心 -> 装甲板
        xa = xc - r * np.cos(yaw)
        ya = yc - r * np.sin(yaw)
        return np.array([xa, ya, za])

    def orientation_to_yaw(self, yaw):
        """
        【核心函数】将观测到的离散 yaw (-pi~pi) 转换为连续 yaw (-inf~inf)
        解决 "小陀螺" 旋转时的角度突变问题
        """
        # 计算当前观测 yaw 与上一次 yaw 的最短距离
        diff = shortest_angular_distance(self.last_yaw, yaw)

        # 累加得到连续角度
        continuous_yaw = self.last_yaw + diff

        # 更新记录
        self.last_yaw = continuous_yaw
        return continuous_yaw

    def handle_armor_jump(self, current_armor):
        """
        处理装甲板跳变 (Switching Armor)
        """
        yaw = self.orientation_to_yaw(current_armor.yaw)
        self.target_state[6] = yaw

        # ================== 物理状态修正开始 ==================
        raw_dz = self.target_state[4] - current_armor.pos[2]

        # 1. 钳制高度差，防止异常 PnP 导致 dz 发散 (限制在 ±15cm)
        self.dz = np.clip(raw_dz, -0.085, 0.085)
        self.target_state[4] = current_armor.pos[2]

        # 2. 斩断 Z 轴错误积分：将 Z 轴速度 (v_za) 清零
        self.target_state[5] = 0.0

        # 3. 施加水平速度阻尼：削弱跳变带来的虚假惯性
        self.target_state[1] *= 0.8  # v_xc 衰减
        self.target_state[3] *= 0.8  # v_yc 衰减
        # ================== 物理状态修正结束 ==================

        # 交换半径
        temp_r = self.target_state[8]
        self.target_state[8] = self.another_r
        self.another_r = temp_r

        self._log("jump", f"{self.c.RED}[Tracker] Armor Jump! Swap radius: {self.c.BLUE}{self.another_r:.3f} {self.c.PINK}-> {self.c.CYAN}{self.target_state[8]:.3f}, dz: {self.dz:.3f}{self.c.RESET}")

        current_p = current_armor.pos
        infer_p = self.get_armor_position_from_state(self.target_state)

        # 修正中心位置，但保留现有的速度状态
        if np.linalg.norm(current_p - infer_p) > self.max_match_distance:
            r = self.target_state[8]

            # 先用当前 yaw 试算
            test_xc = current_p[0] + r * np.cos(yaw)
            test_yc = current_p[1] + r * np.sin(yaw)

            # 模长判断
            if (current_p[0]**2 + current_p[1]**2) > (test_xc**2 + test_yc**2):
                # 如果法向量反了，同样翻转 yaw
                yaw += np.pi
                self.last_yaw = yaw
                self.target_state[6] = yaw  # 同步更新状态机里的 yaw

            self.target_state[0] = current_p[0] + r * np.cos(yaw)
            self.target_state[2] = current_p[1] + r * np.sin(yaw)

            self._log("warn", f"[Tracker] {self.c.RED}Jump Error too large, Adjusted Center Position!{self.c.RESET}")

        # 同步状态到 EKF
        self.ekf.X = self.target_state.copy()

        # ================== 协方差软重置 ==================
        self.ekf.smooth_reset_covariance()
    def update(self, armors, dt):
        """
        armors: 已经 process_armors 处理过的 Armor 对象列表 (World Frame)
        """
        # [新增] 冷却期递减
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        # 1. EKF 预测
        ekf_prediction = self.ekf.predict(dt).copy()

        # 默认目标状态为预测值
        self.target_state = ekf_prediction.copy()

        matched = False

        if len(armors) > 0:

            min_position_diff = float('inf')
            yaw_diff = float('inf')

            # 从预测状态推算当前装甲板应该在哪里
            predicted_armor_pos = self.get_armor_position_from_state(ekf_prediction)
            # 提取 EKF 预测的车体中心位置
            predicted_center_pos = np.array([ekf_prediction[0], ekf_prediction[2], ekf_prediction[4]])

            for armor in armors:
                if armor.id == self.tracked_id:

                    # 计算距离差
                    p_diff = np.linalg.norm(predicted_armor_pos - armor.pos)

                    if p_diff < min_position_diff:
                        min_position_diff = p_diff
                        # 计算 yaw_diff
                        obs_yaw_norm = normalize_angle(armor.yaw)
                        pred_yaw_norm = normalize_angle(ekf_prediction[6])
                        yaw_diff = abs(shortest_angular_distance(pred_yaw_norm, obs_yaw_norm))

                        self.tracked_armor = armor # 记录最佳匹配

            # 匹配逻辑判断
            if min_position_diff < self.max_match_distance and yaw_diff < self.max_match_yaw_diff * DEG2RAD:
                # [情况 A]: 完美匹配
                matched = True
                # 记录 EKF 观测更新的耗时
                # 更新观测向量
                # 注意 1: 必须先处理 Yaw 的连续性
                cont_yaw = self.orientation_to_yaw(self.tracked_armor.yaw)

                # 注意 2: 传入装甲板坐标给 EKF Update
                measurement = np.array([
                    self.tracked_armor.pos[0],
                    self.tracked_armor.pos[1],
                    self.tracked_armor.pos[2],
                    cont_yaw
                ])

                self.target_state = self.ekf.update(measurement)

            elif yaw_diff > self.max_match_yaw_diff * DEG2RAD:
                # [情况 B]: 角度差异过大，判定为装甲板跳变 (Armor Jump)
                
                # [新增防跳防抖] 检查是否在冷却期内
                if self.jump_cooldown > 0:
                    self._log("debug", f"[Tracker] {self.c.PINK}Jump rejected! Cooldown active: {self.jump_cooldown}{self.c.RESET}")
                    # 在冷却期内强行出现的离谱角度大概率是 PnP 野值，直接跳过本次更新（走无匹配逻辑）
                else:
                    center_diff = np.linalg.norm(predicted_center_pos - self.tracked_armor.pos)

                    if center_diff < 0.6:
                        self.handle_armor_jump(self.tracked_armor)
                        self.jump_cooldown = self.jump_cooldown_max  # [新增] 触发跳变后，立刻进入冷却
                        matched = True
                    else:
                        self._log("warn", f"[Tracker] {self.c.RED}Jump rejected! Center diff too large: {center_diff:.3f}m{self.c.RESET}")

            else:
                # [情况 C]: 没匹配上
                # 【修改】 print -> _log (使用 'debug' 标签，这是高频信息，需要在 Node 里节流)
                self._log("debug", f"[Tracker] {self.c.RED}No match! {self.c.PINK}Diff: {self.c.CYAN}{min_position_diff:.3f}{self.c.RESET}, {self.c.PINK}YawDiff: {self.c.CYAN}{yaw_diff:.3f}{self.c.RESET}")

        # 2. 限制半径范围 (防止发散成无限大或负数)
        # 步兵/英雄半径一般在 0.12m 到 0.4m 之间
        if self.target_state[8] < self.radius_params['r_min']:
            self.target_state[8] = self.radius_params['r_min']
        elif self.target_state[8] > self.radius_params['r_max']:
            self.target_state[8] = self.radius_params['r_max']

        self.target_state[1] = np.clip(self.target_state[1], -15.0, 15.0)  # v_xc
        self.target_state[3] = np.clip(self.target_state[3], -15.0, 15.0)  # v_yc
        # 地面机器人的 Z 轴运动主要是悬挂起伏和地形变化，不可能达到 15m/s
        # 限制在 [-2.0, 2.0] m/s 足以应对一般的坡道和颠簸，防止状态炸裂
        self.target_state[5] = np.clip(self.target_state[5], -2.0, 2.0)    # v_za

        # [新增] 清洗异常数值并钳制物理速度 (假设最大车速不超过 15 m/s)
        self.target_state = np.nan_to_num(self.target_state, nan=0.0, posinf=100.0, neginf=-100.0)

        # 塞回给 EKF 时，再次确保是一块干净的独立内存
        self.ekf.X = self.target_state.copy()

        # 3. 状态机流转 (State Machine)
        if self.tracker_state == self.DETECTING:
            if matched:
                self.detect_count += 1
                if self.detect_count > self.tracking_thres:
                    self.tracker_state = self.TRACKING
                    self.detect_count = 0
                    self._log("state", f"[Tracker]{self.c.GREEN} DETECTING {self.c.PINK}-> {self.c.CYAN}TRACKING {self.c.CYAN}(ID: {self.tracked_id}){self.c.RESET}")
            else:
                # 给 DETECTING 一定的容错率（例如容忍丢失次数不超过已捕获次数的一半，或者简单点，不要立刻归零）
                # 这里采用：如果没有匹配上，不立刻 LOST，而是削减置信度（扣除计数），扣到 0 以下才判定彻底失败
                self.detect_count -= 1
                if self.detect_count <= 0:
                    self.detect_count = 0
                    self.tracker_state = self.LOST

        elif self.tracker_state == self.TRACKING:
            if not matched:
                self.tracker_state = self.TEMP_LOST
                self.lost_count = 1  # 明确从 1 开始计数
                self._log("state", f"[Tracker] {self.c.GREEN}TRACKING {self.c.PINK}-> {self.c.CYAN}TEMP_LOST{self.c.RESET}")
            else:
                self.lost_count = 0  # 显式清零，保持状态干净

        elif self.tracker_state == self.TEMP_LOST:
            if not matched:
                self.lost_count += 1
                if self.lost_count > self.lost_thres:
                    self.tracker_state = self.LOST
                    self.lost_count = 0
                    self._log("state", f"[Tracker] {self.c.GREEN}TEMP_LOST {self.c.PINK}-> {self.c.CYAN}LOST{self.c.RESET}")
            else:
                self.tracker_state = self.TRACKING
                self.lost_count = 0
                self._log("state", f"[Tracker] {self.c.GREEN}TEMP_LOST {self.c.PINK}-> {self.c.CYAN}TRACKING{self.c.RESET}")

    def predict_future_state(self):
        """
        基于当前系统状态，预测子弹到达时的车体整体状态 (中心位置 + 整体偏航角)
        """
        current_state = self.target_state.copy()
        if self.bullet_speed < 1e-3:
            return current_state

        xc, yc, za = current_state[0], current_state[2], current_state[4]
        v_xc, v_yc, v_za = current_state[1], current_state[3], current_state[5]
        yaw, v_yaw = current_state[6], current_state[7]
        r = current_state[8]

        # 1. 计算到车体表面的粗略距离 (中心距离减去装甲板半径)
        center_dist = math.sqrt(xc**2 + yc**2 + za**2)
        rough_dist = max(0.1, center_dist - r) 

        # 2. 预测时间 = 飞行时间 + 系统发弹延迟 
        t = (rough_dist / self.bullet_speed) + self.system_delay

        # 3. 计算未来状态
        future_state = current_state.copy()
        future_state[0] = xc + v_xc * t
        future_state[2] = yc + v_yc * t
        future_state[4] = za + v_za * t
        future_state[6] = yaw + v_yaw * t

        return future_state

    def find_all_armors(self, state):
        """
        利用传入的状态向量(未来的整体状态)解算出 4 块装甲板的世界坐标
        """
        xc, yc, za = state[0], state[2], state[4]
        yaw = state[6]
        r1 = state[8]
        r2 = self.another_r
        dz = self.dz

        robot_armors = []
        for i in range(4):
            # 计算每块板的理论角度
            armor_yaw = yaw - i * (math.pi / 2.0)

            # 半径和高度处理
            r = r1 if (i % 2 == 0) else r2
            z = za if (i % 2 == 0) else (za + dz)

            # 计算世界坐标
            ax = xc - r * math.cos(armor_yaw)
            ay = yc - r * math.sin(armor_yaw)

            armor_obj = Armor(self.tracked_id, ax, ay, z, armor_yaw, index=i)
            robot_armors.append(armor_obj)

        return robot_armors

    def find_target(self, robot_armors, state):
        best_armor = target = None
        min_dist = float('inf')

        xc, yc = state[0], state[2]
        yaw_center_to_cam = math.atan2(-yc, -xc)

        # ---------------- 状态更新：引入退出防抖 ----------------
        if abs(self.ekf.X[7]) > self.min_spinning_vel:
            self.min_spinning_frame_count += 1
            self.spinning_frame_lost_count = 0 
            if self.min_spinning_frame_count > self.min_spinning_frame:
                self.min_spinning_frame_count = 0
                self.spin = True
        else:
            self.min_spinning_frame_count = 0 
            if self.spin: 
                self.spinning_frame_lost_count += 1
                if self.spinning_frame_lost_count > self.spinning_frame_lost:
                    self.spin = False
                    self.spinning_frame_lost_count = 0
                    
        target_to_armor_dist = np.linalg.norm(state[0:2])
        
        # ---------------- 选板决策 ----------------
        if self.spin:
            same_side = []
            for armor in robot_armors:
                norm_yaw = normalize_angle(armor.yaw)
                if self.ekf.X[7] >= 0:
                    if norm_yaw <= 0:
                        same_side.append(armor)
                else:
                    if norm_yaw > 0:
                        same_side.append(armor)

            for armor in same_side:
                dist = np.linalg.norm(armor.pos)
                yaw_center_to_armor = math.atan2(armor.pos[1] - yc, armor.pos[0] - xc)
                armor.angle_diff = abs(shortest_angular_distance(yaw_center_to_cam, yaw_center_to_armor))

                if dist < min_dist:
                    best_armor = armor
                    min_dist = dist
            
            target = best_armor
            # 【修复】增加非空判断，防止抛出 NoneType 异常
            if target is not None:
                # 直接通过 index 判断该装甲板对应的半径
                if target.index % 2 == 0:
                    r = self.ekf.X[8]
                else:
                    r = self.another_r
                # 【修复】纠正 sin 和 cos 对应关系以匹配极坐标
                target.pos[0] = xc + math.cos(yaw_center_to_cam) * r
                target.pos[1] = yc + math.sin(yaw_center_to_cam) * r
                
                target_to_armor_dist = abs(np.linalg.norm(best_armor.pos - target.pos))
                target.target_to_armor_dist = target_to_armor_dist

        else:
            sorted_armors = sorted(robot_armors, key=lambda armor: np.linalg.norm(armor.pos))
            candidate_armors = sorted_armors[:2] 
            min_angle_diff = float('inf') 

            for armor in candidate_armors:
                yaw_center_to_armor = math.atan2(armor.pos[1] - yc, armor.pos[0] - xc)
                angle_diff = abs(shortest_angular_distance(yaw_center_to_cam, yaw_center_to_armor))
                armor.angle_diff = angle_diff

                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    target = armor

        self.target = target
        return target

    def world_to_muzzle(self, tf, target, offset_pos, imu_rpy):
        """
        将目标坐标从 [相对于相机] 转换为 [相对于枪口]

        target:     Armor 对象 (target.pos 是世界系下，以相机为原点的坐标)
        offset_pos: [x, y, z] 偏移量，定义在【相机坐标系】下
                    即：从相机光心指向枪口的向量。
                    例如：枪口在相机下方 5cm => offset_pos = [0, 0.05, 0] (假设相机系Y轴向下)
        tf:         坐标转换工具
        imu_rpy:    当前云台姿态
        """
        if target is None:
            return None

        pos_target_world = target.pos

        offset_pos_world = self.cam_to_world(tf, offset_pos, imu_rpy)

        pos_gun_frame_world = pos_target_world - offset_pos_world

        muzzle_target = Armor(target.id,
                              pos_gun_frame_world[0],
                              pos_gun_frame_world[1],
                              pos_gun_frame_world[2],
                              target.yaw,
                              index=target.index)

        muzzle_target.angle_diff = target.angle_diff
        self.muzzle_target = muzzle_target

        return muzzle_target

    def solve_ballistic(self, tf, muzzle_target, cam_to_gun_rpy, imu_rpy):
        """
        基于当前位姿逆解的弹道解算 (Iterative Method)
        """
        bullet_speed = self.bullet_speed

        if muzzle_target is None or bullet_speed < 1e-3:
            return [0.0, 0.0, False]

        # 1. 提取惯性系坐标 (原点在枪口，XYZ轴与世界系平行)
        x, y, z = muzzle_target.pos
        dist_h = math.sqrt(x**2 + y**2)

        if dist_h < 0.1 or dist_h > 12.0 or math.isnan(dist_h):
            return [0.0, 0.0, False]

        g = 9.81
        v = bullet_speed
        k = 0.035

        # 2. 迭代求解补偿重力后的高度 (完全在物理惯性系下计算)
        target_pitch_rad = math.atan2(z, dist_h)
        for i in range(5):
            cos_theta = math.cos(target_pitch_rad)
            if cos_theta < 1e-4: cos_theta = 1e-4

            if k > 1e-5:
                t = (math.exp(k * dist_h) - 1) / (k * v * cos_theta)
            else:
                t = dist_h / (v * cos_theta)

            y_drop = 0.5 * g * t**2
            z_aim = z + y_drop
            target_pitch_rad = math.atan2(z_aim, dist_h)

        # 3. 构造补偿重力后的“虚拟瞄准点” (惯性系坐标)
        aim_point_world = np.array([x, y, z_aim])

        # 4. 【核心逻辑】直接调用你的辅助函数，逆解回当前的相机坐标系
        aim_point_cam = self.world_to_cam(tf, aim_point_world, imu_rpy)

        # 5. 在相机坐标系下直接求取角度增量 (Delta)
        # 相机系标准定义：Z轴向前，X轴向右，Y轴向下
        # 计算所需的偏航和俯仰增量 (可根据云台电机的正负极性在此处加负号)
        delta_yaw = math.atan2(aim_point_cam[0], aim_point_cam[2])
        delta_pitch = math.atan2(aim_point_cam[1], aim_point_cam[2])

        # 6. 加上机械安装补偿 (修正枪管与相机的固有静态偏差)
        delta_yaw += cam_to_gun_rpy[2] * DEG2RAD
        delta_pitch += cam_to_gun_rpy[1] * DEG2RAD

        return [delta_yaw, delta_pitch, True]

    def gimbal_to_deg(self, gimbal_control):
        return [gimbal_control[0] * RAD2DEG, gimbal_control[1] * RAD2DEG, False]

    def can_fire(self, target, gimbal_control):
        """
        【修改】
        can_fire: bool
        """
        yaw, pitch = gimbal_control[0], gimbal_control[1]

        dist_mix = target.dist * self.distance_decress_ratio
        yaw_mix = self.yaw_tolerance_deg * (1 - self.distance_decress_ratio)
        pitch_mix = self.pitch_tolerance_deg * (1 - self.distance_decress_ratio)

        self.yaw_tolerance_deg_mix = dist_mix + yaw_mix
        self.pitch_tolerance_deg_mix = dist_mix + pitch_mix
        if self.spin:
            gimbal_control[2] = (abs(yaw) < self.yaw_tolerance_deg_mix and abs(pitch) < self.pitch_tolerance_deg_mix and target.dist <= self.shootable_dist and target.target_to_armor_dist <= (robot_list[target.id].armor_diagonal / 2))
        else:
            gimbal_control[2] = (abs(yaw) < self.yaw_tolerance_deg_mix and abs(pitch) < self.pitch_tolerance_deg_mix and target.dist <= self.shootable_dist)

        return gimbal_control

    def track(self, tf, msg, imu_rpy, ros_clock):
        """
        【主入口】
        tf: 坐标转换工具
        msg: 相机识别到的原始数据
        imu_rpy: 欧拉角 (roll, pitch, yaw)
        Return: (fire_yaw, fire_pitch, can_fire, log_list)
        """

        # 【新增】每帧开始前清空日志 buffer
        self.log_buffer = []

        gimbal_control = [0.0, 0.0, False]

        # 1. 更新时间步长
        dt = self.update_dt(ros_clock)
        self.dt = dt

        # 2. 数据预处理 (Cam -> World)
        armors = self.process_armors(tf, msg, imu_rpy)

        # 3. 状态机调度
        if self.tracker_state == self.LOST:
            # 如果丢失，尝试初始化
            self.try_init_tracker(armors)
            # LOST 状态下返回空结果 (0, 0, False)，但带上日志
            return gimbal_control, self.log_buffer
        else:
            # 如果正在追踪 (DETECTING / TRACKING / TEMP_LOST)
            # 运行核心更新逻辑 (EKF Predict -> Match -> EKF Update)
            self.update(armors, dt)

        # 4. 检查是否由于丢失太久回到了 LOST
        if self.tracker_state == self.LOST:
            # 如果 update 导致丢失，也要返回空
            self.tracked_id = None
            return gimbal_control, self.log_buffer

        # 5. 整体状态预测与选敌
        # 先推算子弹到达时车体的整体位姿
        future_state = self.predict_future_state()
        
        # 用未来的位姿生成 4 块装甲板
        robot_armors = self.find_all_armors(future_state)

        # 【步骤A】 在未来分布的装甲板中选出最佳目标
        target_cam = self.find_target(robot_armors, future_state)

        if target_cam is not None:
            # 【步骤B】 转换到枪口系，专门用于解算弹道
            target_muzzle = self.world_to_muzzle(tf, target_cam, self.cam_to_gun_pos, imu_rpy)
            gimbal_control = self.solve_ballistic(tf, target_muzzle, self.cam_to_gun_rpy, imu_rpy)
            gimbal_control = self.gimbal_to_deg(gimbal_control)
            gimbal_control = self.can_fire(target_cam, gimbal_control)
            self.gimbal_control = gimbal_control
        else:
            # 【修复】如果没有可用目标，必须清空枪口系目标，防止可视化残留
            self.muzzle_target = None
        # 返回解算结果
        return gimbal_control, self.log_buffer

    def draw_tracking_state_with_snapshot(self, snapshot, tf, img, imu_rpy):
        # 性能优化：如果既没有处于追踪状态，也没有视觉 Yaw 观测数据，直接返回原图
        if snapshot['tracker_state'] == self.LOST and not snapshot['debug_yaw_armors']:
            return img

        draw = img.copy()
        scale = snapshot['text_size']

        # ================= 第一层：先画 Yaw 观测值 =================
        if snapshot['debug_yaw_armors']:
            font_scale = 1.0 * scale
            thickness = max(1, int(3 * scale))
            stroke_thickness = thickness + max(1, int(5 * scale))

            for armor in snapshot['debug_yaw_armors']:
                world_pos = armor.pos
                raw_yaw_rad = armor.yaw

                cam_pos = self.world_to_cam(tf, world_pos, imu_rpy)
                uv, visible = tf.project_point(cam_pos)

                if visible:
                    uv_int = (int(uv[0]), int(uv[1]))
                    yaw_deg = raw_yaw_rad
                    text = f"{yaw_deg:.1f}"

                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    text_org = (uv_int[0] - text_w // 2, uv_int[1] + text_h // 2)

                    cv2.putText(draw, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), stroke_thickness)
                    cv2.putText(draw, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)

        # ================= 第二层：再画 Estimate 预测值（覆盖在上面） =================
        if snapshot['tracker_state'] != self.LOST:
            circle_r = max(2, int(5 * scale))
            marker_size = int(20 * scale)
            thick_1 = max(1, int(1 * scale))
            thick_2 = max(1, int(2 * scale))

            x = snapshot['target_state']
            xc, yc, za = x[0], x[2], x[4]
            yaw = x[6]
            r1 = x[8]
            r2 = snapshot['another_r']
            dz = snapshot['dz']

            virtual_armors_pts = []

            for i in range(4):
                theta = yaw - i * (math.pi / 2.0)
                r = r1 if (i % 2 == 0) else r2
                z = za if (i % 2 == 0) else (za + dz)

                ax = xc - r * math.cos(theta)
                ay = yc - r * math.sin(theta)

                world_pos = np.array([ax, ay, z])
                cam_pos = self.world_to_cam(tf, world_pos, imu_rpy)
                uv, visible = tf.project_point(cam_pos)

                if visible:
                    uv_int = (int(uv[0]), int(uv[1]))
                    color = (0, 255, 0) if i == 0 else (0, 255, 255)
                    cv2.circle(draw, uv_int, circle_r, color, -1)
                    virtual_armors_pts.append(uv_int)
                else:
                    virtual_armors_pts.append(None)

            center_world = np.array([xc, yc, za + dz/2.0])
            cam_c = self.world_to_cam(tf, center_world, imu_rpy)
            uv_c, vis_c = tf.project_point(cam_c)

            if vis_c:
                c_int = (int(uv_c[0]), int(uv_c[1]))
                cv2.drawMarker(draw, c_int, (255, 255, 255), cv2.MARKER_CROSS, marker_size, thick_2)

                line_color = (100, 100, 100)
                # 【关键修复】：显式判断 is not None，防止隐式 bool 转换引发 OpenCV 连线崩溃
                if virtual_armors_pts[0] is not None and virtual_armors_pts[2] is not None:
                    cv2.line(draw, virtual_armors_pts[0], virtual_armors_pts[2], line_color, thick_1)
                if virtual_armors_pts[1] is not None and virtual_armors_pts[3] is not None:
                    cv2.line(draw, virtual_armors_pts[1], virtual_armors_pts[3], line_color, thick_1)

                vx, vy = x[1], x[3]
                speed = math.sqrt(vx**2 + vy**2)
                if speed > 0.1:
                    end_point_world = np.array([xc + vx * 0.5, yc + vy * 0.5, za])
                    cam_end = self.world_to_cam(tf, end_point_world, imu_rpy)
                    uv_end, vis_end = tf.project_point(cam_end)

                    if vis_end:
                        e_int = (int(uv_end[0]), int(uv_end[1]))
                        cv2.arrowedLine(draw, c_int, e_int, (0, 0, 255), thick_2)

        return draw

    def draw_aiming_hud_with_snapshot(self, snapshot, tf, img, imu_rpy):
        """
        绘制瞄准控制与状态信息 (HUD) 视图
        """
        draw = img.copy()
        h, w = draw.shape[:2]

        # ==================== 1. 状态校验 ====================
        if snapshot['target'] is None or snapshot['tracker_state'] == self.LOST:
            cv2.putText(draw, "SEARCHING...", (w // 2 - 80, h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            return draw

        # ==================== 2. 绘制视觉标识 ====================
        cam_tgt = self.world_to_cam(tf, snapshot['target'].pos, imu_rpy)
        uv_tgt, vis_tgt = tf.project_point(cam_tgt)
        if vis_tgt:
            target_center = (int(uv_tgt[0]), int(uv_tgt[1]))
            cv2.circle(draw, target_center, 18, (0, 0, 255), 2)

        if snapshot.get('muzzle_target') is not None:
            aim_pos_cam = self.world_to_cam(tf, snapshot['muzzle_target'].pos, imu_rpy)
            uv_aim, vis_aim = tf.project_point(aim_pos_cam)
            if vis_aim:
                aim_point = (int(uv_aim[0]), int(uv_aim[1]))
                cv2.drawMarker(draw, aim_point, (0, 255, 0), cv2.MARKER_CROSS, 25, 2)

        # ==================== 绘制粉色准星 ====================
        center_x, center_y = w // 2, h // 2
        crosshair_size = 20 
        crosshair_thickness = 2
        pink_color = (255, 0, 255) 
        cv2.line(draw, (center_x - crosshair_size, center_y), (center_x + crosshair_size, center_y), pink_color, crosshair_thickness)
        cv2.line(draw, (center_x, center_y - crosshair_size), (center_x, center_y + crosshair_size), pink_color, crosshair_thickness)

        # ==================== 3. 绘制 HUD 信息板 ====================
        scale = snapshot['text_size']
        start_x = int(20 * scale)
        base_y = int(50 * scale)
        step_y = int(30 * scale)

        font_normal = 0.7 * scale
        font_large = 0.8 * scale
        thick_normal = max(1, int(1 * scale))
        thick_bold = max(1, int(2 * scale))

        dist = snapshot['target'].dist
        angle_diff_deg = snapshot['target'].angle_diff * RAD2DEG
        yaw_tol = snapshot['yaw_tolerance_deg']
        pitch_tol = snapshot['pitch_tolerance_deg']
        
        # 【关键修复】：确保 float 转换，防止全 0 填充带来的类型崩溃
        ekf_yaw_vel = float(snapshot['ekf_yaw_vel'])
        
        gc = snapshot['gimbal_control'] if snapshot['gimbal_control'] else [0.0, 0.0, False]
        gc_yaw, gc_pitch, can_fire = float(gc[0]), float(gc[1]), bool(gc[2])

        is_spin = snapshot['spin']

        color_fire = (0, 255, 0) if can_fire else (0, 0, 255)
        text_fire = "FIRE ENABLE" if can_fire else "HOLD FIRE"
        color_spin = (0, 255, 0) if is_spin else (0, 0, 255)
        text_spin = "SPIN" if is_spin else "NORMAL"

        hud_lines = [
            (f"[{text_spin}]", font_large, color_spin, thick_bold, 5 * scale),
            (f"Yaw Vel: {ekf_yaw_vel:5.2f} rad/s", font_normal, (255, 255, 255), thick_normal, step_y * 1),
            (f"Dist   : {dist:5.2f} m", font_normal, (255, 255, 255), thick_normal, step_y * 2),
            (f"Diff   : {angle_diff_deg:5.1f} deg", font_normal, (255, 255, 255), thick_normal, step_y * 3),
            (f"Pitch  : {gc_pitch:5.2f} deg", font_normal, (255, 255, 255), thick_normal, step_y * 4),
            (f"Yaw    : {gc_yaw:5.2f} deg", font_normal, (255, 255, 255), thick_normal, step_y * 5),
            (f"P_Tol  : {pitch_tol:5.2f} deg", font_normal, (255, 255, 255), thick_normal, step_y * 6),
            (f"Y_Tol  : {yaw_tol:5.2f} deg", font_normal, (255, 255, 255), thick_normal, step_y * 7),
            (f"[{text_fire}]", font_large, color_fire, thick_bold, step_y * 8 + 5 * scale)
        ]

        for text, font_scale, color, thickness, y_offset in hud_lines:
            pos = (start_x, int(base_y + y_offset))
            cv2.putText(draw, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        return draw

    # 【关键修复】：彻底更新接口，调用上述两张新图的绘制逻辑
    def display_with_snapshot(self, snapshot, flag, tf, cv_img, imu_rpy):
        tracking_state_img = None
        aiming_hud_img = None
        
        # 强制调用新名字的函数
        if flag:
            tracking_state_img = self.draw_tracking_state_with_snapshot(snapshot, tf, cv_img, imu_rpy)
        aiming_hud_img = self.draw_aiming_hud_with_snapshot(snapshot, tf, cv_img, imu_rpy)
            
        return aiming_hud_img, tracking_state_img

    def get_render_snapshot(self):
        """提取当前状态快照，用于无锁渲染"""
        return {
            'tracker_state': self.tracker_state,
            'target_state': np.copy(self.target_state),
            'another_r': self.another_r,
            'dz': self.dz,
            # 【关键修复】使用列表推导式和 clone() 替换深拷贝
            'debug_yaw_armors': [a.clone() for a in self.debug_yaw_armors],
            'target': self.target.clone() if self.target else None,
            'muzzle_target': self.muzzle_target.clone() if self.muzzle_target else None,
            'spin': self.spin,
            'yaw_tolerance_deg': self.yaw_tolerance_deg_mix,
            'pitch_tolerance_deg': self.pitch_tolerance_deg_mix,
            'gimbal_control': list(self.gimbal_control) if self.gimbal_control else None,
            'ekf_yaw_vel': self.target_state[7],
            'text_size': self.text_size
        }