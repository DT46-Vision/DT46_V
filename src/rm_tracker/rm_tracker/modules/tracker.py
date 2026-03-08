import math
import numpy as np
from .ekf import ExtendedKalmanFilter
import cv2

# 常量定义
RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

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
    def __init__(self):
        self.PINK = "\033[38;5;218m"
        self.CYAN = "\033[96m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.BLUE = "\033[34m"
        self.RESET = "\033[0m"

class Armor:
    # 限制允许的属性，省掉 __dict__ 的内存开销
    __slots__ = ['id', 'pos', 'yaw', 'index', 'angle_diff']

    def __init__(self, id, x, y, z, yaw, index=None):
        self.id = str(id)
        self.pos = np.array([x, y, z])
        self.yaw = yaw

        # 之前在字典里乱塞的属性，现在正式给名分
        self.index = index      # 0,1,2,3 (装甲板序号)
        self.angle_diff = 0.0   # 枪口偏离目标的角度
    def clone(self):
        """
        【关键修复】手写极速克隆，替代耗时百倍的 copy.deepcopy
        """
        a = Armor(self.id, self.pos[0], self.pos[1], self.pos[2], self.yaw, self.index)
        a.angle_diff = self.angle_diff
        return a

class Tracker:
    # 状态机枚举
    LOST = 0        # 丢失
    DETECTING = 1   # 检测中
    TRACKING = 2    # 跟踪中
    TEMP_LOST = 3   # 短暂丢失

    def __init__(self):
        self.c = ColorPrint()
        self.dt = 0.01
        self.target_color = 0                               # 目标敌方阵营 (0: RED, 1: BLUE)
        self.cam_to_gun_pos = np.array([0.0, -0.05, 0.0])   # [外参] 相机相对于枪口的平移向量 (x:右, y:下, z:前)
        self.cam_to_gun_rpy = np.array([0.0, 0.0, 0.0])     # [外参] 相机相对于枪口的旋转欧拉角 (Roll, Pitch, Yaw)
        self.ekf = ExtendedKalmanFilter()                   # 扩展卡尔曼滤波器 (EKF) 核心实例
        self.last_time = None                               # 上一帧的时间戳，用于计算帧间时间差 dt
        self.dist_tol = 0.15                                # [预处理] 同一装甲板不同观测点的距离容差 (防止误判)
        self.max_match_distance = 0.2                       # [匹配] EKF 预测值与观测值的最大欧氏距离阈值 (m)
        self.max_match_yaw_diff = 1.0                       # [匹配] 判定装甲板切换 (Armor Jump) 的 Yaw 角度差阈值 (rad)
        self.tracking_thres = 5                             # 进入 TRACKING 状态所需的连续检测帧数
        self.lost_thres = 10                                # 进入 LOST 状态所需的连续丢失帧数
        self.tracker_state = self.LOST                      # 跟踪器 FSM 当前状态 (LOST/DETECTING/TRACKING/TEMP_LOST)
        self.detect_count = 0                               # 连续检测帧计数器 (用于状态确认)
        self.lost_count = 0                                 # 连续丢失帧计数器 (用于状态确认)
        self.tracked_id = None                              # 当前锁定追踪的装甲板 ID
        self.tracked_armor = None                           # 上一帧匹配成功的装甲板对象 (用于时序关联)
        self.debug_yaw_armors = []                          # 当前帧所有有效的装甲板观测数据
        self.target_state = np.zeros(9)                     # EKF 预测的 9 维系统状态向量
        self.ekf_QR_params = None                           # EKF QR 参数
        self.radius_params = None                           # 旋转半径参数
        self.last_yaw = 0.0                                 # 上一帧的连续化 Yaw 角 (用于处理角度跳变与去卷绕)
        self.dz = 0.0                                       # [几何] 当前板与另一组板的高度差 (z轴偏移)
        self.another_r = 0.26                               # [几何] 另一组装甲板的旋转半径
        self.spin = False                                   # [几何] 小陀螺旋转标准位
        self.min_spinning_frame = 10                        # 小陀螺旋转的最小帧数
        self.spinning_frame_lost = 5                        # 小陀螺帧数丢失阈值
        self.min_spinning_frame_count = 0                   # 小陀螺帧数计数器
        self.spinning_frame_lost_count = 0                  # 小陀螺帧数丢失计数器
        self.min_spinning_vel = 5.0                         # 小陀螺最低门限
        self.target = None                                  # 最终解算出的目标状态
        self.bullet_speed = 28.0                            # 弹道速度 (m/s)
        self.muzzle_target = None                           # 弹道解算后的目标点
        self.yaw_threshold_deg = 5.0                        # 允许发射的 yaw 角度阈值
        self.pitch_threshold_deg = 2.0                      # 允许发射的 pitch 角度阈值
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

        # 【删除原有的平滑逻辑】
        # if hasattr(self, 'prev_dt'):
        #     dt = 0.8 * dt + 0.2 * self.prev_dt
        # self.prev_dt = dt

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
        self.ekf.init_state(xa, ya, za, yaw, self.radius_params['r1'])

        # 初始化物理参数
        self.dz = 0.0
        self.another_r = self.radius_params['r2']

        self.target_state = self.ekf.X
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
                yaw = self.orientation_to_yaw(current_armor.yaw + np.pi)
                self.target_state[6] = yaw  # 同步更新状态机里的 yaw

            self.target_state[0] = current_p[0] + r * np.cos(yaw)
            self.target_state[2] = current_p[1] + r * np.sin(yaw)

            self._log("warn", f"[Tracker] {self.c.RED}Jump Error too large, Adjusted Center Position!{self.c.RESET}")

        # 同步状态到 EKF
        self.ekf.X = self.target_state

        # ================== 协方差软重置 ==================
        # 适度放大位置和角度的方差，让滤波器在跳变后几帧稍微更信任观测
        self.ekf.P[0, 0] += 0.05  # xc 的方差轻微放大
        self.ekf.P[2, 2] += 0.05  # yc 的方差轻微放大
        self.ekf.P[4, 4] += 0.05  # za 的方差轻微放大
        self.ekf.P[6, 6] += 0.2   # yaw 角度的方差适度放大

        # [新增] 给半径的方差稍微松个绑，让它能适应新板子的物理误差
        self.ekf.P[8, 8] += 0.01
    def update(self, armors, dt):
        """
        armors: 已经 process_armors 处理过的 Armor 对象列表 (World Frame)
        """
        # 1. EKF 预测
        ekf_prediction = self.ekf.predict(dt)

        # 默认目标状态为预测值
        self.target_state = ekf_prediction

        matched = False

        if len(armors) > 0:
            # # 寻找同 ID 且距离最近的装甲板
            # same_id_armor = None
            # same_id_count = 0

            min_position_diff = float('inf')
            yaw_diff = float('inf')

            # 从预测状态推算当前装甲板应该在哪里
            predicted_armor_pos = self.get_armor_position_from_state(ekf_prediction)
            # 提取 EKF 预测的车体中心位置
            predicted_center_pos = np.array([ekf_prediction[0], ekf_prediction[2], ekf_prediction[4]])

            for armor in armors:
                if armor.id == self.tracked_id:
                    # same_id_armor = armor
                    # same_id_count += 1

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

            # elif same_id_count == 1 and yaw_diff > self.max_match_yaw_diff * DEG2RAD: # 判定为：车转过去了，这是新的一块板子
                # # 记录跳变处理的耗时
                # self.handle_armor_jump(same_id_armor)
                # matched = True # Jump 之后认为匹配成功，但不需要再次 update EKF（因为 handle 里已经重置了）
            elif yaw_diff > self.max_match_yaw_diff * DEG2RAD:
                # [情况 B]: 角度差异过大，判定为装甲板跳变 (Armor Jump)
                # 增加防暴走保护：计算新装甲板与车体中心的距离
                center_diff = np.linalg.norm(predicted_center_pos - self.tracked_armor.pos)

                # 正常步兵/英雄半径在 0.2~0.3m，考虑到运动学误差，0.6m 是一个安全的物理极限
                if center_diff < 0.6:
                    self.handle_armor_jump(self.tracked_armor)
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
            self.ekf.X[8] = self.radius_params['r_min']
        elif self.target_state[8] > self.radius_params['r_max']:
            self.target_state[8] = self.radius_params['r_max']
            self.ekf.X[8] = self.radius_params['r_max']
        # [新增] 清洗异常数值并钳制物理速度 (假设最大车速不超过 15 m/s)
        self.target_state = np.nan_to_num(self.target_state, nan=0.0, posinf=100.0, neginf=-100.0)
        self.target_state[1] = np.clip(self.target_state[1], -15.0, 15.0)  # v_xc
        self.target_state[3] = np.clip(self.target_state[3], -15.0, 15.0)  # v_yc

        # 地面机器人的 Z 轴运动主要是悬挂起伏和地形变化，不可能达到 15m/s
        # 限制在 [-2.0, 2.0] m/s 足以应对一般的坡道和颠簸，防止状态炸裂
        self.target_state[5] = np.clip(self.target_state[5], -2.0, 2.0)    # v_za
        # ============================================
        self.ekf.X = self.target_state

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
    def find_all_armors(self):
        """
        利用 EKF 状态解算出当前时刻 4 块装甲板的世界坐标
        返回: List[Armor]
        """
        x = self.target_state
        xc, yc, za = x[0], x[2], x[4]
        yaw = x[6]
        r1 = x[8]
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

    def find_target(self, robot_armors):
        """
        选出最佳目标 (Armor 对象)
        """
        best_armor = None
        min_dist = float('inf')

        xc, yc = self.target_state[0], self.target_state[2]
        yaw_center_to_cam = math.atan2(-yc, -xc)

        # ---------------- 状态更新：引入退出防抖 ----------------
        if abs(self.ekf.X[7]) > self.min_spinning_vel:
            self.min_spinning_frame_count += 1
            self.spinning_frame_lost_count = 0  # 角速度达标，清空丢失计数
            if self.min_spinning_frame_count > self.min_spinning_frame:
                self.min_spinning_frame_count = 0
                self.spin = True
        else:
            self.min_spinning_frame_count = 0  # 角速度未达标，清空进入计数
            if self.spin:  # 如果当前在小陀螺状态，开始累计丢失帧
                self.spinning_frame_lost_count += 1
                if self.spinning_frame_lost_count > self.spinning_frame_lost:
                    self.spin = False
                    self.spinning_frame_lost_count = 0

        # ---------------- 选板决策 ----------------
        if self.spin:
            # 小陀螺模式：“打半边”策略
            same_side = []
            for armor in robot_armors:
                if self.ekf.X[7] > 0:
                    if armor.yaw > 0:
                        same_side.append(armor)
                else:
                    if armor.yaw < 0:
                        same_side.append(armor)

            for armor in same_side:
                dist = np.linalg.norm(armor.pos)
                yaw_center_to_armor = math.atan2(armor.pos[1] - yc, armor.pos[0] - xc)
                armor.angle_diff = abs(shortest_angular_distance(yaw_center_to_cam, yaw_center_to_armor))

                if dist < min_dist:
                    best_armor = armor
                    min_dist = dist

        else:
            # 非小陀螺模式：候选前二，选夹角最小（最正对）
            sorted_armors = sorted(robot_armors, key=lambda armor: np.linalg.norm(armor.pos))
            candidate_armors = sorted_armors[:2] # 候选前两块装甲板

            min_angle_diff = float('inf') # 用于筛选最正对的板子

            for armor in candidate_armors:
                yaw_center_to_armor = math.atan2(armor.pos[1] - yc, armor.pos[0] - xc)
                angle_diff = abs(shortest_angular_distance(yaw_center_to_cam, yaw_center_to_armor))
                armor.angle_diff = angle_diff

                # 比较角度差，越小越正对
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    best_armor = armor

        self.target = best_armor

        return best_armor

    def world_to_muzzle(self, target, offset_pos, tf, imu_rpy):
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

    def solve_ballistic(self, muzzle_target, cam_to_gun_rpy, imu_rpy):
        """
        全向弹道解算 (Iterative Method)

        Args:
            muzzle_target:  Armor对象 (世界坐标，原点在枪口)
            cam_to_gun_rpy: [roll, pitch, yaw] 机械安装误差/静态补偿 (单位: 度)
            imu_rpy:        [roll, pitch, yaw] 当前云台姿态 (单位: 度)
        """
        # 使用 self.bullet_speed 读取类成员变量，避免参数传错
        bullet_speed = self.bullet_speed

        if muzzle_target is None or bullet_speed < 1e-3:
            return [0.0, 0.0, False]

        # 1. 提取坐标
        x, y, z = muzzle_target.pos
        dist_h = math.sqrt(x**2 + y**2) # 水平距离

        # 增加防爆护盾：距离太近、超过 25 米、或是 NaN 时直接放弃解算
        if dist_h < 0.1 or dist_h > 12.0 or math.isnan(dist_h):
            return [0.0, 0.0, False]

        if dist_h < 0.1:
            return [0.0, 0.0, False]

        # 2. 物理参数
        g = 9.81
        v = bullet_speed
        k = 0.035  # 空气阻力系数

        # 3. 迭代求解【物理】绝对 Pitch 角度
        # 目标是找到一个 theta，使得子弹的抛物线经过 (dist_h, z)

        # 初始猜测
        target_pitch_rad = math.atan2(z, dist_h)

        # 迭代求解 (补偿重力下坠 + 空气阻力)
        for i in range(5):
            cos_theta = math.cos(target_pitch_rad)
            if cos_theta < 1e-4: cos_theta = 1e-4

            # 计算飞行时间 t
            if k > 1e-5:
                t = (math.exp(k * dist_h) - 1) / (k * v * cos_theta)
            else:
                t = dist_h / (v * cos_theta)

            # 计算下坠补偿量
            y_drop = 0.5 * g * t**2

            # 更新目标角度
            z_aim = z + y_drop
            target_pitch_rad = math.atan2(z_aim, dist_h)

        # 4. 计算 Yaw (几何角度)
        target_yaw_rad = math.atan2(y, x)

        # 5. 加上【机械补偿】 (cam_to_gun_rpy)
        # 这一步不能少，用于修正枪管和相机的固有偏差
        offset_pitch_rad = cam_to_gun_rpy[1] * DEG2RAD
        offset_yaw_rad   = cam_to_gun_rpy[2] * DEG2RAD

        final_target_pitch = target_pitch_rad + offset_pitch_rad
        final_target_yaw   = target_yaw_rad + offset_yaw_rad

        # 6. 计算电控控制增量 (Delta)
        current_pitch_rad = imu_rpy[1] * DEG2RAD
        current_yaw_rad   = imu_rpy[2] * DEG2RAD

        # 【Yaw】 使用最短路径差值
        delta_yaw = shortest_angular_distance(current_yaw_rad, final_target_yaw)

        # 【Pitch】 根据你的实测，你的 IMU 定义是反的，所以这里用加法
        # (即: 目标角度 + 当前负角度 = 偏差)
        delta_pitch = final_target_pitch + current_pitch_rad

        return [delta_yaw, delta_pitch, True]

    def gimbal_to_deg(self, gimbal_control):
        return [gimbal_control[0] * RAD2DEG, gimbal_control[1] * RAD2DEG, False]

    def can_fire(self, gimbal_control):
        """
        【修改】
        can_fire: bool
        """
        yaw, pitch = gimbal_control[0], gimbal_control[1]
        if abs(yaw) < self.yaw_threshold_deg and abs(pitch) < self.pitch_threshold_deg:
            gimbal_control[2] = True
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
            return [0.0, 0.0, False], self.log_buffer
        else:
            # 如果正在追踪 (DETECTING / TRACKING / TEMP_LOST)
            # 运行核心更新逻辑 (EKF Predict -> Match -> EKF Update)
            self.update(armors, dt)

        # 4. 检查是否由于丢失太久回到了 LOST
        if self.tracker_state == self.LOST:
            # 如果 update 导致丢失，也要返回空
            self.tracked_id = None
            return [0.0, 0.0, False], self.log_buffer

        # 5. 生成虚拟装甲板并选敌
        robot_armors = self.find_all_armors()

        # 【步骤A】 找出最佳目标（这是相机系坐标！）
        target_cam = self.find_target(robot_armors)

        # =================【核心：只打半边/开火窗口逻辑】=================
        gimbal_control = [0.0, 0.0, False]

        if target_cam is not None:
            # 【步骤B】 转换到枪口系，专门用于解算弹道
            target_muzzle = self.world_to_muzzle(target_cam, self.cam_to_gun_pos, tf, imu_rpy)
            gimbal_control = self.solve_ballistic(target_muzzle, self.cam_to_gun_rpy, imu_rpy)
            gimbal_control = self.gimbal_to_deg(gimbal_control)
            gimbal_control = self.can_fire(gimbal_control)
            self.gimbal_control = gimbal_control

        # 返回解算结果
        return gimbal_control, self.log_buffer

    def draw_estimate_with_snapshot(self, snapshot, tf, img, imu_rpy):
        """
        绘制 EKF 预测的车辆底盘和装甲板位置。
        """
        # 【修正】self.tracker_state -> snapshot['tracker_state']
        if snapshot['tracker_state'] == self.LOST:
            return img

        draw = img.copy()
        scale = snapshot['text_size']

        circle_r = max(2, int(5 * scale))
        marker_size = int(20 * scale)
        thick_1 = max(1, int(1 * scale))
        thick_2 = max(1, int(2 * scale))

        x = snapshot['target_state']
        xc, yc, za = x[0], x[2], x[4]
        yaw = x[6]
        r1 = x[8]
        # 【修正】self.another_r -> snapshot['another_r']
        # 【修正】self.dz -> snapshot['dz']
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
            if virtual_armors_pts[0] and virtual_armors_pts[2]:
                cv2.line(draw, virtual_armors_pts[0], virtual_armors_pts[2], line_color, thick_1)
            if virtual_armors_pts[1] and virtual_armors_pts[3]:
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

    def draw_observation_yaw_with_snapshot(self, snapshot, tf, img, imu_rpy):
        """
        绘制 process_armors 中提取的观测装甲板的 Yaw 值
        """
        # 【修正】self.debug_yaw_armors -> snapshot['debug_yaw_armors']
        if not snapshot['debug_yaw_armors']:
            return img

        draw = img.copy()
        scale = snapshot['text_size']

        font_scale = 1.0 * scale
        thickness = max(1, int(3 * scale))
        stroke_thickness = thickness + max(1, int(5 * scale))

        # 【修正】self.debug_yaw_armors -> snapshot['debug_yaw_armors']
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

        return draw

    # 【修正】补上了 snapshot 参数
    def draw_ballistic_with_snapshot(self, snapshot, tf, img, imu_rpy):
        """
        绘制弹道解算视图
        """
        draw = img.copy()
        h, w = draw.shape[:2]

        # 【修正】self.target -> snapshot['target'] 和 snapshot['tracker_state']
        if snapshot['target'] is None or snapshot['tracker_state'] == self.LOST:
            cv2.putText(draw, "SEARCHING...", (w//2 - 80, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            return draw

        # 【修正】self.target.pos -> snapshot['target'].pos
        tgt_pos_world = snapshot['target'].pos

        cam_tgt = self.world_to_cam(tf, tgt_pos_world, imu_rpy)
        uv_tgt, vis_tgt = tf.project_point(cam_tgt)

        if vis_tgt:
            target_center = (int(uv_tgt[0]), int(uv_tgt[1]))
            cv2.circle(draw, target_center, 8, (0, 0, 255), -1)
            cv2.circle(draw, target_center, 18, (0, 0, 255), 2)

        # 【修正】self.muzzle_target.pos -> snapshot['muzzle_target'].pos
        if snapshot.get('muzzle_target') is not None:
            aim_pos_world = snapshot['muzzle_target'].pos
            aim_pos_cam = self.world_to_cam(tf, aim_pos_world, imu_rpy)

            uv_aim, vis_aim = tf.project_point(aim_pos_cam)
            if vis_aim:
                aim_point = (int(uv_aim[0]), int(uv_aim[1]))
                cv2.drawMarker(draw, aim_point, (0, 255, 0), cv2.MARKER_CROSS, 25, 2)

        # 【修正】self.text_size -> snapshot['text_size']
        scale = snapshot['text_size']
        start_x_scaled = int(20 * scale)
        base_y_scaled = int(50 * scale)
        step_scaled = int(30 * scale)

        font_07 = 0.7 * scale
        font_08 = 0.8 * scale

        thick_1 = max(1, int(1 * scale))
        thick_2 = max(1, int(2 * scale))

        dist = np.linalg.norm(tgt_pos_world)
        # 【修正】self.target.angle_diff -> snapshot['target'].angle_diff
        angle_diff_deg = snapshot['target'].angle_diff * RAD2DEG

        # 【修正】self.gimbal_control 和 self.spin 处理
        gc = snapshot['gimbal_control']
        if gc is None:
            gc = [0.0, 0.0, False]

        status_color = (0, 255, 0) if gc[2] else (0, 0, 255)
        status_text = "FIRE ENABLE" if gc[2] else "HOLD FIRE"
        spin_status = "SPIN" if snapshot['spin'] else "NORMAL"
        spin_status_color = (0, 255, 0) if snapshot['spin'] else (0, 0, 255)

        # 【修正】self.ekf.X[7] -> snapshot['ekf_yaw_vel']
        info_lines = [
            (f"[{spin_status}]", font_08, spin_status_color, thick_2, 5 * scale),
            (f"yaw_val: {snapshot['ekf_yaw_vel']:.2f} rad", font_07, (255, 255, 255), thick_1, step_scaled * 1),
            (f"Dist : {dist:.2f} m", font_07, (255, 255, 255), thick_1, step_scaled * 2),
            (f"Pitch: {gc[1]:.2f} deg", font_07, (255, 255, 255), thick_1, step_scaled * 3),
            (f"Yaw  : {gc[0]:.2f} deg", font_07, (255, 255, 255), thick_1, step_scaled * 4),
            (f"Diff : {angle_diff_deg:.1f} deg", font_07, (255, 255, 255), thick_1, step_scaled * 5),
            (f"[{status_text}]", font_08, status_color, thick_2, step_scaled * 6 + 5 * scale)
        ]

        for text, font_scale, color, thickness, y_offset in info_lines:
            pos = (start_x_scaled, int(base_y_scaled + y_offset))
            cv2.putText(draw, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        return draw

    def display_with_snapshot(self, snapshot, flag, tf, cv_img, imu_rpy):
        estimate_img = None
        yaw_debug_img = None
        ballistic_img = self.draw_ballistic_with_snapshot(snapshot,tf, cv_img, imu_rpy)
        if flag:
            estimate_img = self.draw_estimate_with_snapshot(snapshot,tf, cv_img, imu_rpy)
            yaw_debug_img = self.draw_observation_yaw_with_snapshot(snapshot,tf, cv_img, imu_rpy)
        return ballistic_img, estimate_img, yaw_debug_img

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
            # 【关键修复】列表的浅拷贝非常快，无需 deepcopy
            'gimbal_control': list(self.gimbal_control) if self.gimbal_control else None,
            'ekf_yaw_vel': self.ekf.X[7] if self.ekf else 0.0,
            'text_size': self.text_size
        }
