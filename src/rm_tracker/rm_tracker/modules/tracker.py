import math
import numpy as np
import time
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
    __slots__ = ['id', 'pos', 'yaw', 'index', 'angle_diff', 'cost']

    def __init__(self, id, x, y, z, yaw, index=None):
        self.id = str(id)
        self.pos = np.array([x, y, z])
        self.yaw = yaw

        # 之前在字典里乱塞的属性，现在正式给名分
        self.index = index      # 0,1,2,3 (装甲板序号)
        self.angle_diff = 0.0   # 枪口偏离目标的角度
        self.cost = 0.0         # 选板代价

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
        self.dist_tol = 0.05                                # [预处理] 同一装甲板不同观测点的距离容差 (防止误判)
        self.max_match_distance = 0.2                       # [匹配] EKF 预测值与观测值的最大欧氏距离阈值 (m)
        self.max_match_yaw_diff = 1.0                         # [匹配] 判定装甲板切换 (Armor Jump) 的 Yaw 角度差阈值 (rad)
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
        self.min_spinning_speed_tol = 0.05                  # 小陀螺最低门限
        self.target = None                                  # 最终解算出的目标状态
        self.bullet_speed = 28.0                            # 弹道速度 (m/s)
        self.muzzle_target = None                           # 弹道解算后的目标点
        self.yaw_threshold_deg = 5.0                        # 允许发射的 yaw 角度阈值
        self.pitch_threshold_deg = 2.0                      # 允许发射的 pitch 角度阈值
        self.gimbal_control = None                          # 云台控制信息
        self.log_buffer = []                                # 系统日志缓存队列 (用于调试/可视化)

    def _log(self, log_type, msg):
        """
        log_type: 用于在 Node 中决定是否节流，例如 "state", "jump", "debug"
        msg: 具体内容
        """
        self.log_buffer.append((log_type, msg))

    def update_dt(self, current_ros_time):
        """
        current_ros_time: 传入 node.get_clock().now()
        """
        if self.last_time is None:
            self.last_time = current_ros_time
            return 0.01

        # 计算纳秒差并转为秒
        duration = (current_ros_time - self.last_time).nanoseconds / 1e9

        # 钳制异常值
        dt = max(0.001, min(duration, 0.1))

        # 指数平滑滤波，减少 dt 抖动对 EKF 的影响
        # dt = alpha * new_dt + (1 - alpha) * last_dt
        if hasattr(self, 'prev_dt'):
            dt = 0.8 * dt + 0.2 * self.prev_dt

        self.prev_dt = dt
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

            yaw_rad = math.radians(world_yaw_deg) # 转弧度并归一化(滤波器需要弧度制)
            raw_yaw = normalize_angle(yaw_rad)    # 弧度制归一化 -180 ~ 180

            armors.append(Armor(aid, world_pos[0], world_pos[1], world_pos[2], raw_yaw))

        self.debug_yaw_armors = debug_yaw_armors    # 用于 debug 视觉 PnP 数据
        return armors

    def try_init_tracker(self, armors):
        """
        【初始化/恢复追踪】
        逻辑：深度排序(最近优先) -> 中心优先(容差范围内) -> 状态判定
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

        # 情况 A: ID 变化 (新目标或首次锁定) -> 重置锁定
        if self.tracked_id != chosen_armor.id:
            self.lock_target(chosen_armor)

        # 情况 B: ID 不变 (短暂丢失后找回) -> 恢复追踪
        else:
            self.tracker_state = self.DETECTING
            self._log("sys", f"[Tracker] {self.c.GREEN}Resume tracking ID: {self.c.CYAN}{self.tracked_id}{self.c.RESET}")

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
        【难点】处理装甲板跳变 (Switching Armor)
        当检测到同 ID 装甲板，但 Yaw 角发生巨大突变时触发
        """

        yaw = self.orientation_to_yaw(current_armor.yaw) # 把角度差加上去，这样角度就正常了且连续了
        self.target_state[6] = yaw                       # 强制更新 Yaw (重置连续角度逻辑)

        self.dz = self.target_state[4] - current_armor.pos[2]        # 更新 dz: 旧板预测高度 - 新板观测高度 (假设车平动 z 轴不变，z 的突变完全是 dz 引起的)
        self.target_state[4] = current_armor.pos[2] # 强制把 EKF 的高度拉到新板子上

        # 交换半径
        temp_r = self.target_state[8]
        self.target_state[8] = self.another_r
        self.another_r = temp_r
        # 【修改】 print -> _log
        self._log("jump", f"{self.c.RED}[Tracker] Armor Jump! Swap radius: {self.c.BLUE}{self.another_r:.3f} {self.c.PINK}-> {self.c.CYAN}{self.target_state[8]:.3f}, dz: {self.dz:.3f}{self.c.RESET}")

        # EKF 安全检查
        current_p = current_armor.pos        # 推算新的车中心
        infer_p = self.get_armor_position_from_state(self.target_state)

        if np.linalg.norm(current_p - infer_p) > self.max_match_distance: # 如果跳变后位置偏差过大，说明 EKF 已经发散了，需要重置车中心位置
            r = self.target_state[8]
            self.target_state[0] = current_p[0] + r * np.cos(yaw) # xc
            self.target_state[1] = 0                              # v_xc
            self.target_state[2] = current_p[1] + r * np.sin(yaw) # yc
            self.target_state[3] = 0                              # v_yc
            self.target_state[4] = current_p[2]                   # za
            self.target_state[5] = 0                              # v_za
            # 【修改】 print -> _log
            self._log("warn", f"[Tracker] {self.c.RED}Jump Error too large, Reset State!{self.c.RESET}")

        # 将修改后的状态写回 EKF
        self.ekf.X = self.target_state
        self.ekf.P = np.eye(9) # 重置协方差，防止跳变瞬间的方差爆炸影响收敛

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
            # 寻找同 ID 且距离最近的装甲板
            same_id_armor = None
            same_id_count = 0

            min_position_diff = float('inf')
            yaw_diff = float('inf')

            # 从预测状态推算当前装甲板应该在哪里
            predicted_armor_pos = self.get_armor_position_from_state(ekf_prediction)

            for armor in armors:
                if armor.id == self.tracked_id:
                    same_id_armor = armor
                    same_id_count += 1

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
            if min_position_diff < self.max_match_distance and yaw_diff < math.radians(self.max_match_yaw_diff):
                # [情况 A]: 完美匹配
                matched = True

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

            elif same_id_count == 1 and yaw_diff > math.radians(self.max_match_yaw_diff): # 判定为：车转过去了，这是新的一块板子

                self.handle_armor_jump(same_id_armor)
                matched = True # Jump 之后认为匹配成功，但不需要再次 update EKF（因为 handle 里已经重置了）

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

        # 3. 状态机流转 (State Machine)
        if self.tracker_state == self.DETECTING:
            if matched:
                self.detect_count += 1
                if self.detect_count > self.tracking_thres:
                    self.tracker_state = self.TRACKING
                    self.detect_count = 0
                    # 【修改】 print -> _log
                    self._log("state", f"[Tracker]{self.c.GREEN} DETECTING {self.c.PINK}-> {self.c.CYAN}TRACKING {self.c.CYAN}(ID: {self.tracked_id}){self.c.RESET}")
            else:
                self.detect_count = 0
                self.tracker_state = self.LOST

        elif self.tracker_state == self.TRACKING:
            if not matched:
                self.tracker_state = self.TEMP_LOST
                self.lost_count += 1
                # 【修改】 print -> _log
                self._log("state", f"[Tracker] {self.c.GREEN}TRACKING {self.c.PINK}-> {self.c.CYAN}TEMP_LOST{self.c.RESET}")

        elif self.tracker_state == self.TEMP_LOST:
            if not matched:
                self.lost_count += 1
                if self.lost_count > self.lost_thres:
                    self.tracker_state = self.LOST
                    self.lost_count = 0
                    # 【修改】 print -> _log
                    self._log("state", f"[Tracker] {self.c.GREEN}TEMP_LOST {self.c.PINK}-> {self.c.CYAN}LOST{self.c.RESET}")
            else:
                self.tracker_state = self.TRACKING
                self.lost_count = 0
                # 【修改】 print -> _log
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
        min_cost = float('inf')

        xc, yc = self.target_state[0], self.target_state[2]
        yaw_center_to_cam = math.atan2(-yc, -xc)

        for armor in robot_armors:
            # 【修改】直接访问属性
            dist = np.linalg.norm(armor.pos)
            cost = dist

            # 算出板子相对于车中心的角度
            yaw_center_to_armor = math.atan2(armor.pos[1] - yc, armor.pos[0] - xc)

            # 算出夹角
            angle_diff = abs(shortest_angular_distance(yaw_center_to_cam, yaw_center_to_armor))

            # 【修改】直接写入属性，不再是字典赋值
            armor.angle_diff = angle_diff
            armor.cost = cost

            if cost < min_cost:
                min_cost = cost
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
        offset_pitch_rad = math.radians(cam_to_gun_rpy[1])
        offset_yaw_rad   = math.radians(cam_to_gun_rpy[2])

        final_target_pitch = target_pitch_rad + offset_pitch_rad
        final_target_yaw   = target_yaw_rad + offset_yaw_rad

        # 6. 计算电控控制增量 (Delta)
        current_pitch_rad = math.radians(imu_rpy[1])
        current_yaw_rad   = math.radians(imu_rpy[2])

        # 【Yaw】 使用最短路径差值
        delta_yaw = shortest_angular_distance(current_yaw_rad, final_target_yaw)

        # 【Pitch】 根据你的实测，你的 IMU 定义是反的，所以这里用加法
        # (即: 目标角度 + 当前负角度 = 偏差)
        delta_pitch = final_target_pitch + current_pitch_rad

        return [delta_yaw, delta_pitch, True]

    def gimbal_to_deg(self, gimbal_control):
        return [math.degrees(gimbal_control[0]), math.degrees(gimbal_control[1]), False]

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

    def draw_estimate(self, tf, img, imu_rpy):
        """
        绘制 EKF 预测的车辆底盘和装甲板位置。
        【修改】移除了 Target 高亮逻辑，仅展示估计的物理结构。
        """
        if self.tracker_state == self.LOST:
            return img
        draw = img.copy()

        # 获取当前 EKF 状态
        x = self.target_state
        xc, yc, za = x[0], x[2], x[4]
        yaw = x[6]
        r1 = x[8]
        r2 = self.another_r
        dz = self.dz

        # 准备绘制数据，用于画连线
        virtual_armors_pts = []

        for i in range(4):
            # --- 1. 计算虚拟位置 ---
            theta = yaw - i * (math.pi / 2.0)
            r = r1 if (i % 2 == 0) else r2
            z = za if (i % 2 == 0) else (za + dz)

            ax = xc - r * math.cos(theta)
            ay = yc - r * math.sin(theta)

            world_pos = np.array([ax, ay, z])
            cam_pos = self.world_to_cam(tf, world_pos, imu_rpy)
            uv, visible = tf.project_point(cam_pos)

            # --- 2. 绘制装甲板 (仅作为结构展示) ---
            if visible:
                uv_int = (int(uv[0]), int(uv[1]))

                # 统一使用黄色，仅用绿色区分 0 号板(正面)
                color = (0, 255, 255)  # Yellow
                if i == 0:
                    color = (0, 255, 0) # Green

                # 绘制实心圆点
                cv2.circle(draw, uv_int, 5, color, -1)
                # 标序号
                cv2.putText(draw, f"{i}", (uv_int[0] + 5, uv_int[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                virtual_armors_pts.append(uv_int)
            else:
                virtual_armors_pts.append(None)

        # --- 3. 绘制中心和连线 ---
        center_world = np.array([xc, yc, za + dz/2.0])
        cam_c = self.world_to_cam(tf, center_world, imu_rpy)
        uv_c, vis_c = tf.project_point(cam_c)

        if vis_c:
            c_int = (int(uv_c[0]), int(uv_c[1]))
            # 绘制车中心十字
            cv2.drawMarker(draw, c_int, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

            # 绘制连线 (展示底盘结构)
            line_color = (100, 100, 100)
            if virtual_armors_pts[0] and virtual_armors_pts[2]:
                cv2.line(draw, virtual_armors_pts[0], virtual_armors_pts[2], line_color, 1)
            if virtual_armors_pts[1] and virtual_armors_pts[3]:
                cv2.line(draw, virtual_armors_pts[1], virtual_armors_pts[3], line_color, 1)

            # --- 4. 绘制速度矢量 ---
            vx, vy = x[1], x[3]
            speed = math.sqrt(vx**2 + vy**2)
            if speed > 0.1:
                end_point_world = np.array([xc + vx * 0.5, yc + vy * 0.5, za])
                cam_end = self.world_to_cam(tf, end_point_world, imu_rpy)
                uv_end, vis_end = tf.project_point(cam_end)

                if vis_end:
                    e_int = (int(uv_end[0]), int(uv_end[1]))
                    cv2.arrowedLine(draw, c_int, e_int, (0, 0, 255), 2)

        return draw

    def draw_observation_yaw(self, tf, img, imu_rpy):
        """
        绘制 process_armors 中提取的观测装甲板的 Yaw 值
        """
        if not self.debug_yaw_armors:
            return img
        draw = img.copy()
        for armor in self.debug_yaw_armors:
            # 1. 获取观测数据 (已经是世界坐标系)
            world_pos = armor.pos
            raw_yaw_rad = armor.yaw

            # 2. 转换回像素坐标
            cam_pos = self.world_to_cam(tf, world_pos, imu_rpy)
            uv, visible = tf.project_point(cam_pos)

            if visible:
                uv_int = (int(uv[0]), int(uv[1]))

                yaw_deg = raw_yaw_rad
                text = f"{yaw_deg:.1f}"

                # 4. 计算文字居中
                font_scale = 1
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # 文字位置：在装甲板中心稍微向上一点
                text_org = (uv_int[0] - text_w // 2, uv_int[1] + text_h // 2)

                # 5. 绘制 (洋红色以区别于 EKF 的预测值)
                # 描边
                cv2.putText(draw, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 5)
                # 字体
                cv2.putText(draw, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)

        return draw

    def draw_ballistic(self, tf, img, imu_rpy):
        """
        绘制弹道解算视图：
        1. 红色 LOCKED: 目标的真实视觉位置 (无视差)
        2. 绿色 AIM:    枪管的实际瞄准点 (包含视差 + 重力补偿)
        """
        draw = img.copy()
        h, w = draw.shape[:2]

        if self.target is None or self.tracker_state == self.LOST:
            cv2.putText(draw, "SEARCHING...", (w//2 - 80, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            return draw

        # 1. 绘制【目标装甲板】(LOCKED)
        tgt_pos_world = self.target.pos

        cam_tgt = self.world_to_cam(tf, tgt_pos_world, imu_rpy)
        uv_tgt, vis_tgt = tf.project_point(cam_tgt)

        target_center = None
        if vis_tgt:
            target_center = (int(uv_tgt[0]), int(uv_tgt[1]))
            # 红圈锁定
            cv2.circle(draw, target_center, 8, (0, 0, 255), -1)
            cv2.circle(draw, target_center, 18, (0, 0, 255), 2)
            cv2.putText(draw, "TARGET", (target_center[0] + 25, target_center[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. 绘制【枪口瞄准点】(AIM)

        aim_pos_world = self.muzzle_target.pos

        aim_pos_cam = self.world_to_cam(tf, aim_pos_world, imu_rpy)

        # D. 投影回图像
        uv_aim, vis_aim = tf.project_point(aim_pos_cam)

        if vis_aim:
            aim_point = (int(uv_aim[0]), int(uv_aim[1]))
            # 绿色十字准星
            cv2.drawMarker(draw, aim_point, (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(draw, "AIM (GUN)", (aim_point[0] + 10, aim_point[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 画线：体现视差和重力落差
            if target_center:
                cv2.line(draw, target_center, aim_point, (255, 255, 0), 1)

        # 3. 绘制【信息面板】
        dist = np.linalg.norm(tgt_pos_world)
        angle_diff_deg = math.degrees(self.target.angle_diff)

        start_x, start_y = 20, 50
        line_step = 30
        status_color = (0, 255, 0) if self.gimbal_control[2] else (0, 0, 255)
        status_text = "FIRE ENABLE" if self.gimbal_control[2] else "HOLD FIRE"

        cv2.putText(draw, f"Dist : {dist:.2f} m", (start_x, start_y + line_step * 0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(draw, f"Pitch: {self.gimbal_control[1]:.2f} deg", (start_x, start_y + line_step * 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(draw, f"Yaw  : {self.gimbal_control[0]:.2f} deg", (start_x, start_y + line_step * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(draw, f"Diff : {angle_diff_deg:.1f} deg", (start_x, start_y + line_step * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(draw, f"[{status_text}]", (start_x, start_y + line_step * 4 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return draw
