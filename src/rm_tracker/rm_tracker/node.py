# ros2 功能包
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data   # <--- 新增这一行
# 自己写的日志管理器
from .modules.logger import LogThrottler                         # 日志节流
# 各种消息类型
from rm_interfaces.msg import ArmorsMsg, Decision, GimbalControl # 装甲板、决策、云台控制 消息
from geometry_msgs.msg import Vector3Stamped                     # 解析 dm 陀螺仪消息
from cv_bridge import CvBridge                                   # ros图像-cv图像 转换器
from sensor_msgs.msg import Image, CameraInfo                    # 原始图像、相机内参 消息
# 消息发布
from std_msgs.msg import Header
# 追踪器
import numpy as np                                               # 必要数学库
from .modules.rm_tf_tools import RmTF                            # 坐标系变换模块
from .modules.tracker import Tracker                             # 追踪器

from rcl_interfaces.msg import SetParametersResult               # 参数回调

# 多线程
import queue
import threading

class RmTracker(Node):
    def __init__(self):
        super().__init__('rm_tracker')

        # ----------------------参数声明----------------------
        # ------------------------NODE-----------------------
        self.declare_parameter('follow_decision', False) # 是否跟随决策节点的颜色选择
        self.declare_parameter('log_throttle_ms', 1000)         # 日志节流（ms）
        self.declare_parameter('rotation_rpy_r', 0.0)           # 陀螺仪到相机姿态变换 -> r
        self.declare_parameter('rotation_rpy_p', 0.0)           # 陀螺仪到相机姿态变换 -> p
        self.declare_parameter('rotation_rpy_y', -180.0)        # 陀螺仪到相机姿态变换 -> y
        self.declare_parameter('show_rpy', False)               # 陀螺仪调试 - 显示原始数据和调整后的数据
        self.declare_parameter('debug', False)                  # 显示调试信息
        self.declare_parameter('text_size', 1.0)                # 显示文字大小
        self.declare_parameter('display_fps_limit', True)       # 控制显示帧率
        self.declare_parameter('display', False)                # 显示处理结果
        # -----------------------TRACKER---------------------
        self.declare_parameter('target_color', 0)               # 目标敌方阵营 (0: RED, 1: BLUE)
        self.declare_parameter('cam_to_gun_pos_x', 0.0)         # [外参] 相机相对于枪口的平移向量 -> x (x:右, y:下, z:前)
        self.declare_parameter('cam_to_gun_pos_y', -0.05)       # [外参] 相机相对于枪口的平移向量 -> y
        self.declare_parameter('cam_to_gun_pos_z', 0.0)         # [外参] 相机相对于枪口的平移向量 -> z
        self.declare_parameter('cam_to_gun_rpy_r', 0.0)         # [外参] 相机相对于枪口的旋转欧拉角 -> r (Roll, Pitch, Yaw)
        self.declare_parameter('cam_to_gun_rpy_p', 0.0)         # [外参] 相机相对于枪口的旋转欧拉角 -> p
        self.declare_parameter('cam_to_gun_rpy_y', 0.0)         # [外参] 相机相对于枪口的旋转欧拉角 -> y
        self.declare_parameter('dist_tol', 0.05)                # [装甲板预处理] 同一装甲板不同观测点的距离容差 (防止识别同一台车辆时一直跳跃跟踪装甲板)
        self.declare_parameter('max_match_distance', 0.2)       # [匹配] EKF 预测值与观测值的最大欧氏距离阈值 (m)
        self.declare_parameter('max_match_yaw_diff', 57.0)      # [匹配] 判定装甲板切换 (Armor Jump) 的 Yaw 角度差阈值 (rad)
        self.declare_parameter('jump_cooldown_max', 20)         # [匹配] 装甲板跳转的冷却时间 (帧)
        self.declare_parameter('tracking_thres', 5)             # 进入 TRACKING 状态所需的连续检测帧数
        self.declare_parameter('lost_thres', 10)                # 进入 LOST 状态所需的连续丢失帧数
        self.declare_parameter('ekf_QR_q_xyz', 20.0)            # EKF QR 参数 - 过程噪声 Q 位置和速度的噪声系数 (s2qxyz)
        self.declare_parameter('ekf_QR_q_yaw', 100.0)           # EKF QR 参数 - 过程噪声 Q 偏航角的噪声系数 (s2qyaw)
        self.declare_parameter('ekf_QR_q_r', 800.0)             # EKF QR 参数 - 过程噪声 Q 半径的噪声系数 (s2qr)
        self.declare_parameter('ekf_QR_r_xyz_factor', 0.05)     # EKF QR 参数 - 观测噪声 R 位置观测的动态噪声系数 (距离 * factor)
        self.declare_parameter('ekf_QR_r_yaw', 0.02)            # EKF QR 参数 - 观测噪声 R 偏航角观测的固定方差
        self.declare_parameter('ekf_QR_stable_dist', 1.5)       # EKF QR 状态参数 - 稳定 detection 状态的半径门限 (m)
        self.declare_parameter('radius_r_max', 0.4)             # 旋转半径参数 - r_max
        self.declare_parameter('radius_r_min', 0.12)            # 旋转半径参数 - r_min
        self.declare_parameter('system_delay', 0.1)             # 系统延迟
        self.declare_parameter('min_spinning_frame', 10)        # 小陀螺门限
        self.declare_parameter('spinning_frame_lost', 5)        # 小陀螺丢失帧数门限
        self.declare_parameter('min_spinning_vel', 2.5)         # 旋转速度
        self.declare_parameter('bullet_speed', 28.0)            # 子弹速度
        self.declare_parameter('shootable_dist', 3.0)           # 允许发射的子弹距离阈值
        self.declare_parameter('distance_decress_ratio', 0.60)  # 距离缩减比例
        self.declare_parameter('yaw_tolerance_deg', 5.0)        # 允许发射的 yaw 角度阈值
        self.declare_parameter('pitch_tolerance_deg', 2.0)      # 允许发射的 pitch 角度阈值
        # ----------------------获取参数----------------------
        # ------------------------NODE-----------------------
        log_throttle_ms = self.get_parameter('log_throttle_ms').value   # 日志节流
        rotation_r = self.get_parameter('rotation_rpy_r').value    # 陀螺仪到相机姿态变换
        rotation_p = self.get_parameter('rotation_rpy_p').value
        rotation_y = self.get_parameter('rotation_rpy_y').value
        self.rotation_rpy = np.array([rotation_r, rotation_p, rotation_y])
        self.show_rpy = self.get_parameter('show_rpy').value  
        self.debug = self.get_parameter('debug').value          # 陀螺仪调试
        self.text_size = self.get_parameter('text_size').value  # 显示文字大小
        self.display_fps_limit = self.get_parameter('display_fps_limit').value
        self.display = self.get_parameter('display').value      # 显示处理结果
        # -----------------------TRACKER---------------------
        target_color = self.get_parameter('target_color').value    # 目标敌方阵营
        # [外参] 相机相对于枪口的平移向量
        cg_x = self.get_parameter('cam_to_gun_pos_x').value
        cg_y = self.get_parameter('cam_to_gun_pos_y').value
        cg_z = self.get_parameter('cam_to_gun_pos_z').value
        # [外参] 相机相对于枪口的旋转欧拉角
        cg_r = self.get_parameter('cam_to_gun_rpy_r').value
        cg_p = self.get_parameter('cam_to_gun_rpy_p').value
        cg_y_ang = self.get_parameter('cam_to_gun_rpy_y').value
        # [匹配与阈值]
        dist_tol = self.get_parameter('dist_tol').value                        # [装甲板预处理]
        max_match_distance = self.get_parameter('max_match_distance').value    # [匹配] 距离阈值
        max_match_yaw_diff = self.get_parameter('max_match_yaw_diff').value    # [匹配] 角度阈值
        jump_cooldown_max = self.get_parameter('jump_cooldown_max').value      # 装甲板跳转冷却时间  
        tracking_thres = self.get_parameter('tracking_thres').value            # TRACKING 阈值
        lost_thres = self.get_parameter('lost_thres').value                    # LOST 阈值
        # [EKF QR 参数]
        ekf_q_xyz = self.get_parameter('ekf_QR_q_xyz').value                # Process Noise Q: XYZ
        ekf_q_yaw = self.get_parameter('ekf_QR_q_yaw').value                # Process Noise Q: Yaw
        ekf_q_r = self.get_parameter('ekf_QR_q_r').value                    # Process Noise Q: Radius
        ekf_r_xyz_factor = self.get_parameter('ekf_QR_r_xyz_factor').value  # Measurement Noise R: XYZ Factor
        ekf_r_yaw = self.get_parameter('ekf_QR_r_yaw').value                # Measurement Noise R: Yaw
        ekf_stable_dist = self.get_parameter('ekf_QR_stable_dist').value
        QR_params = {
            'q_xyz': ekf_q_xyz,
            'q_yaw': ekf_q_yaw,
            'q_r': ekf_q_r,
            'r_xyz_factor': ekf_r_xyz_factor,
            'r_yaw': ekf_r_yaw,
            'stable_dist': ekf_stable_dist
        }
        # [旋转半径参数]
        radius_r_max = self.get_parameter('radius_r_max').value
        radius_r_min = self.get_parameter('radius_r_min').value
        radius_params = {
            'r_max': radius_r_max,
            'r_min': radius_r_min
        }
        system_delay = self.get_parameter('system_delay').value
        # 反小陀螺参数
        min_spinning_frame = self.get_parameter('min_spinning_frame').value
        spinning_frame_lost = self.get_parameter('spinning_frame_lost').value
        min_spinning_vel = self.get_parameter('min_spinning_vel').value
        bullet_speed = self.get_parameter('bullet_speed').value
        # 发弹判断
        shootable_dist = self.get_parameter('shootable_dist').value
        distance_decress_ratio = self.get_parameter('distance_decress_ratio').value
        yaw_tolerance_deg = self.get_parameter('yaw_tolerance_deg').value
        pitch_tolerance_deg = self.get_parameter('pitch_tolerance_deg').value
        # ----------------------初始化对象----------------------
        self.log_throttler = LogThrottler(self, log_throttle_ms)        # 1. 初始化 LogThrottler
        self.tracker = Tracker()                                        # 2. 初始化 Tracker   
        self.tracker.target_color = target_color        
        self.tracker.cam_to_gun_pos = np.array([cg_x, cg_y, cg_z])
        self.tracker.cam_to_gun_rpy = np.array([cg_r, cg_p, cg_y_ang])
        self.tracker.dist_tol = dist_tol
        self.tracker.max_match_distance = max_match_distance
        self.tracker.max_match_yaw_diff = max_match_yaw_diff
        self.tracker.jump_cooldown_max = jump_cooldown_max
        self.tracker.tracking_thres = int(tracking_thres)
        self.tracker.lost_thres = int(lost_thres)
        self.tracker.ekf_QR_params = QR_params
        self.tracker.radius_params = radius_params
        self.tracker.system_delay = system_delay
        self.tracker.min_spinning_frame = min_spinning_frame
        self.tracker.spinning_frame_lost = spinning_frame_lost
        self.tracker.min_spinning_vel = min_spinning_vel
        self.tracker.bullet_speed = bullet_speed
        self.tracker.shootable_dist = shootable_dist
        self.tracker.distance_decress_ratio = distance_decress_ratio
        self.tracker.yaw_tolerance_deg = yaw_tolerance_deg
        self.tracker.pitch_tolerance_deg = pitch_tolerance_deg
        self.imu_rpy = None
        self.tf = RmTF()
        self.bridge = CvBridge() # 初始化转换器

        # ---------- 新增：图像渲染抽帧控制 ----------
        self.last_render_time = self.get_clock().now()

        # ---------- 优先初始化变量 ----------
        # FPS 计算
        self.process_counter = 0
        self.last_fps_log_time = self.get_clock().now()  # 上次日志输出时间

        # 初始化锁、队列
        self.tracker_lock = threading.Lock()
        self.render_lock = threading.Lock()   # 新增：专用于保护渲染数据的锁
        self.render_snapshot = None           # 新增：存储渲染所需的轻量级数据快照
        self.track_queue = queue.Queue(maxsize=1)
        
        # ---------- 最后启动独立线程 ----------
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

        self.add_on_set_parameters_callback(self._on_params)

        # 订阅 /imu/rpy
        self.sub_imu_rpy = self.create_subscription(
            Vector3Stamped,
            '/imu/rpy',
            self.imu_rpy_cb,
            qos_profile_sensor_data
        )
        self.sub_armors = self.create_subscription(
            ArmorsMsg,
            '/detector/armors_info',
            self.armors_cb,
            qos_profile_sensor_data
        )

        self.sub_caminfo = self.create_subscription(
            CameraInfo,
            "/camera_info",
            self.camera_info_cb,
            qos_profile_sensor_data
        )

        self.sub_opponent_color = self.create_subscription(
            Decision,
            '/nav/decision',
            self.cb_opponent_color,
            qos_profile_sensor_data
        )

        self.sub_raw_img = self.create_subscription(
            Image,
            '/image_raw',
            self.res_img_cb,
            qos_profile_sensor_data
        )

        self.pub_tracking_state_img = self.create_publisher(
            Image,
            '/tracker/tracking_state_img',
            qos_profile_sensor_data
        )

        self.pub_ballistic_img = self.create_publisher(
            Image,
            '/tracker/ballistic_img',
            qos_profile_sensor_data    # <--- 同上
        )

        self.pub_gimbal_control = self.create_publisher(
            GimbalControl,
            '/tracker/gimbal_control',
            qos_profile_sensor_data
        )

    def is_changed(self, old_val, new_val, tol=1e-5):
        return abs(float(old_val) - float(new_val)) > tol

    def _on_params(self, params):
        # 【关键修改】在处理参数更新时全程加锁，防止与追踪运算线程冲突
        with self.tracker_lock:
            reset_required = False  # 标记是否需要强制重置滤波器

            for param in params:
                name = param.name
                value = param.value

                # -----------------------------------------------------------
                # 1. 节点与调试参数 (更新即可，无需重置 Tracker)
                # -----------------------------------------------------------
                if name == 'log_throttle_ms':
                    self.log_throttler._default_ms = int(value)
                elif name == 'show_rpy':
                    self.show_rpy = value
                elif name == 'debug':
                    self.debug = value
                elif name == 'text_size':
                    self.tracker.text_size = value
                elif name == 'display_fps_limit':
                    self.display_fps_limit = value
                elif name == 'display':
                    self.display = value

                # IMU 修正参数 (仅更新数组)
                elif name == 'rotation_rpy_r':
                    self.rotation_rpy[0] = value
                elif name == 'rotation_rpy_p':
                    self.rotation_rpy[1] = value
                elif name == 'rotation_rpy_y':
                    self.rotation_rpy[2] = value

                # -----------------------------------------------------------
                # 2. 追踪策略与外参 (需要检查是否变化，变化则重置)
                # -----------------------------------------------------------
                elif name == 'target_color':
                    if self.tracker.target_color != int(value):
                        self.tracker.target_color = int(value)
                        reset_required = True

                elif name == 'tracking_thres':
                    self.tracker.tracking_thres = int(value)
                elif name == 'lost_thres':
                    self.tracker.lost_thres = int(value)

                elif name == 'dist_tol':
                    if self.is_changed(self.tracker.dist_tol, value):
                        self.tracker.dist_tol = value
                        reset_required = True
                elif name == 'max_match_distance':
                    if self.is_changed(self.tracker.max_match_distance, value):
                        self.tracker.max_match_distance = value
                        reset_required = True
                elif name == 'max_match_yaw_diff':
                    if self.is_changed(self.tracker.max_match_yaw_diff, value):
                        self.tracker.max_match_yaw_diff = value
                        reset_required = True
                elif name == 'jump_cooldown_max':
                    if self.is_changed(self.tracker.jump_cooldown_max, value):
                        self.tracker.jump_cooldown_max = int(value)
                        reset_required = True

                # 外参平移
                elif name == 'cam_to_gun_pos_x':
                    if self.is_changed(self.tracker.cam_to_gun_pos[0], value):
                        self.tracker.cam_to_gun_pos[0] = value
                elif name == 'cam_to_gun_pos_y':
                    if self.is_changed(self.tracker.cam_to_gun_pos[1], value):
                        self.tracker.cam_to_gun_pos[1] = value
                elif name == 'cam_to_gun_pos_z':
                    if self.is_changed(self.tracker.cam_to_gun_pos[2], value):
                        self.tracker.cam_to_gun_pos[2] = value

                # 外参旋转
                elif name == 'cam_to_gun_rpy_r':
                    if self.is_changed(self.tracker.cam_to_gun_rpy[0], value):
                        self.tracker.cam_to_gun_rpy[0] = value
                elif name == 'cam_to_gun_rpy_p':
                    if self.is_changed(self.tracker.cam_to_gun_rpy[1], value):
                        self.tracker.cam_to_gun_rpy[1] = value
                elif name == 'cam_to_gun_rpy_y':
                    if self.is_changed(self.tracker.cam_to_gun_rpy[2], value):
                        self.tracker.cam_to_gun_rpy[2] = value

                # -----------------------------------------------------------
                # 3. 滤波器核心参数 (敏感参数，变化必须重置)
                # -----------------------------------------------------------
                # 半径模型参数
                elif name == 'radius_r_max':
                    if self.is_changed(self.tracker.radius_params['r_max'], value):
                        self.tracker.radius_params['r_max'] = value
                        reset_required = True
                elif name == 'radius_r_min':
                    if self.is_changed(self.tracker.radius_params['r_min'], value):
                        self.tracker.radius_params['r_min'] = value
                        reset_required = True

                # EKF 过程噪声 Q
                elif name == 'ekf_QR_q_xyz':
                    if self.is_changed(self.tracker.ekf_QR_params['q_xyz'], value):
                        self.tracker.ekf_QR_params['q_xyz'] = value
                        reset_required = True
                elif name == 'ekf_QR_q_yaw':
                    if self.is_changed(self.tracker.ekf_QR_params['q_yaw'], value):
                        self.tracker.ekf_QR_params['q_yaw'] = value
                        reset_required = True
                elif name == 'ekf_QR_q_r':
                    if self.is_changed(self.tracker.ekf_QR_params['q_r'], value):
                        self.tracker.ekf_QR_params['q_r'] = value
                        reset_required = True

                # EKF 观测噪声 R
                elif name == 'ekf_QR_r_xyz_factor':
                    if self.is_changed(self.tracker.ekf_QR_params['r_xyz_factor'], value):
                        self.tracker.ekf_QR_params['r_xyz_factor'] = value
                        reset_required = True
                elif name == 'ekf_QR_r_yaw':
                    if self.is_changed(self.tracker.ekf_QR_params['r_yaw'], value):
                        self.tracker.ekf_QR_params['r_yaw'] = value
                        reset_required = True

                elif name == 'ekf_QR_stable_dist':
                    if self.is_changed(self.tracker.ekf_QR_params['stable_dist'], value):
                        self.tracker.ekf_QR_params['stable_dist'] = value
                        reset_required = True

                elif name == 'system_delay':
                    if self.is_changed(self.tracker.system_delay, value):
                        self.tracker.system_delay = value
                        reset_required = True

                # 小陀螺判断
                elif name == 'min_spinning_frame':
                    if self.is_changed(self.tracker.min_spinning_frame, value):
                        self.tracker.min_spinning_frame = value
                        reset_required = True

                elif name == 'spinning_frame_lost':
                    if self.is_changed(self.tracker.spinning_frame_lost, value):
                        self.tracker.spinning_frame_lost = value
                        reset_required = True

                elif name == 'min_spinning_vel':
                    if self.is_changed(self.tracker.min_spinning_vel, value):
                        self.tracker.min_spinning_vel = value
                        reset_required = True

                elif name == 'bullet_speed':
                    if self.is_changed(self.tracker.bullet_speed, value):
                        self.tracker.bullet_speed = value
                        reset_required = False
                
                # 发弹判断
                elif name == 'shootable_dist':
                    if self.is_changed(self.tracker.shootable_dist, value):
                        self.tracker.shootable_dist = value
                        reset_required = False

                elif name == 'distance_decress_ratio':
                    if self.is_changed(self.tracker.distance_decress_ratio, value):
                        self.tracker.distance_decress_ratio = value
                        reset_required = False

                elif name == 'yaw_tolerance_deg':
                    if self.is_changed(self.tracker.yaw_tolerance_deg, value):
                        self.tracker.yaw_tolerance_deg = value
                        reset_required = False

                elif name == 'pitch_tolerance_deg':
                    if self.is_changed(self.tracker.pitch_tolerance_deg, value):
                        self.tracker.pitch_tolerance_deg = value
                        reset_required = False

            # -----------------------------------------------------------
            # 4. 执行重置逻辑
            # -----------------------------------------------------------
            if reset_required:
                self.get_logger().warn("[Params] 核心参数发生实质变更，执行 Tracker 重置。")
               # 强制设为 LOST 并清除 ID，这将导致下一帧调用 init_ekf
                self.tracker.tracker_state = self.tracker.LOST
                self.tracker.tracked_id = None

        return SetParametersResult(successful=True)
    # FPS
    def check_and_get_fps(self):
        """检查并更新FPS"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_fps_log_time).nanoseconds / 1e9
  
        self.current_fps = self.process_counter / dt
        self.process_counter = 0
        self.last_fps_log_time = current_time
            
        return self.current_fps

    # ---------- 装甲板数据入队 ----------
    def armors_cb(self, msg: ArmorsMsg):
        """仅负责将最新数据放入队列，不进行耗时计算"""
        if self.imu_rpy is None:
            return

        data = {
            'msg': msg,
            'imu_rpy': self.imu_rpy
        }

        # 队列满则剔除旧帧，保证处理最新帧
        if self.track_queue.full():
            try:
                self.track_queue.get_nowait()
            except queue.Empty:
                pass
                
        try:
            self.track_queue.put_nowait(data)
        except queue.Full:
            pass

    # ---------- 独立处理线程 ----------
    def _processing_worker(self):
        """独立线程：负责耗时的追踪计算和云台控制发布"""
        while rclpy.ok():
            try:
                # 1. 阻塞获取数据包 (单独处理队列异常)
                data = self.track_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"获取队列数据异常: {e}")
                continue

            try:
                # 2. 正常获取到数据后，处理逻辑
                self.process_counter += 1

                # 优先使用相机曝光时的原始时间戳
                try:
                    ros_clock = Time.from_msg(data['msg'].header.stamp)
                except Exception:
                    # 兜底方案
                    ros_clock = self.get_clock().now()

                # 【关键】加锁进行追踪运算，防止与图像渲染线程冲突
                with self.tracker_lock:
                    # 追踪
                    gimbal_control, logs = self.tracker.track(
                        self.tf, 
                        data['msg'], 
                        data['imu_rpy'], 
                        ros_clock
                    )
                    # 运算完毕后，立刻提取当前状态的快照
                    snapshot = self.tracker.get_render_snapshot()

                # 将快照写入共享变量（使用极短的渲染锁）
                with self.render_lock:
                    self.render_snapshot = snapshot

                # 发布云台控制指令 (锁外执行)
                if gimbal_control is not None:
                    GB = GimbalControl()
                    GB.header = Header()
                    GB.header.stamp = ros_clock.to_msg()
                    GB.header.frame_id = 'tracking_frame'
                    GB.yaw = float(gimbal_control[0])
                    GB.pitch = float(gimbal_control[1])
                    GB.can_fire = int(gimbal_control[2])
                    self.pub_gimbal_control.publish(GB)

                # 日志处理
                if self.debug:
                    for log_type, log_msg in logs:
                        if log_type in ["sys", "state", "warn", "jump"]:
                            self.get_logger().info(log_msg)
                        elif log_type == "debug":
                            if self.log_throttler.should_log("tracker_debug"):
                                self.get_logger().warn(log_msg)
                
                if self.log_throttler.should_log("gimbal_control_info"):
                    target_color_str = f"{self.tracker.c.PINK}跟踪红色{self.tracker.c.RESET}" if self.tracker.target_color == 0 else f"{self.tracker.c.PINK}跟踪蓝色{self.tracker.c.RESET}" if self.tracker.target_color == 1 else "我不知道"
                    self.get_logger().info(f"[rm_tracker] FPS: {self.check_and_get_fps():.2f} {target_color_str} Gimbal - pitch: {gimbal_control[1]:.2f} || yaw: {gimbal_control[0]:.2f} || fire: {gimbal_control[2]:.0f}")

            except Exception as e:
                self.get_logger().error(f"处理线程异常: {e}")

    # ---------- 传感器回调 ----------
    def imu_rpy_cb(self, msg: Vector3Stamped):
        raw_rpy = [msg.vector.x, msg.vector.y, msg.vector.z]
        imu_rpy = self.tf.rotate_pose_axis(raw_rpy, self.rotation_rpy)
        self.imu_rpy = imu_rpy

        if self.log_throttler.should_log("show_rpy") and self.show_rpy:
            self.get_logger().info(
                f"[rm_tracker] RPY received: roll={raw_rpy[0]:.2f}, pitch={raw_rpy[1]:.2f}, yaw={raw_rpy[2]:.2f}"
            )
            self.get_logger().info(
                f"[rm_tracker] RPY processed: roll={imu_rpy[0]:.2f}, pitch={imu_rpy[1]:.2f}, yaw={imu_rpy[2]:.2f}"
            )

    def camera_info_cb(self, msg: CameraInfo):
        # 只需要设置一次即可，避免重复计算
        if self.tf.has_camera_info:
            return

        self.get_logger().info(f"收到相机内参! {msg.width}x{msg.height}")

        # 将展平的数组重塑为 3x3 矩阵
        k = np.array(msg.k).reshape(3, 3)
        d = np.array(msg.d)
        self.tf.set_camera_info(k, d, width=msg.width, height=msg.height)

    # ---------- 图像渲染回调 ----------
    def res_img_cb(self, msg: Image):
        # 性能开关：如果没开启显示，直接不处理图像，节省 CPU
        if not self.display:
            return

        # --- 新增：抽帧降频逻辑 ---
        current_time = self.get_clock().now()
        dt = (current_time - self.last_render_time).nanoseconds / 1e9
        
        # 如果距离上次渲染时间不足 1/30 秒，直接跳过，释放 CPU 给追踪线程
        if dt < (1.0 / 30.0) and self.display_fps_limit:
            return
            
        self.last_render_time = current_time
        # --------------------------

        # 1. 极速获取当前快照并释放锁
        with self.render_lock:
            if self.render_snapshot is None:
                return
            current_snapshot = self.render_snapshot
            
        try:
            # ROS Image -> OpenCV
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 核心绘制逻辑
            if self.imu_rpy is not None and self.tf.has_camera_info:
                # 2. 完全无锁渲染：将快照传入新的 display 函数
                ballistic_img, tracking_state_img = self.tracker.display_with_snapshot(
                    current_snapshot,
                    self.debug,
                    self.tf,
                    cv_img,
                    self.imu_rpy
                )
            else:
                return

            # OpenCV -> ROS Image 并发布
            if tracking_state_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(tracking_state_img, "bgr8")
                out_msg.header = msg.header
                self.pub_tracking_state_img.publish(out_msg)
                
            if ballistic_img is not None:
                out_msg = self.bridge.cv2_to_imgmsg(ballistic_img, "bgr8")
                out_msg.header = msg.header
                self.pub_ballistic_img.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"图像处理回调异常: {e}")

    # ---------- 决策回调 ----------
    def cb_opponent_color(self, msg: Decision):
        try:
            follow = int(self.get_parameter('follow_decision').value)
            if follow != 1:
                return

            if hasattr(msg, 'color'):
                cur = int(self.get_parameter('target_color').value)
                if msg.color != cur:
                    color = int(msg.color)
                    # 【关键修改】只调用 set_parameters 即可。它会自动触发 _on_params，
                    # 并在 _on_params 内部安全加锁并修改 tracker 的颜色，不需要在这里再次赋值。
                    self.set_parameters([rclpy.parameter.Parameter(
                        'target_color',
                        rclpy.parameter.Parameter.Type.INTEGER,
                        int(color)
                    )])
                    self.get_logger().info(f"追踪颜色切换为 {color}")
        except Exception as e:
            self.get_logger().error(f"处理 Decision 异常：{e}")
def main(args=None):
    rclpy.init(args=args)
    node = RmTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass 
    finally:
        node.destroy_node()
        # 增加检查，防止重复 shutdown
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()