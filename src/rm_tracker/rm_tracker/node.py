# ros2 功能包
import rclpy
from rclpy.node import Node
# 自己写的日志管理器
from .modules.logger import LogThrottler                         # 日志节流
# 各种消息类型
from rm_interfaces.msg import ArmorsMsg, Decision, GimbalControl, Heartbeat # 装甲板、决策、云台控制 消息 心跳包
from geometry_msgs.msg import Vector3Stamped                     # 解析 dm 陀螺仪消息
from cv_bridge import CvBridge                                   # ros图像-cv图像 转换器
from sensor_msgs.msg import Image, CameraInfo                    # 原始图像、相机内参 消息
# 消息发布
from rclpy.publisher import Publisher
from std_msgs.msg import Header
# 追踪器
import numpy as np                                               # 必要数学库
from .modules.rm_tf_tools import RmTF                            # 坐标系变换模块
from .modules.tracker import Tracker                             # 追踪器

from rcl_interfaces.msg import SetParametersResult               # 参数回调

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
        self.declare_parameter('debug_info', False)
        self.declare_parameter('display_res', False)            # 显示处理结果
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
        self.declare_parameter('max_match_yaw_diff', 57.0)       # [匹配] 判定装甲板切换 (Armor Jump) 的 Yaw 角度差阈值 (rad)
        self.declare_parameter('tracking_thres', 5)             # 进入 TRACKING 状态所需的连续检测帧数
        self.declare_parameter('lost_thres', 10)                # 进入 LOST 状态所需的连续丢失帧数
        self.declare_parameter('ekf_QR_q_xyz', 20.0)            # EKF QR 参数 - 过程噪声 Q 位置和速度的噪声系数 (s2qxyz)
        self.declare_parameter('ekf_QR_q_yaw', 100.0)           # EKF QR 参数 - 过程噪声 Q 偏航角的噪声系数 (s2qyaw)
        self.declare_parameter('ekf_QR_q_r', 800.0)             # EKF QR 参数 - 过程噪声 Q 半径的噪声系数 (s2qr)
        self.declare_parameter('ekf_QR_r_xyz_factor', 0.05)     # EKF QR 参数 - 观测噪声 R 位置观测的动态噪声系数 (距离 * factor)
        self.declare_parameter('ekf_QR_r_yaw', 0.02)            # EKF QR 参数 - 观测噪声 R 偏航角观测的固定方差
        self.declare_parameter('radius_r1', 0.26)               # 旋转半径参数 - r1 (默认一长一短)
        self.declare_parameter('radius_r2', 0.26)               # 旋转半径参数 - r2
        self.declare_parameter('radius_r_max', 0.4)             # 旋转半径参数 - r_max
        self.declare_parameter('radius_r_min', 0.12)            # 旋转半径参数 - r_min
        self.declare_parameter('bullet_speed', 28.0)            # 子弹速度
        self.declare_parameter('yaw_threshold_deg', 5.0)        # 允许发射的 yaw 角度阈值
        self.declare_parameter('pitch_threshold_deg', 2.0)      # 允许发射的 pitch 角度阈值
        # ----------------------获取参数----------------------
        # ------------------------NODE-----------------------
        log_throttle_ms = self.get_parameter('log_throttle_ms').value   # 日志节流
        rotation_r = self.get_parameter('rotation_rpy_r').value    # 陀螺仪到相机姿态变换
        rotation_p = self.get_parameter('rotation_rpy_p').value
        rotation_y = self.get_parameter('rotation_rpy_y').value
        self.rotation_rpy = np.array([rotation_r, rotation_p, rotation_y])
        self.debug_info = self.get_parameter('debug_info').value
        self.show_rpy = self.get_parameter('show_rpy').value            # 陀螺仪调试
        self.display_res = self.get_parameter('display_res').value      # 显示处理结果
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
        tracking_thres = self.get_parameter('tracking_thres').value            # TRACKING 阈值
        lost_thres = self.get_parameter('lost_thres').value                    # LOST 阈值
        # [EKF QR 参数]
        ekf_q_xyz = self.get_parameter('ekf_QR_q_xyz').value                # Process Noise Q: XYZ
        ekf_q_yaw = self.get_parameter('ekf_QR_q_yaw').value                # Process Noise Q: Yaw
        ekf_q_r = self.get_parameter('ekf_QR_q_r').value                    # Process Noise Q: Radius
        ekf_r_xyz_factor = self.get_parameter('ekf_QR_r_xyz_factor').value  # Measurement Noise R: XYZ Factor
        ekf_r_yaw = self.get_parameter('ekf_QR_r_yaw').value                # Measurement Noise R: Yaw
        QR_params = {
            'q_xyz': ekf_q_xyz,
            'q_yaw': ekf_q_yaw,
            'q_r': ekf_q_r,
            'r_xyz_factor': ekf_r_xyz_factor,
            'r_yaw': ekf_r_yaw
        }
        # [旋转半径参数]
        radius_r1 = self.get_parameter('radius_r1').value
        radius_r2 = self.get_parameter('radius_r2').value
        radius_r_max = self.get_parameter('radius_r_max').value
        radius_r_min = self.get_parameter('radius_r_min').value
        radius_params = {
            'r1': radius_r1,
            'r2': radius_r2,
            'r_max': radius_r_max,
            'r_min': radius_r_min
        }
        bullet_speed = self.get_parameter('bullet_speed').value
        # 发弹判断
        yaw_threshold_deg = self.get_parameter('yaw_threshold_deg').value
        pitch_threshold_deg = self.get_parameter('pitch_threshold_deg').value
        # ----------------------初始化对象----------------------
        self.log_throttler = LogThrottler(self, log_throttle_ms)        # 1. 初始化 LogThrottler
        self.tracker = Tracker()                                        # 2. 初始化 Tracker
        self.tracker.target_color = target_color
        self.tracker.ekf_QR_params = QR_params
        self.tracker.radius_params = radius_params
        self.tracker.max_match_distance = max_match_distance
        self.tracker.max_match_yaw_diff = max_match_yaw_diff
        self.tracker.dist_tol = dist_tol
        self.tracker.tracking_thres = int(tracking_thres)
        self.tracker.lost_thres = int(lost_thres)
        self.tracker.cam_to_gun_pos = np.array([cg_x, cg_y, cg_z])
        self.tracker.cam_to_gun_rpy = np.array([cg_r, cg_p, cg_y_ang])
        self.tracker.bullet_speed = bullet_speed
        self.tracker.yaw_threshold_deg = yaw_threshold_deg
        self.tracker.pitch_threshold_deg = pitch_threshold_deg
        self.imu_rpy = None
        self.tf = RmTF()
        self.bridge = CvBridge() # 初始化转换器

        self.add_on_set_parameters_callback(self._on_params)

        # 订阅 /imu/rpy
        self.sub_imu_rpy = self.create_subscription(
            Vector3Stamped,
            '/imu/rpy',
            self.imu_rpy_cb,
            1
        )
        self.sub_armors = self.create_subscription(
            ArmorsMsg,
            '/detector/armors_info',
            self.armors_cb,
            1
        )

        self.sub_caminfo = self.create_subscription(
            CameraInfo,
            "/camera_info",
            self.camera_info_cb,
            10
        )

        self.sub_opponent_color = self.create_subscription(
            Decision,
            '/nav/decision',
            self.cb_opponent_color,
            10
        )

        self.sub_raw_img = self.create_subscription(
            Image,
            '/image_raw',
            self.res_img_cb,
            10
        )

        self.pub_estimate_img = self.create_publisher(
            Image,
            '/tracker/estimate_img',
            10
        )

        self.pub_yaw_debug_img = self.create_publisher(
            Image,
            '/tracker/yaw_debug_img',
            10
        )

        self.pub_ballistic_img = self.create_publisher(
            Image,
            '/tracker/ballistic_img',
            10
        )

        self.pub_gimbal_control = self.create_publisher(
            GimbalControl,
            '/tracker/gimbal_control',
            10
        )

        # heartbeat
        self.heartbeat = self.create_publisher(
            Heartbeat, '/tracker/heartbeat', 10)
        self.heartbeat_timer = self.create_timer(0.5, self._publish_heartbeat)

    def is_changed(self, old_val, new_val, tol=1e-5):
        return abs(float(old_val) - float(new_val)) > tol

    def _on_params(self, params):
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
            elif name == 'debug_info':
                self.debug_info = value
            elif name == 'display_res':
                self.display_res = value

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
                # 整数直接比较
                if self.tracker.target_color != int(value):
                    self.tracker.target_color = int(value)
                    reset_required = True

            # 状态机阈值 (整数)
            elif name == 'tracking_thres':
                self.tracker.tracking_thres = int(value)
            elif name == 'lost_thres':
                self.tracker.lost_thres = int(value)

            # 匹配阈值 (浮点数，使用 is_changed)
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

            # 外参平移
            elif name == 'cam_to_gun_pos_x':
                if self.is_changed(self.tracker.cam_to_gun_pos[0], value):
                    self.tracker.cam_to_gun_pos[0] = value
                    reset_required = True
            elif name == 'cam_to_gun_pos_y':
                if self.is_changed(self.tracker.cam_to_gun_pos[1], value):
                    self.tracker.cam_to_gun_pos[1] = value
                    reset_required = True
            elif name == 'cam_to_gun_pos_z':
                if self.is_changed(self.tracker.cam_to_gun_pos[2], value):
                    self.tracker.cam_to_gun_pos[2] = value
                    reset_required = True

            # 外参旋转
            elif name == 'cam_to_gun_rpy_r':
                if self.is_changed(self.tracker.cam_to_gun_rpy[0], value):
                    self.tracker.cam_to_gun_rpy[0] = value
                    reset_required = True
            elif name == 'cam_to_gun_rpy_p':
                if self.is_changed(self.tracker.cam_to_gun_rpy[1], value):
                    self.tracker.cam_to_gun_rpy[1] = value
                    reset_required = True
            elif name == 'cam_to_gun_rpy_y':
                if self.is_changed(self.tracker.cam_to_gun_rpy[2], value):
                    self.tracker.cam_to_gun_rpy[2] = value
                    reset_required = True

            # -----------------------------------------------------------
            # 3. 滤波器核心参数 (敏感参数，变化必须重置)
            # -----------------------------------------------------------
            # 半径模型参数
            elif name == 'radius_r1':
                if self.is_changed(self.tracker.radius_params['r1'], value):
                    self.tracker.radius_params['r1'] = value
                    reset_required = True
            elif name == 'radius_r2':
                if self.is_changed(self.tracker.radius_params['r2'], value):
                    self.tracker.radius_params['r2'] = value
                    reset_required = True
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

            elif name == 'bullet_speed':
                if self.is_changed(self.tracker.bullet_speed, value):
                    self.tracker.bullet_speed = value
                    reset_required = True

            # 发弹角度判断
            elif name == 'yaw_threshold_deg':
                if self.is_changed(self.tracker.yaw_threshold_deg, value):
                    self.tracker.yaw_threshold_deg = value
                    reset_required = False
            elif name == 'pitch_threshold_deg':
                if self.is_changed(self.tracker.pitch_threshold_deg, value):
                    self.tracker.pitch_threshold_deg = value
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

    def armors_cb(self, msg: ArmorsMsg):
        if self.imu_rpy is None:
            return
        else:
            # 追踪
            gimbal_control, logs = self.tracker.track(self.tf, msg, self.imu_rpy, self.get_clock().now())

            # 发布云台控制信息
            GB = GimbalControl()
            GB.header = Header()
            GB.header.stamp = self.get_clock().now().to_msg()
            GB.header.frame_id = 'tracking_frame'
            GB.yaw = float(gimbal_control[0])
            GB.pitch = float(gimbal_control[1])
            GB.can_fire = int(gimbal_control[2])
            self.pub_gimbal_control.publish(GB)

            if self.debug_info:
                for log_type, log_msg in logs:
                    # 策略 1: 状态切换、初始化、警告 -> 直接打印 (因为频率低且重要)
                    if log_type in ["sys", "state", "warn", "jump"]:
                        self.get_logger().info(log_msg)

                    # 策略 2: 调试信息 (如 No match) -> 走节流阀 (防止刷屏)
                    elif log_type == "debug":
                        if self.log_throttler.should_log("tracker_debug"): # 使用特定的 key
                            self.get_logger().warn(log_msg)
            # 换算 self.tracker.dt 变成fps
            fps = 1.0 / self.tracker.dt

            if self.tracker.target_color == 0:
                target_color_str = f"{self.tracker.c.PINK}跟踪{self.tracker.c.RED}红色{self.tracker.c.RESET}"
            elif self.tracker.target_color == 1:
                target_color_str = f"{self.tracker.c.PINK}跟踪{self.tracker.c.BLUE}蓝色{self.tracker.c.RESET}"
            else:
                target_color_str = f"{self.tracker.c.PINK}我不知道{self.tracker.c.RESET}"

            if self.log_throttler.should_log("gimbal_control_info"): # 使用特定的 key
                            self.get_logger().info(f"[rm_tracker]"+
                                                   f"{self.tracker.c.PINK}FPS{self.tracker.c.RESET}: {self.tracker.c.CYAN}{fps:.2f}{self.tracker.c.RESET}"+
                                                   f" {target_color_str}"
                                                   f" {self.tracker.c.PINK}Gimbal control{self.tracker.c.RESET}"+
                                                   f" - {self.tracker.c.GREEN}pitch:{self.tracker.c.RESET} {self.tracker.c.CYAN}{gimbal_control[1]:.2f}{self.tracker.c.RESET}"+
                                                   f" || {self.tracker.c.GREEN}yaw:{self.tracker.c.RESET} {self.tracker.c.CYAN}{gimbal_control[0]:.2f}{self.tracker.c.RESET}"+
                                                   f" || {self.tracker.c.GREEN}fire:{self.tracker.c.RESET} {self.tracker.c.CYAN}{gimbal_control[2]:.0f}{self.tracker.c.RESET}"
                                                   )

    def camera_info_cb(self, msg: CameraInfo):
        # 只需要设置一次即可，避免重复计算
        if self.tf.has_camera_info:
            return

        self.get_logger().info(f"收到相机内参! {msg.width}x{msg.height}")

        # 将展平的数组重塑为 3x3 矩阵
        k = np.array(msg.k).reshape(3, 3)
        d = np.array(msg.d)

        # 【核心修改】直接把 msg.width 和 msg.height 传进去
        # 这样 rm_tf_tools 就能利用它们算出 width/2 和 height/2 了
        self.tf.set_camera_info(k, d, width=msg.width, height=msg.height)

    def res_img_cb(self, msg: Image):
        # 1. 性能开关：如果没开启显示，直接不处理图像，节省 CPU
        if not self.display_res:
            return

        try:
            # 3. ROS Image -> OpenCV
            # 使用 bgr8 格式，因为 opencv 默认绘图是用 BGR
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 4. 核心绘制逻辑
            # 必须保证有 IMU 数据（用于坐标回转）和 相机内参（用于投影）
            if self.imu_rpy is not None and self.tf.has_camera_info:
                # 调用 tracker 的绘图函数
                estimate_img = self.tracker.draw_estimate(self.tf, cv_img, self.imu_rpy)
                yaw_debug_img = self.tracker.draw_observation_yaw(self.tf, cv_img, self.imu_rpy)
                ballistic_img = self.tracker.draw_ballistic(self.tf, cv_img, self.imu_rpy)
            else:
                return None

            # 5. OpenCV -> ROS Image 并发布
            out_msg = self.bridge.cv2_to_imgmsg(estimate_img, "bgr8")
            out_msg.header = msg.header # 保持原始的时间戳和坐标系
            self.pub_estimate_img.publish(out_msg)

            out_msg = self.bridge.cv2_to_imgmsg(yaw_debug_img, "bgr8")
            out_msg.header = msg.header
            self.pub_yaw_debug_img.publish(out_msg)

            out_msg = self.bridge.cv2_to_imgmsg(ballistic_img, "bgr8")
            out_msg.header = msg.header
            self.pub_ballistic_img.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"图像处理回调异常: {e}")
            
    # ---------- heartbeat ----------
    def _publish_heartbeat(self):
        now = self.get_clock().now()
        msg = Heartbeat()
        msg.heartbeat_time = now.to_msg().sec
        self.heartbeat.publish(msg)

    # ---------- 决策回调 ----------
    def cb_opponent_color(self, msg: Decision):
        try:
            follow = int(self.get_parameter('follow_decision').value)
            if follow != 1:
                return

            if hasattr(msg, 'color'):
                cur = int(self.get_parameter('target_color').value)
                if msg.color != cur:
                    self.set_parameters([rclpy.parameter.Parameter(
                        'target_color',
                        rclpy.parameter.Parameter.Type.INTEGER,
                        int(msg.color)
                    )])
                    self.tracker.target_color = msg.color
                    self.get_logger().info(f"追踪颜色切换为 {int(msg.color)}")
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
        rclpy.shutdown()

if __name__ == '__main__':
    main()
