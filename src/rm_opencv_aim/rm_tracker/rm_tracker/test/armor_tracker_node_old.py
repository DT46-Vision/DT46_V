"""
Armor Tracker Node (IMM-3D, xyz 输入=毫米)
--------------------------------------
- 订阅:  /detector/armors_info (ArmorsCppMsg) —— armors[i].dx,dy,dz 单位: **毫米**
- 转换:  本节点将 dx,dy,dz 交给 Tracker (IMM-3D)
- 追踪:  交给 rm_tracker.armor_tracker.Tracker（内部 IMM 基于 xyz(mm)）
- 发布:  /tracker/target (ArmorTracking) —— yaw/pitch(度) 与 shoot_flag
- 参数:  IMM + 弹道参数均暴露在 ROS 参数中，改动会即时生效
"""

import rclpy
import cv2
import numpy as np
import tf_transformations
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import Image, CameraInfo
from rcl_interfaces.msg import SetParametersResult
from rm_opencv_aim.rm_tracker.rm_tracker.modules.tracker import Tracker
from geometry_msgs.msg import PoseStamped, TransformStamped
from rm_interfaces.msg import ArmorsCppMsg, ArmorTracking, Decision
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R

class LogThrottler:
    def __init__(self, node: Node, default_ms: int = 1000):
        self._node = node
        self._default_ms = int(default_ms)
        self._last_ns = {}

    def should_log(self, key: str, throttle_ms: int = None):
        if throttle_ms is None:
            throttle_ms = self._default_ms
        throttle_ns = int(throttle_ms) * 1_000_000
        now_ns = self._node.get_clock().now().nanoseconds
        last_ns = self._last_ns.get(key)
        if last_ns is None or (now_ns - last_ns) >= throttle_ns:
            self._last_ns[key] = now_ns
            return True
        return False

class ArmorTrackerNode(Node):
    def __init__(self, name: str = "armor_tracker_node"):
        super().__init__(name)

        # Tracker（内部接收 mm；Tracker 内部已做单位归一）
        self.tracker = Tracker(logger=self.get_logger())
        self.bridge = CvBridge()
        self._c = self.tracker.color

        # ---------- 显示/打印/FPS ----------
        self.declare_parameter('dis_res', False)
        self.declare_parameter('use_geometric_center', True)
        self.declare_parameter('log_throttle_ms', 1000)
        self.declare_parameter('fps_window_sec', 1.0)

        # ---------- 追踪参数 ----------
        self.declare_parameter('use_kf',           True)
        self.declare_parameter('max_lost_ms',      200)   # 默认 200 ms
        self.declare_parameter('tracking_color',   0)     # 0:红, 1:蓝, 10:不追
        self.declare_parameter('follow_decision',  0)
        self.declare_parameter('track_deep_tol',   10.0)  # mm
        self.declare_parameter('shoot_yaw_max',    1.5)   # deg
        self.declare_parameter('shoot_pitch_max',  1.0)   # deg

        # ---------- 滤波参数 ----------
        self.declare_parameter('q_acc', 0.05)
        self.declare_parameter('r_meas', 0.02)
        self.declare_parameter('win_size', 8)
        self.declare_parameter('poly_degree', 2)
        self.declare_parameter('alpha', 0.7)
        self.declare_parameter('future_scale', 1.0)
        self.declare_parameter('q_rot', 0.002)
        self.declare_parameter('r_rot', 0.01)

        # ---------- 外参（米/度） ----------
        self.declare_parameter('camera_tx_m',      0.0)
        self.declare_parameter('camera_ty_m',      -0.03)
        self.declare_parameter('camera_tz_m',      0.0)
        self.declare_parameter('camera_yaw_deg',   -4.0)
        self.declare_parameter('camera_pitch_deg', 0.0)
        self.declare_parameter('camera_roll_deg',  0.0)

        # ---------- 弹道参数 ----------
        self.declare_parameter('use_ballistics', True)
        self.declare_parameter('bullet_speed_mps', 30.0)
        self.declare_parameter('gravity', 9.81)
        self.declare_parameter('extra_latency_s', 0.02)

        # ---------- 弹道收敛/稳态（新增） ----------
        self.declare_parameter('ballistic_min_dist_m',     0.25)
        self.declare_parameter('ballistic_max_pitch_deg',  40.0)
        self.declare_parameter('ballistic_conv_tol_deg',   0.05)
        self.declare_parameter('ballistic_time_damping',   0.5)

        self.add_on_set_parameters_callback(self._on_params)

        # ---------- 通信 ----------
        self.sub_caminfo = self.create_subscription(
            CameraInfo, "/camera_info", self._cb_camera_info, 10)
        self.sub_armors = self.create_subscription(
            ArmorsCppMsg, '/detector/armors_info', self._cb_armors, 10)
        self.sub_decision = self.create_subscription(
            Decision, '/nav/decision', self._cb_decision, 10)
        self.sub_raw_img = self.create_subscription(
            Image, '/image_raw', self._cb_res_img, 10)

        self.pub_tracking = self.create_publisher(ArmorTracking, '/tracker/target', 10)
        self.pub_res_img = self.create_publisher(Image, '/tracker/res_img', 10)

        self.pub_pose = self.create_publisher(PoseStamped, "/tracker/pose", 10)
        self.pub_marker = self.create_publisher(Marker, "armor_marker", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ---------- 显示/计时/FPS ----------
        self.dis_res = self.get_parameter('dis_res').value
        self.log = LogThrottler(self, default_ms=int(self.get_parameter('log_throttle_ms').value))
        self._fps_window_sec = float(self.get_parameter('fps_window_sec').value)
        self._last_fps_time = self.get_clock().now()
        self._processed_in_window = 0

        # dt 用于给 Tracker/IMM
        self._last_update_time = self.get_clock().now()

        self.get_logger().info('Armor Tracker Node started (IMM-3D + Ballistics).')

        # 初始化：同步参数到 tracker + 重建 KF
        self._sync_all_params_to_tracker()

    # ---------- 同步全部参数到 Tracker ----------
    def _sync_all_params_to_tracker(self):
        t = self.tracker
        gp = self.get_parameter

        t.use_geometric_center = bool(gp('use_geometric_center').value)

        # 追踪与限幅
        t.use_kf          = bool(gp('use_kf').value)
        t.max_lost_ms     = int(gp('max_lost_ms').value)
        t.tracking_color  = int(gp('tracking_color').value)
        t.follow_decision = int(gp('follow_decision').value)
        t.track_deep_tol  = float(gp('track_deep_tol').value)
        t.shoot_yaw_max   = float(gp('shoot_yaw_max').value)
        t.shoot_pitch_max = float(gp('shoot_pitch_max').value)

        # 外参
        t.camera_tx_m      = float(gp('camera_tx_m').value)
        t.camera_ty_m      = float(gp('camera_ty_m').value)
        t.camera_tz_m      = float(gp('camera_tz_m').value)
        t.camera_yaw_deg   = float(gp('camera_yaw_deg').value)
        t.camera_pitch_deg = float(gp('camera_pitch_deg').value)
        t.camera_roll_deg  = float(gp('camera_roll_deg').value)

        # 弹道参数
        t.use_ballistics         = bool(gp('use_ballistics').value)
        t.bullet_speed_mps       = float(gp('bullet_speed_mps').value)
        t.gravity                = float(gp('gravity').value)
        t.extra_latency_s        = float(gp('extra_latency_s').value)

        # 弹道收敛/稳态（新增）
        t.ballistic_min_dist_m   = float(gp('ballistic_min_dist_m').value)
        t.ballistic_max_pitch_deg= float(gp('ballistic_max_pitch_deg').value)
        t.ballistic_conv_tol_deg = float(gp('ballistic_conv_tol_deg').value)
        t.ballistic_time_damping = float(gp('ballistic_time_damping').value)

        # 滤波参数
        t.q_acc         = float(gp('q_acc').value)
        t.r_meas        = float(gp('r_meas').value)
        t.win_size      = int(gp('win_size').value)
        t.poly_degree   = int(gp('poly_degree').value)
        t.alpha         = float(gp('alpha').value)
        t.future_scale  = float(gp('future_scale').value)
        t.q_rot         = float(gp('q_rot').value)
        t.r_rot         = float(gp('r_rot').value)

        # 同步给滤波器
        try:
            t.kf.set_params(
                q_acc=t.q_acc,
                r_meas=t.r_meas,
                win_size=t.win_size,
                poly_degree=t.poly_degree,
                alpha=t.alpha,
                future_scale=t.future_scale,
                q_rot=t.q_rot,
                r_rot=t.r_rot,
            )
            self.get_logger().info("滤波参数同步完成。")
        except Exception as e:
            self.get_logger().error(f"滤波参数同步失败: {e}")

    # ---------- 动态参数回调 ----------
    def _on_params(self, params):
        for p in params:
            name, val = p.name, p.value
            try:
                if name in [
                    'q_acc', 'r_meas', 'win_size', 'poly_degree',
                    'alpha', 'future_scale', 'q_rot', 'r_rot'
                ]:
                    setattr(self.tracker, name, val)
                    # 实时同步
                    self.tracker.kf.set_params(**{name: val})
                    self.get_logger().info(f"滤波参数 {name} 已更新为 {val}")
                elif name == 'dis_res':
                    self.dis_res = val
                elif name == 'log_throttle_ms':
                    self.log._default_ms = val
                elif hasattr(self.tracker, name):
                    setattr(self.tracker, name, val)
                    self.get_logger().info(f"Tracker 参数 {name} 已更新为 {val}")
            except Exception as e:
                self.get_logger().warn(f"参数 {name} 更新失败：{e}")

        return SetParametersResult(successful=True)

    def _cb_camera_info(self, msg: CameraInfo):
        """
        接收 CameraInfo 并更新 tracker 的相机内参。
        统一使用 OpenCV 坐标系 (X→右, Y→下, Z→前)。
        """
        # --- 基本内参 (K矩阵行主序) ---
        self.tracker.fx = float(msg.k[0])
        self.tracker.fy = float(msg.k[4])
        self.tracker.cx = float(msg.k[2])
        self.tracker.cy = float(msg.k[5])

        # --- 几何中心替代（可选）---
        if self.get_parameter('use_geometric_center').value:
            w = float(msg.width)
            h = float(msg.height)
            self.tracker.cx = w / 2.0
            self.tracker.cy = h / 2.0
            self.get_logger().info(
                f"Using geometric center: cx={self.tracker.cx:.1f}, cy={self.tracker.cy:.1f}"
            )
        else:
            # 保持标定光心
            self.get_logger().info(
                f"Using calibration principal point: cx={self.tracker.cx:.1f}, cy={self.tracker.cy:.1f}"
            )

        # --- 畸变系数 D ---
        try:
            # msg.d 可能为空或长度不足，取前5个够用（k1,k2,p1,p2,k3）
            self.tracker.dist = np.array(list(msg.d)[:5], dtype=float)
        except Exception:
            self.tracker.dist = np.zeros(5, dtype=float)
            self.get_logger().warn("CameraInfo.d is empty, assuming zero distortion.")

        # --- 状态更新 ---
        self.tracker.has_camera_info = True

        # 可选：只收一次 CameraInfo 即可（防止反复更新）
        try:
            self.destroy_subscription(self.sub_caminfo)
        except Exception:
            pass

        self.get_logger().info(
            f"Camera info received: fx={self.tracker.fx:.1f}, fy={self.tracker.fy:.1f}, "
            f"cx={self.tracker.cx:.1f}, cy={self.tracker.cy:.1f}, D={self.tracker.dist.tolist()}"
        )

    def target_pub_to_node(self, msg: ArmorTracking):
        try:
            now = self.get_clock().now()
            dt = (now - self._last_update_time).nanoseconds / 1e9
            if dt <= 0.0 or dt > 0.2:
                dt = 0.01
            self._last_update_time = now

            if hasattr(self.tracker, 'update_dt'):
                self.tracker.update_dt(dt)

            track_color = int(self.get_parameter('tracking_color').value)
            if track_color == 10:
                if self.log.should_log('no_track_color'):
                    self.get_logger().warn("tracking_color=10：不追踪装甲板。")
                return

            shoot_flag, yaw_sent, pitch_sent, msg_color = self.tracker.track_armor(msg)

            out = ArmorTracking()
            out.header = Header()
            out.header.stamp = now.to_msg()
            out.header.frame_id = 'tracking_armor_frame'
            out.yaw = float(yaw_sent)
            out.pitch = float(pitch_sent)
            out.shoot_flag = int(shoot_flag)
            self.pub_tracking.publish(out)

            if msg_color and self.log.should_log('track_color'):
                self.get_logger().info(msg_color)

            if self.log.should_log('pub_tracking'):
                c = self._c
                self.get_logger().info(
                    f"发布 {c.CYAN}yaw{c.RESET}:{c.PINK}{out.yaw:.2f}{c.RESET} "
                    f"|| {c.CYAN}pitch{c.RESET}:{c.PINK}{out.pitch:.2f}{c.RESET} "
                    f"|| {c.CYAN}shoot{c.RESET}:{c.PINK}{out.shoot_flag:.0f}{c.RESET}"
                )

            self._processed_in_window += 1
            if (now - self._last_fps_time).nanoseconds >= int(self._fps_window_sec * 1e9):
                secs = (now - self._last_fps_time).nanoseconds / 1e9
                fps = self._processed_in_window / max(secs, 1e-6)
                self.get_logger().warning(f"[FPS_tracker] {fps:.1f}")
                self._processed_in_window = 0
                self._last_fps_time = now

        except Exception as e:
            self.get_logger().error(f"处理 ArmorsCppMsg 异常：{e}")

    # def pub_to_rviz(self):
    #     try:
    #         now = self.get_clock().now().to_msg()

    #         # --- 相机位置 ---
    #         t_cam = TransformStamped()
    #         t_cam.header.stamp = now
    #         t_cam.header.frame_id = "world"  # 世界坐标系
    #         t_cam.child_frame_id = "camera_link"
    #         t_cam.transform.translation.x = 0.0
    #         t_cam.transform.translation.y = 0.0
    #         t_cam.transform.translation.z = 0.40
    #         t_cam.transform.rotation.w = 1.0

    #         # 旋转：相机竖直放置，绕 X 轴旋转 -90°，调整 Y 轴为向上
    #         # 用 SciPy 直接生成四元数 (x, y, z, w)
    #         q_cam = R.from_euler("x", -90.0, degrees=True).as_quat()
    #         t_cam.transform.rotation.x = float(q_cam[0])
    #         t_cam.transform.rotation.y = float(q_cam[1])
    #         t_cam.transform.rotation.z = float(q_cam[2])
    #         t_cam.transform.rotation.w = float(q_cam[3])

    #         self.tf_broadcaster.sendTransform(t_cam)

    #         # --- 如果没有目标，清空所有数据 ---
    #         if not self.tracker.if_find:
    #             # 清空 Pose
    #             pose_msg = PoseStamped()
    #             pose_msg.header.stamp = now
    #             pose_msg.header.frame_id = "camera_link"
    #             pose_msg.pose.orientation.w = 1.0
    #             self.pub_pose.publish(pose_msg)

    #             # 清空装甲板 Marker
    #             marker = Marker()
    #             marker.header.stamp = now
    #             marker.header.frame_id = "camera_link"
    #             marker.ns = "armor"
    #             marker.id = 0
    #             marker.action = Marker.DELETE
    #             self.pub_marker.publish(marker)

    #             # 清空机器人 Marker
    #             marker_robot = Marker()
    #             marker_robot.header.stamp = now
    #             marker_robot.header.frame_id = "camera_link"
    #             marker_robot.ns = "robot"
    #             marker_robot.id = 1
    #             marker_robot.action = Marker.DELETE
    #             self.pub_marker.publish(marker_robot)

    #             # 清空装甲板 TF
    #             t_armor = TransformStamped()
    #             t_armor.header.stamp = now
    #             t_armor.header.frame_id = "camera_link"
    #             t_armor.child_frame_id = "tracked_armor"
    #             t_armor.transform.rotation.w = 1.0
    #             self.tf_broadcaster.sendTransform(t_armor)

    #             # 清空机器人 TF
    #             t_robot = TransformStamped()
    #             t_robot.header.stamp = now
    #             t_robot.header.frame_id = "camera_link"
    #             t_robot.child_frame_id = "tracked_robot"
    #             t_robot.transform.rotation.w = 1.0
    #             self.tf_broadcaster.sendTransform(t_robot)

    #             # 清空通过机器人预测的装甲板
    #             for i in range(4):
    #                 marker_pred = Marker()
    #                 marker_pred.header.stamp = now
    #                 marker_pred.header.frame_id = "camera_link"
    #                 marker_pred.ns = "estimated_armor"
    #                 marker_pred.id = i
    #                 marker_pred.action = Marker.DELETE
    #                 self.pub_marker.publish(marker_pred)

    #                 t_pred = TransformStamped()
    #                 t_pred.header.stamp = now
    #                 t_pred.header.frame_id = "camera_link"
    #                 t_pred.child_frame_id = f"estimated_armor_{i}"
    #                 t_pred.transform.rotation.w = 1.0
    #                 self.tf_broadcaster.sendTransform(t_pred)

    #             return  # 无目标时直接返回

    #         # --- 发布装甲板位姿 ---
    #         armor_f = self.tracker.tracking_armor_filtered
    #         x, y, z = armor_f.tvec
    #         if z != 0.0:
    #             # 直接用 Object.quat（x,y,z,w）
    #             q = armor_f.quat

    #             # Pose
    #             pose_msg = PoseStamped()
    #             pose_msg.header.stamp = now
    #             pose_msg.header.frame_id = "camera_link"
    #             pose_msg.pose.position.x = float(x)
    #             pose_msg.pose.position.y = float(y)
    #             pose_msg.pose.position.z = float(z)
    #             pose_msg.pose.orientation.x = float(q[0])
    #             pose_msg.pose.orientation.y = float(q[1])
    #             pose_msg.pose.orientation.z = float(q[2])
    #             pose_msg.pose.orientation.w = float(q[3])
    #             self.pub_pose.publish(pose_msg)

    #             # TF
    #             t = TransformStamped()
    #             t.header.stamp = now
    #             t.header.frame_id = "camera_link"
    #             t.child_frame_id = "tracked_armor"
    #             t.transform.translation.x = float(x)
    #             t.transform.translation.y = float(y)
    #             t.transform.translation.z = float(z)
    #             t.transform.rotation.x = float(q[0])
    #             t.transform.rotation.y = float(q[1])
    #             t.transform.rotation.z = float(q[2])
    #             t.transform.rotation.w = float(q[3])
    #             self.tf_broadcaster.sendTransform(t)

    #             # Marker（装甲板矩形）
    #             marker = Marker()
    #             marker.header.stamp = now
    #             marker.header.frame_id = "camera_link"
    #             marker.ns = "armor"
    #             marker.id = 0
    #             marker.type = Marker.CUBE
    #             marker.action = Marker.ADD
    #             marker.pose = pose_msg.pose
    #             marker.scale.x = 0.135  # 宽
    #             marker.scale.y = 0.056  # 高
    #             marker.scale.z = 0.01   # 厚
    #             marker.color.r = 1.0
    #             marker.color.g = 1.0
    #             marker.color.b = 0.0
    #             marker.color.a = 0.6
    #             self.pub_marker.publish(marker)

    #         # --- 发布机器人位姿 ---
    #         if self.tracker.tracking_robot is None:
    #             self.get_logger().warn("机器人位姿不存在，跳过发布机器人姿态")
    #             return

    #         robot_pose = self.tracker.tracking_robot.robot
    #         x_r, y_r, z_r = robot_pose.tvec
    #         if z_r != 0.0:
    #             q_r = robot_pose.quat  # 直接四元数

    #             # Pose
    #             pose_msg_robot = PoseStamped()
    #             pose_msg_robot.header.stamp = now
    #             pose_msg_robot.header.frame_id = "camera_link"
    #             pose_msg_robot.pose.position.x = float(x_r)
    #             pose_msg_robot.pose.position.y = float(y_r)
    #             pose_msg_robot.pose.position.z = float(z_r)
    #             pose_msg_robot.pose.orientation.x = float(q_r[0])
    #             pose_msg_robot.pose.orientation.y = float(q_r[1])
    #             pose_msg_robot.pose.orientation.z = float(q_r[2])
    #             pose_msg_robot.pose.orientation.w = float(q_r[3])
    #             self.pub_pose.publish(pose_msg_robot)

    #             # TF
    #             t_robot = TransformStamped()
    #             t_robot.header.stamp = now
    #             t_robot.header.frame_id = "camera_link"
    #             t_robot.child_frame_id = "tracked_robot"
    #             t_robot.transform.translation.x = float(x_r)
    #             t_robot.transform.translation.y = float(y_r)
    #             t_robot.transform.translation.z = float(z_r)
    #             t_robot.transform.rotation.x = float(q_r[0])
    #             t_robot.transform.rotation.y = float(q_r[1])
    #             t_robot.transform.rotation.z = float(q_r[2])
    #             t_robot.transform.rotation.w = float(q_r[3])
    #             self.tf_broadcaster.sendTransform(t_robot)

    #             # Marker（机器人）
    #             marker_robot = Marker()
    #             marker_robot.header.stamp = now
    #             marker_robot.header.frame_id = "camera_link"
    #             marker_robot.ns = "robot"
    #             marker_robot.id = 1
    #             marker_robot.type = Marker.CUBE
    #             marker_robot.action = Marker.ADD
    #             marker_robot.pose = pose_msg_robot.pose
    #             marker_robot.scale.x = 0.2
    #             marker_robot.scale.y = 0.2
    #             marker_robot.scale.z = 0.2
    #             marker_robot.color.r = 0.0
    #             marker_robot.color.g = 1.0
    #             marker_robot.color.b = 0.0
    #             marker_robot.color.a = 0.6
    #             self.pub_marker.publish(marker_robot)

    #             # --- 发布通过机器人计算出的装甲板位姿 ---
    #             robot = self.tracker.tracking_robot
    #             for i, armor in enumerate((robot.armor_0, robot.armor_1, robot.armor_2, robot.armor_3)):
    #                 self._publish_estimated_armor(armor, now, i)

    #     except Exception as e:
    #         self.get_logger().warn(f"[RViz2] 发布机器人姿态失败: {e}")


    # def _publish_estimated_armor(self, armor, now, i):
    #     x, y, z = armor.tvec
    #     rx, ry, rz = armor.rvec

    #     if z != 0.0:
    #         # 计算旋转矩阵，并将其转为四元数
    #         R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=float))
    #         q = tf_transformations.quaternion_from_matrix(
    #             np.vstack((np.hstack((R, np.zeros((3, 1)))), [0, 0, 0, 1]))
    #         )

    #         # 发布装甲板 Pose
    #         pose_msg = PoseStamped()
    #         pose_msg.header.stamp = now
    #         pose_msg.header.frame_id = "camera_link"
    #         pose_msg.pose.position.x = float(x)
    #         pose_msg.pose.position.y = float(y)
    #         pose_msg.pose.position.z = float(z)
    #         pose_msg.pose.orientation.x = float(q[0])
    #         pose_msg.pose.orientation.y = float(q[1])
    #         pose_msg.pose.orientation.z = float(q[2])
    #         pose_msg.pose.orientation.w = float(q[3])
    #         self.pub_pose.publish(pose_msg)

    #         # 发布装甲板的 TF
    #         t = TransformStamped()
    #         t.header.stamp = now
    #         t.header.frame_id = "camera_link"
    #         t.child_frame_id = f"estimated_armor_{i}"
    #         t.transform.translation.x = float(x)
    #         t.transform.translation.y = float(y)
    #         t.transform.translation.z = float(z)
    #         t.transform.rotation.x = float(q[0])
    #         t.transform.rotation.y = float(q[1])
    #         t.transform.rotation.z = float(q[2])
    #         t.transform.rotation.w = float(q[3])
    #         self.tf_broadcaster.sendTransform(t)

    #         # 发布装甲板的 Marker（矩形装甲板）
    #         marker = Marker()
    #         marker.header.stamp = now
    #         marker.header.frame_id = "camera_link"
    #         marker.ns = "estimated_armor"
    #         marker.id = i  # 装甲板唯一标识
    #         marker.type = Marker.CUBE   # 立方体表示装甲板
    #         marker.action = Marker.ADD

    #         marker.pose = pose_msg.pose
    #         marker.scale.x = 0.135  # 装甲板宽度 (m)
    #         marker.scale.y = 0.056  # 装甲板高度 (m)
    #         marker.scale.z = 0.01   # 装甲板厚度 (m)

    #         marker.color.r = 1.0
    #         marker.color.g = 0.0  # 装甲板是黄色的
    #         marker.color.b = 1.0
    #         marker.color.a = 0.6  # 半透明

    #         self.pub_marker.publish(marker)

    # ---------- Armors 回调 ----------
    def _cb_armors(self, msg: ArmorsCppMsg):
        self.target_pub_to_node(msg)
        # self.pub_to_rviz()

    # ---------- 决策回调 ----------
    def _cb_decision(self, msg: Decision):
        try:
            follow = int(self.get_parameter('follow_decision').value)
            if follow != 1:
                return
            if hasattr(msg, 'color'):
                cur = int(self.get_parameter('tracking_color').value)
                if msg.color != cur:
                    self.set_parameters([rclpy.parameter.Parameter(
                        'tracking_color',
                        rclpy.parameter.Parameter.Type.INTEGER,
                        int(msg.color)
                    )])
                    self.get_logger().info(f"追踪颜色切换为 {int(msg.color)}")
        except Exception as e:
            self.get_logger().error(f"处理 Decision 异常：{e}")

    def _cb_res_img(self, msg):
        if not self.dis_res:
            return
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 调用 tracker 的绘制函数
        cv_img = self.tracker.display_result(cv_img)

        # 发布结果图像
        img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
        img_msg.header = msg.header
        self.pub_res_img.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmorTrackerNode("armor_tracker_node")
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
