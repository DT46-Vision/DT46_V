from typing import Tuple
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

class RmTF:
    def __init__(self):
        self.yaw = None
        # 【新增】相机内参初始化 (默认为单位矩阵，防止未设置时报错)
        self.camera_matrix = np.eye(3, dtype=np.float64)
        self.dist_coeffs = np.zeros((1, 5), dtype=np.float64)
        self.has_camera_info = False
        self.cx = 0
        self.cy = 0

        # 缓存 project_point 需要的常量，避免高频创建
        self._zero_rvec = np.zeros((3, 1), dtype=np.float64)
        self._origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        # ==========================================
        # 【新增】预计算并缓存高频使用的静态旋转矩阵
        # ==========================================
        self._cam_to_imu_mat = self._euler_to_mat_xyz_deg([-90.0, 0.0, -90.0])
        self._imu_to_cam_mat = self._euler_to_mat_zyx_deg([90.0, 0.0, 90.0])
    # 【修改】增加 width 和 height 参数
    def set_camera_info(self, camera_matrix, dist_coeffs, width=640, height=480):
        """
        设置相机内参，并强制构建“理想光心”
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self.has_camera_info = True

        # ==========================================
        # 核心修改：在这里做“光心矫正” (Virtual Ideal Camera)
        # ==========================================
        # 强制把 K 矩阵里的光心 (cx, cy) 改成图像几何中心
        # 这样 cv2.projectPoints 算出来的点，(0,0,Z) 永远在画面正中央
        ideal_cx = width / 2.0
        ideal_cy = height / 2.0

        self.camera_matrix[0, 2] = ideal_cx
        self.camera_matrix[1, 2] = ideal_cy

        # 更新类属性，方便调试查看
        self.cx = ideal_cx
        self.cy = ideal_cy

        # print(f"【RmTF】已构建虚拟理想相机: cx={self.cx}, cy={self.cy}")
    # ---------------- 新增：纯 Numpy 矩阵生成方法 ----------------
    def _euler_to_mat_xyz_deg(self, rpy):
        """轻量级：欧拉角 (度) 转旋转矩阵 (XYZ顺序)"""
        r, p, y = np.radians(rpy)
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        # XYZ 外旋顺序矩阵连乘
        return Rz @ Ry @ Rx

    def _euler_to_mat_zyx_deg(self, rpy):
        """轻量级：欧拉角 (度) 转旋转矩阵 (ZYX顺序)"""
        # 对于 zyx 顺序，传入的三个元素分别对应 Z, Y, X 轴的旋转角度
        ang_z, ang_y, ang_x = np.radians(rpy)
        
        Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], 
                       [np.sin(ang_z), np.cos(ang_z), 0], 
                       [0, 0, 1]])
                       
        Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], 
                       [0, 1, 0], 
                       [-np.sin(ang_y), 0, np.cos(ang_y)]])
                       
        Rx = np.array([[1, 0, 0], 
                       [0, np.cos(ang_x), -np.sin(ang_x)], 
                       [0, np.sin(ang_x), np.cos(ang_x)]])
                       
        # ZYX 外旋顺序矩阵连乘
        return Rx @ Ry @ Rz
    # -------------------------------------------------------------
    def rotate_pose_axis(self, raw_rpy, rotation_rpy, order='xyz'):
        """
        初始化 IMU 姿态
        参数:
        raw_rpy      : [rx, ry, rz] 当前的姿态 (欧拉角)
        rotation_rpy : [rx, ry, rz] 需要叠加的旋转/修正 (欧拉角)
        order        : 旋转顺序，默认 'xyz'
        """
        # 增加数据有效性校验，防止 NaN 或 Inf 导致四元数异常崩溃
        if np.any(np.isnan(raw_rpy)) or np.any(np.isinf(raw_rpy)) or \
        np.any(np.isnan(rotation_rpy)) or np.any(np.isinf(rotation_rpy)):
            # 遇到无效数据时直接返回，或者根据节点逻辑返回 None / 上一帧数据
            # self.get_logger().warn('接收到无效的 IMU RPY 数据，已跳过处理。')
            return raw_rpy 

        # 1. 把“当前的姿态”变成旋转对象 R1
        r_current = R.from_euler(order, raw_rpy, degrees=True)

        # 2. 把“修正的旋转”变成旋转对象 R2
        r_fix = R.from_euler(order, rotation_rpy, degrees=True)

        # 3. 姿态叠加 (矩阵乘法)
        r_final = r_current * r_fix

        # 4. 变回欧拉角
        fixed_rpy = r_final.as_euler(order, degrees=True)

        # 如果是第一帧数据，记录初始 Yaw 作为基准
        if self.yaw is None:
            self.yaw = fixed_rpy[2]

        # 计算相对角度
        relative_yaw = fixed_rpy[2] - self.yaw

        # 归一化到 [-180, 180]
        fixed_rpy[2] = (relative_yaw + 180) % 360 - 180

        return fixed_rpy

    def rotate_pos_axis(self, raw_xyz, rotation_rpy, order='xyz'):
        """
        使用纯 Numpy 矩阵乘法替换 SciPy Rotation，大幅降低高频调用开销

        参数:
        raw_xyz      : [x, y, z] 当前在相机坐标系下的位移 (tvec)
        rotation_rpy : [rx, ry, rz] 旋转修正量 (欧拉角)
        order        : 旋转顺序，默认 'xyz'

        返回:
        rotated_xyz  : 旋转后的 [x, y, z] 坐标
        """
        pos_vec = np.array(raw_xyz, dtype=np.float64)

        # 防御性编程：强制转为 list 比较，防止传入 numpy array 导致多维真值计算报错
        rpy_list = list(rotation_rpy)
        
        # 1. 拦截高频静态变换
        if order == 'xyz' and rpy_list == [-90.0, 0.0, -90.0]:
            rot_mat = self._cam_to_imu_mat
        elif order == 'zyx' and rpy_list == [90.0, 0.0, 90.0]:
            rot_mat = self._imu_to_cam_mat
            
        # 2. 动态变换（如每帧的 IMU 实时角度），走轻量级 Numpy 计算
        else:
            if order == 'xyz':
                rot_mat = self._euler_to_mat_xyz_deg(rotation_rpy)
            elif order == 'zyx':
                rot_mat = self._euler_to_mat_zyx_deg(rotation_rpy)
            else:
                # 兜底：处理未优化的旋转顺序
                r_fix = R.from_euler(order, rotation_rpy, degrees=True)
                return r_fix.apply(pos_vec).tolist()

        # 3. 矩阵点乘并返回列表格式，兼容原有逻辑
        return rot_mat.dot(pos_vec).tolist()

    # 【新增】3D -> 2D 投影函数
    def project_point(self, xyz_cam) -> Tuple[Tuple[int, int], bool]:
        """
        将相机系 3D 坐标投影到 2D 像素坐标
        :param xyz_cam: [x, y, z] mm (相机坐标系下)
        :return: ((u, v), is_valid)
        """
        if not self.has_camera_info or xyz_cam is None:
            return (0, 0), False

        # 【新增】：如果点在相机光心后方 (Z <= 0)，物理上不可见，直接丢弃
        # 相机坐标系标准是 Z 轴朝前。给个极小值 1e-3 防止除零风险
        if xyz_cam[2] <= 1e-3:
            return (0, 0), False
        
        # 确保输入是 float64 numpy 数组
        tvec = np.array(xyz_cam, dtype=np.float64).reshape(3, 1)

        try:
            # OpenCV 投影函数
            # 参数说明：
            # 1. objectPoints: 3D点数组。使用初始化的 self._origin_3d
            # 2. rvec: 旋转向量 self._zero_rvec
            # 3. tvec: 平移向量 (即点的相机坐标)
            # 4. cameraMatrix: 内参矩阵 K
            # 5. distCoeffs: 畸变系数
            points_2d, _ = cv2.projectPoints(
                self._origin_3d,
                self._zero_rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs 
            )

            u = int(points_2d[0][0][0])
            v = int(points_2d[0][0][1])

            # 简单的范围保护 (防止数值爆炸导致绘图报错)
            # 这里设一个比较宽的范围，具体的视场范围判定交给使用者
            if u < -10000 or u > 10000 or v < -10000 or v > 10000:
                return (u, v), False

            return (u, v), True

        except Exception as e:
            print(f"Project point error: {e}")
            return (0, 0), False

if __name__ == '__main__':
    tf = RmTF()
    imu_rpy = [0, 0, 90]
    cam_fix = [0, 0, 180]
    print(tf.rotate_pose_axis(raw_rpy = imu_rpy, rotation_rpy = cam_fix, order = 'xyz'))