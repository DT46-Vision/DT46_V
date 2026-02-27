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

    def rotate_pose_axis(self, raw_rpy, rotation_rpy, order='xyz'):
        """
        初始化 IMU 姿态
        参数:
        raw_rpy      : [rx, ry, rz] 当前的姿态 (欧拉角)
        rotation_rpy : [rx, ry, rz] 需要叠加的旋转/修正 (欧拉角)
        order        : 旋转顺序，默认 'xyz'
        """

        # 1. 把“当前的姿态”变成旋转对象 R1
        r_current = R.from_euler(order, raw_rpy, degrees=True)

        # 2. 把“修正的旋转”变成旋转对象 R2
        r_fix = R.from_euler(order, rotation_rpy, degrees=True)

        # 3. 姿态叠加 (矩阵乘法)
        # 逻辑：Final = Fix * Current (相当于在当前姿态的基础上，叠加一个修正)
        # 注意：乘法顺序很重要。
        # 如果 rotation_rpy 是“世界坐标系下的修正”，放在左边 (Fix * Current)
        # 如果 rotation_rpy 是“自身坐标系下的修正”，放在右边 (Current * Fix)
        # 针对你的 [90] + [-90] -> [0] 的需求，在这个简单 Z 轴场景下左右乘结果一样，
        # 但通常坐标系变换（父子关系）是左乘。
        r_final = r_current * r_fix

        # 4. 变回欧拉角
        fixed_rpy = r_final.as_euler(order, degrees=True)

        # 如果是第一帧数据，记录初始 Yaw 作为基准
        if self.yaw is None:
            self.yaw = fixed_rpy[2]

        # 计算相对角度
        relative_yaw = fixed_rpy[2] - self.yaw

        # 【核心修改】归一化到 [-180, 180]
        # 公式逻辑：先加180，对360取模，再减180
        fixed_rpy[2] = (relative_yaw + 180) % 360 - 180

        return fixed_rpy

    def rotate_pos_axis(self, raw_xyz, rotation_rpy, order='xyz'):
        """
        参数:
        raw_xyz      : [x, y, z] 当前在相机坐标系下的位移 (tvec)
        rotation_rpy : [rx, ry, rz] 旋转修正量 (欧拉角)
        order        : 旋转顺序，默认 'xyz'

        返回:
        rotated_xyz  : 旋转后的 [x, y, z] 坐标
        """
        # 1. 把“修正的旋转”变成旋转对象
        # 这代表了坐标系之间的旋转关系（例如从相机坐标系转到云台坐标系）
        r_fix = R.from_euler(order, rotation_rpy, degrees=True)

        # 2. 将输入转换为 numpy 数组以进行矩阵运算
        pos_vec = np.array(raw_xyz)

        # 3. 执行旋转变换
        # .apply() 会自动处理旋转矩阵与向量的乘法
        rotated_xyz = r_fix.apply(pos_vec)

        # 4. 转换回列表或保持 numpy 格式返回
        return rotated_xyz.tolist()

    # 【新增】3D -> 2D 投影函数
    def project_point(self, xyz_cam) -> Tuple[Tuple[int, int], bool]:
        """
        将相机系 3D 坐标投影到 2D 像素坐标
        :param xyz_cam: [x, y, z] mm (相机坐标系下)
        :return: ((u, v), is_valid)
        """
        if not self.has_camera_info or xyz_cam is None:
            return (0, 0), False

        # 确保输入是 float64 numpy 数组
        tvec = np.array(xyz_cam, dtype=np.float64).reshape(3, 1)

        # 假设点相对于相机只有位移，没有旋转 (rvec 为 0)
        # 因为 tvec 已经是点在相机坐标系下的位置了
        rvec = np.zeros((3, 1), dtype=np.float64)

        try:
            # OpenCV 投影函数
            # 参数说明：
            # 1. objectPoints: 3D点数组。这里设为(0,0,0)，因为我们用 tvec 来表示点的位置
            # 2. rvec: 旋转向量 (0)
            # 3. tvec: 平移向量 (即点的相机坐标)
            # 4. cameraMatrix: 内参矩阵 K
            # 5. distCoeffs: 畸变系数 <--- 【关键点：这里必须传进来】
            points_2d, _ = cv2.projectPoints(
                np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs  # <--- 加上这个参数，OpenCV 就会自动处理畸变
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
    print(tf.imu_2_cam(raw_rpy = imu_rpy, rotation_rpy = cam_fix, order = 'xyz'))
