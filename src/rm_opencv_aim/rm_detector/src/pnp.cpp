#include "rm_detector/pnp.hpp"

namespace DT46_VISION {

    // 解析相机内参矩阵和畸变系数
    bool PNP::parseCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr& msg, cv::Mat& K, cv::Mat& D) {
        if (msg == nullptr) {
            return false;
        }

        // 内参矩阵 K (3x3)
        K = (cv::Mat_<double>(3, 3) <<
            msg->k[0], msg->k[1], msg->k[2],
            msg->k[3], msg->k[4], msg->k[5],
            msg->k[6], msg->k[7], msg->k[8]);

        // 畸变系数矩阵 D (1x5)
        D = (cv::Mat_<double>(1, 5) <<
            msg->d[0], msg->d[1], msg->d[2], msg->d[3], msg->d[4]);

        return true;
    }

    // 根据物体尺寸ID选择物体3D坐标 (单位: mm)
    std::vector<cv::Point3f> PNP::getObjectPoints(int object_size) {
        // 定义灯条和装甲板的物理尺寸
        // 坐标系定义: X右, Y下, Z前 (物体平面)
        // 确保这里的顺序与 imagePoints (左上, 左下, 右上, 右下) 一一对应
        if (object_size == 6 || object_size == 0) {
            // 大装甲板 (225mm x 58mm) (修正了Y的半高为29)
            return {
                {-112.5f, -29.0f, 0.0f},  // 0: 左上
                {-112.5f,  29.0f, 0.0f},  // 1: 左下
                { 112.5f, -29.0f, 0.0f},  // 2: 右上
                { 112.5f,  29.0f, 0.0f}   // 3: 右下
            };
        } else {
            // 小装甲板 (135mm x 58mm)
            return {
                {-67.5f, -29.0f, 0.0f},  // 0: 左上
                {-67.5f,  29.0f, 0.0f},  // 1: 左下
                { 67.5f, -29.0f, 0.0f},  // 2: 右上
                { 67.5f,  29.0f, 0.0f}   // 3: 右下
            };
        }
    }

    // 处理装甲板灯点坐标并返回 (dx, dy, dz)
    std::tuple<double, double, double, double> PNP::processArmorCorners(
        const sensor_msgs::msg::CameraInfo::SharedPtr& cam_info,
        bool use_geometric_center, // 是否用图像几何中心替代标定光心
        const cv::Mat& frame,
        const Armor& armor,
        int class_id
        ) {
        // 提取四个灯条端点
        std::vector<cv::Point2f> imagePoints = {
            armor.light1_up,
            armor.light1_down,
            armor.light2_up,
            armor.light2_down
        };

        if (imagePoints.size() != 4) {
            RCLCPP_ERROR(logger_, "装甲板灯点数量无效: %zu，预期为4", imagePoints.size());
            return {0.0, 0.0, -1.0, 0.0};
        }

        // 获取相机参数
        cv::Mat cameraMatrix, distCoeffs;
        if (!parseCameraInfo(cam_info, cameraMatrix, distCoeffs)) {
            // 默认参数 (如果没有收到 camera_info，请务必校准这里)
            // 这里的默认值可能会导致巨大的误差，请确认日志中是否有 "no cam_info"
            cameraMatrix = (cv::Mat_<double>(3, 3) <<
                1320.127401, 0.0, 609.90294,
                0.0, 1329.050651, 457.308236,
                0.0, 0.0, 1.0);
            distCoeffs = (cv::Mat_<double>(1, 5) <<
                -0.034135, 0.131210, -0.015866, -0.004433, 0.0);

            RCLCPP_WARN(logger_, "no cam_info");
        }

        // 强制使用图像几何中心 (仅在特定裁切策略下使用)
        if (use_geometric_center) {
            cameraMatrix.at<double>(0,2) = frame.cols / 2.0;
            cameraMatrix.at<double>(1,2) = frame.rows / 2.0;
        }

        std::vector<cv::Point3f> objectPoints = getObjectPoints(class_id);

        // 3. 稳健的 PnP 解算 (SQPNP 对平面目标的抗模糊性更好)
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, imagePoints,
                                    cameraMatrix, distCoeffs,
                                    rvec, tvec,
                                    false,
                                    cv::SOLVEPNP_IPPE); // 【修正1】换用 SQPNP

        if (!success) {
            RCLCPP_WARN(logger_, "solvePnP失败，无法估计位姿");
            return {0.0, 0.0, -1.0, 0.0};
        }

        double dx = tvec.at<double>(0);
        double dy = tvec.at<double>(1);
        double dz = tvec.at<double>(2);

        // 4. 计算绝对 Yaw
        cv::Mat R;
        cv::Rodrigues(rvec, R);

        double r13 = R.at<double>(0, 2);
        double r33 = R.at<double>(2, 2);
        double pnp_yaw_rad = atan2(r13, r33); // 此时是绝对角度

        // 归一化逻辑 (正对时 yaw 为 0)
        double normalized_yaw = pnp_yaw_rad + CV_PI;

        // 约束到 [-PI, PI]
        if (normalized_yaw > CV_PI) normalized_yaw -= 2 * CV_PI;
        if (normalized_yaw < -CV_PI) normalized_yaw += 2 * CV_PI;
        // -------------------------------------------------------------

        double yaw_deg = normalized_yaw * 180.0 / CV_PI;

        return {dx, dy, dz, yaw_deg};
    }
}
