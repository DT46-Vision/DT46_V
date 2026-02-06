#include "rm_detector/pnp.hpp"

namespace DT46_VISION {

    // 解析相机内参矩阵和畸变系数的函数
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

    // 根据物体尺寸ID选择物体3D坐标
    std::vector<cv::Point3f> PNP::getObjectPoints(int object_size) {
        if (object_size == 6 || object_size == 0) {
            // 大装甲板 (134mm x 58mm)，只保留四个角点
            return {
                {-66.0f, -29.7f, 8.0f},  // 左上
                {-66.0f,  29.7f, 8.0f},  // 左下
                { 66.0f, -29.7f, 8.0f},  // 右上
                { 66.0f,  29.7f, 8.0f}   // 右下
            };
        } else {
            // 小装甲板 (134mm x 58mm)，只保留四个角点
            return {
                {-66.0f, -29.7f, 8.0f},  // 左上
                {-66.0f,  29.7f, 8.0f},  // 左下
                { 66.0f, -29.7f, 8.0f},  // 右上
                { 66.0f,  29.7f, 8.0f}   // 右下
            };
        }
    }

    // 处理装甲板灯点坐标并返回 (dx, dy, dz)
    std::tuple<double, double, double, double, double, double> PNP::processArmorCorners(
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
            return {0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
        }

        // 获取相机参数
        cv::Mat cameraMatrix, distCoeffs;

        // 尝试解析相机信息
        if (!parseCameraInfo(cam_info, cameraMatrix, distCoeffs)) {
            // 解析失败，或者没有接收到 camera_info ，使用默认 USB 相机参数
            cameraMatrix = (cv::Mat_<double>(3, 3) <<
                1320.127401, 0.0, 609.90294,
                0.0, 1329.050651, 457.308236,
                0.0, 0.0, 1.0);
            distCoeffs = (cv::Mat_<double>(1, 5) <<
                -0.034135, 0.131210, -0.015866, -0.004433, 0.0);

            RCLCPP_WARN(logger_, "no cam_info");
        }

        std::vector<cv::Point3f> objectPoints = getObjectPoints(class_id); // 物体3D坐标

        // 如果要强制几何中心当光心，修改cx,cy
        if (use_geometric_center) {
            int W = frame.cols;
            int H = frame.rows;
            cameraMatrix.at<double>(0,2) = W / 2.0;  // 替换cx
            cameraMatrix.at<double>(1,2) = H / 2.0;  // 替换cy
            // RCLCPP_WARN(logger_, "使用几何中心 (%.1f, %.1f) 替代光心", W/2.0, H/2.0);
        }

        // PnP 解算
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(objectPoints, imagePoints,
                                    cameraMatrix, distCoeffs,
                                    rvec, tvec);

        if (!success) {
            RCLCPP_WARN(logger_, "solvePnP失败，无法估计位姿");
            return {0.0, 0.0, -1.0, 0.0, 0.0, 0.0};
        }
        // --- BA 矫正 (LM refine) ---
        try {
            cv::solvePnPRefineLM(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distCoeffs,
                rvec,
                tvec,
                cv::TermCriteria(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                    20,   // 最大迭代次数
                    1e-6  // 收敛阈值
                )
            );
        } catch (const cv::Exception &e) {
            RCLCPP_WARN(logger_, "solvePnPRefineLM 异常: %s", e.what());
        }

        // 转成 double
        double dx = tvec.at<double>(0);
        double dy = tvec.at<double>(1); // y 轴方向在 OpenCV 是反的
        double dz = tvec.at<double>(2);

        double rx = rvec.at<double>(0);
        double ry = rvec.at<double>(1); // y 轴方向在 OpenCV 是反的
        double rz = rvec.at<double>(2);

        return {dx, dy, dz, rx, ry, rz};
    }

}
