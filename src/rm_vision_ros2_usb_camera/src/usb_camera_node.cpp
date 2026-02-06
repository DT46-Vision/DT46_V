#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <linux/videodev2.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>

class USBCameraNode : public rclcpp::Node
{
public:
    USBCameraNode() : Node("usb_camera_node")
    {
        // ===== 参数声明 =====
        this->declare_parameter<int>("camera_index", 0);
        this->declare_parameter<bool>("exposure_auto", true);
        this->declare_parameter<int>("exposure_time", 800);
        this->declare_parameter<double>("fps", 120.0);
        this->declare_parameter<int>("gain", 128);
        this->declare_parameter<int>("frame_width", 1920);
        this->declare_parameter<int>("frame_height", 1080);
        this->declare_parameter<std::string>("camera_name", "usb_camera");
        this->declare_parameter<std::string>("camera_info_url", "");

        // 分辨率只读一次（不可动态修改）
        this->get_parameter_or("frame_width", frame_width_, 1920);
        this->get_parameter_or("frame_height", frame_height_, 1080);

        // 其他参数（允许动态修改）
        this->get_parameter_or("camera_index", camera_index_, 0);
        this->get_parameter_or("fps", fps_, 120.0);
        this->get_parameter_or("exposure_auto", exposure_auto_, true);
        this->get_parameter_or("exposure_time", exposure_time_, 800);
        this->get_parameter_or("gain", gain_, 128);
        this->get_parameter_or("camera_name", camera_name_, std::string("usb_camera"));
        this->get_parameter_or("camera_info_url", camera_info_url_, std::string(""));

        // ===== CameraInfoManager =====
        cinfo_mgr_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name_, camera_info_url_);
        if (!camera_info_url_.empty()) {
            if (cinfo_mgr_->loadCameraInfo(camera_info_url_)) {
                RCLCPP_INFO(this->get_logger(), "加载相机标定文件: %s", camera_info_url_.c_str());
            } else {
                RCLCPP_WARN(this->get_logger(), "加载相机标定文件失败: %s", camera_info_url_.c_str());
            }
        }

        // ===== V4L2 硬设置 =====
        configure_v4l2();

        // 打开相机
        cap_.open(camera_index_, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            RCLCPP_FATAL(this->get_logger(), "无法打开摄像头 %d", camera_index_);
            throw std::runtime_error("Camera open failed");
        }

        // 强制 OpenCV 使用 MJPG + FPS
        cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
        cap_.set(cv::CAP_PROP_FPS, fps_);

        // 打印实际参数
        log_camera_info("OpenCV 初始化后");

        // 同步 fps 到相机实际值
        fps_ = cap_.get(cv::CAP_PROP_FPS);

        // 应用曝光和增益
        apply_exposure();
        apply_gain();

        // 发布图像 & camera_info
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
        cinfo_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 10);

        // 定时器：按 fps 定时抓取图像
        create_timer();

        // 注册动态参数回调
        callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&USBCameraNode::parameters_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(),
            "相机确认工作参数: %dx%d @ %.2f FPS, 曝光模式: %s, 曝光时间: %d, 增益: %d",
            frame_width_, frame_height_, fps_,
            exposure_auto_ ? "自动" : "手动",
            exposure_time_, gain_);
    }

private:
    // ========== V4L2 配置 ==========
    void configure_v4l2()
    {
        std::string device = "/dev/video" + std::to_string(camera_index_);
        int fd = open(device.c_str(), O_RDWR);
        if (fd < 0) {
            RCLCPP_ERROR(this->get_logger(), "无法打开设备 %s", device.c_str());
            return;
        }

        struct v4l2_format fmt;
        std::memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = frame_width_;
        fmt.fmt.pix.height = frame_height_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            RCLCPP_WARN(this->get_logger(), "V4L2 设置分辨率/格式失败");
        } else {
            RCLCPP_INFO(this->get_logger(),
                        "V4L2 设置分辨率成功: %dx%d, 格式=MJPG",
                        fmt.fmt.pix.width, fmt.fmt.pix.height);
        }

        struct v4l2_streamparm parm;
        std::memset(&parm, 0, sizeof(parm));
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = static_cast<unsigned int>(fps_);
        if (ioctl(fd, VIDIOC_S_PARM, &parm) < 0) {
            RCLCPP_WARN(this->get_logger(), "V4L2 设置帧率失败");
        } else {
            double actual_fps =
                (double)parm.parm.capture.timeperframe.denominator /
                (double)parm.parm.capture.timeperframe.numerator;
            RCLCPP_INFO(this->get_logger(), "V4L2 设置帧率成功: %.2f fps", actual_fps);
        }

        close(fd);
    }

    // ========== V4L2 控制 ==========
    void set_v4l2_control(int control_id, int value)
    {
        std::string device = "/dev/video" + std::to_string(camera_index_);
        int fd = open(device.c_str(), O_RDWR);
        if (fd < 0) {
            RCLCPP_ERROR(this->get_logger(), "无法打开设备 %s", device.c_str());
            return;
        }

        struct v4l2_control ctrl;
        ctrl.id = control_id;
        ctrl.value = value;
        if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
            RCLCPP_WARN(this->get_logger(), "设置 V4L2 控件失败: %d", control_id);
        } else {
            RCLCPP_INFO(this->get_logger(), "V4L2 控件 %d 设置为 %d", control_id, value);
        }

        close(fd);
    }

    // ========== 曝光设置 ==========
    void apply_exposure()
    {
        if (exposure_auto_) {
            if (!cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3)) {
                set_v4l2_control(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_AUTO);
            }
        } else {
            if (!cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1)) {
                set_v4l2_control(V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL);
            }
            if (!cap_.set(cv::CAP_PROP_EXPOSURE, exposure_time_)) {
                set_v4l2_control(V4L2_CID_EXPOSURE_ABSOLUTE, exposure_time_);
            }
        }
    }

    // ========== 增益设置 ==========
    void apply_gain()
    {
        if (!cap_.set(cv::CAP_PROP_GAIN, gain_)) {
            set_v4l2_control(V4L2_CID_GAIN, gain_);
        }
    }

    // ========== 参数动态更新 ==========
    rcl_interfaces::msg::SetParametersResult
    parameters_callback(const std::vector<rclcpp::Parameter> &params)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;

        for (const auto &param : params) {
            if (param.get_name() == "fps") {
                double req_fps = param.as_double();
                cap_.set(cv::CAP_PROP_FPS, req_fps);
                double actual = cap_.get(cv::CAP_PROP_FPS);
                if (std::abs(actual - req_fps) > 1e-2) {
                    RCLCPP_WARN(this->get_logger(),
                        "请求 fps=%.2f 不支持，回退到 %.2f", req_fps, actual);
                }
                fps_ = actual;
                create_timer();
                RCLCPP_INFO(this->get_logger(), "更新 fps = %.2f", fps_);
            }
            else if (param.get_name() == "exposure_auto") {
                exposure_auto_ = param.as_bool();
                apply_exposure();
                RCLCPP_INFO(this->get_logger(), "更新曝光模式 = %s",
                            exposure_auto_ ? "自动" : "手动");
            }
            else if (param.get_name() == "exposure_time") {
                exposure_time_ = param.as_int();
                if (!exposure_auto_) {
                    apply_exposure();
                    RCLCPP_INFO(this->get_logger(), "更新曝光时间 = %d", exposure_time_);
                }
            }
            else if (param.get_name() == "gain") {
                gain_ = param.as_int();
                apply_gain();
                RCLCPP_INFO(this->get_logger(), "更新增益 = %d", gain_);
            }
        }
        return result;
    }

    // ========== 打印相机状态 ==========
    void log_camera_info(const std::string &prefix)
    {
        double rw = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
        double rh = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
        double rfps = cap_.get(cv::CAP_PROP_FPS);
        int fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));
        char fourcc_str[] = {
            static_cast<char>(fourcc & 0XFF),
            static_cast<char>((fourcc >> 8) & 0XFF),
            static_cast<char>((fourcc >> 16) & 0XFF),
            static_cast<char>((fourcc >> 24) & 0XFF),
            0
        };
        RCLCPP_INFO(this->get_logger(),
            "%s: %dx%d @ %.2f fps, 格式=%s",
            prefix.c_str(), (int)rw, (int)rh, rfps, fourcc_str);
    }

    // ========== 采帧 ==========
    void capture_frame()
    {
        cv::Mat frame;
        if (!cap_.read(frame)) {
            RCLCPP_WARN(this->get_logger(), "捕获帧失败");
            return;
        }

        auto stamp = this->now();
        std_msgs::msg::Header header;
        header.stamp = stamp;
        header.frame_id = "camera_optical_frame";

        auto img_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
        image_pub_->publish(*img_msg);

        auto ci = cinfo_mgr_->getCameraInfo();
        ci.header = header;
        cinfo_pub_->publish(ci);
    }

    // ========== 定时器管理 ==========
    void create_timer()
    {
        auto period = std::chrono::milliseconds(static_cast<int>(1000.0 / fps_));
        timer_ = this->create_wall_timer(period, std::bind(&USBCameraNode::capture_frame, this));
    }

    // 成员
    cv::VideoCapture cap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cinfo_pub_;
    std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_mgr_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_handle_;

    int camera_index_;
    int frame_width_;
    int frame_height_;
    double fps_;
    bool exposure_auto_;
    int exposure_time_;
    int gain_;
    std::string camera_name_;
    std::string camera_info_url_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<USBCameraNode>());
    rclcpp::shutdown();
    return 0;
}
