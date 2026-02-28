#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <condition_variable> // 引入

#include "rm_detector/detector.hpp"
#include "rm_detector/pnp.hpp"
#include "rm_detector/classifier.hpp"
#include "rm_interfaces/msg/armor_info.hpp"
#include "rm_interfaces/msg/armors_msg.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

using namespace std::chrono;
using namespace cv;
using namespace std;

namespace DT46_VISION {

    class ArmorDetectorNode : public rclcpp::Node {
        
    public:
        ArmorDetectorNode() : Node("rm_detector") {
            // ---------------- 参数声明 ----------------
            // 先声明所有参数（从 detector_params.yaml 复制默认值）
            this->declare_parameter<std::string>("cls_model_file", "mlp.onnx");

            // roi 裁剪参数
            this->declare_parameter<bool>("roi_crop", false);
            this->declare_parameter<double>("roi_scale", 0.5);
            
            // 灯条过滤参数
            this->declare_parameter<int>("light_area_min", 5);
            this->declare_parameter<double>("light_h_w_ratio", 5.0);
            this->declare_parameter<int>("light_angle_min", -35);
            this->declare_parameter<int>("light_angle_max", 35);
            this->declare_parameter<double>("light_red_ratio", 2.0);
            this->declare_parameter<double>("light_blue_ratio", 2.0);

            // 装甲板匹配参数
            this->declare_parameter<double>("height_rate_tol", 1.3);
            this->declare_parameter<double>("height_multiplier_min", 1.8);
            this->declare_parameter<double>("height_multiplier_max", 3.0);

            // 图像处理参数
            this->declare_parameter<int>("binary_val", 120);
            this->declare_parameter<int>("detect_color", 2);
            this->declare_parameter<bool>("display_mode", false);

            // 其他参数
            this->declare_parameter<bool>("use_geometric_center", true);
            this->declare_parameter<int>("print_period_ms", 1000); // ms

            // ---------------- 参数读取 ----------------
            std::string cls_model_file   = get_required_param<std::string>("cls_model_file");

            // 拼接模型路径
            std::string pkg_share = ament_index_cpp::get_package_share_directory("rm_detector");
            cls_model_path_ = (std::filesystem::path(pkg_share) / "model" / cls_model_file).string();

            RCLCPP_INFO(this->get_logger(), "Classifier model file: %s", cls_model_file.c_str());
            RCLCPP_INFO(this->get_logger(), "Classifier model path: %s", cls_model_path_.c_str());

            Params params = {
                get_required_param<int>("light_area_min"),
                get_required_param<double>("light_h_w_ratio"),
                get_required_param<int>("light_angle_min"),
                get_required_param<int>("light_angle_max"),
                get_required_param<double>("light_red_ratio"),
                get_required_param<double>("light_blue_ratio"),
                get_required_param<double>("height_rate_tol"),
                get_required_param<double>("height_multiplier_min"),
                get_required_param<double>("height_multiplier_max"),
                get_required_param<bool>("roi_crop"),
                get_required_param<double>("roi_scale")
            };
            
            detect_color_        = get_required_param<int>("detect_color");
            display_mode_        = get_required_param<bool>("display_mode");
            binary_val_          = get_required_param<int>("binary_val");
            use_geometric_center_= get_required_param<bool>("use_geometric_center");
            print_period_ms_.store(get_required_param<int>("print_period_ms")); // 多线程访问
            
            // ---------------- Detector 初始化 ----------------
            detector_ = std::make_shared<ArmorDetector>(detect_color_, display_mode_, binary_val_, params);
            pnp_      = std::make_shared<PNP>(this->get_logger());

            reload_classifier_impl_(cls_model_path_);

            // 动态参数回调
            callback_handle_ = this->add_on_set_parameters_callback(
                std::bind(&ArmorDetectorNode::parameters_callback, this, std::placeholders::_1));

            // ---------------- 订阅/发布 ----------------
            auto sensor_qos = rclcpp::SensorDataQoS();
            sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/image_raw", sensor_qos, std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));
            sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
                "/camera_info", 10, std::bind(&ArmorDetectorNode::camera_info_callback, this, std::placeholders::_1));

            publisher_armors_     = this->create_publisher<rm_interfaces::msg::ArmorsMsg>("/detector/armors_info", 10);
            publisher_result_img_ = this->create_publisher<sensor_msgs::msg::Image>("/detector/result", 10);
            publisher_crop_img_    = this->create_publisher<sensor_msgs::msg::Image>("/detector/crop_img", 10);
            publisher_bin_img_    = this->create_publisher<sensor_msgs::msg::Image>("/detector/bin_img", 10);
            publisher_armor_img_  = this->create_publisher<sensor_msgs::msg::Image>("/detector/img_armor", 10);
            publisher_armor_processed_img_  = this->create_publisher<sensor_msgs::msg::Image>("/detector/img_armor_processed", 10);

            // 工作线程detect_color_
            running_.store(true);
            worker_ = std::thread(&ArmorDetectorNode::processing_loop, this);

            RCLCPP_INFO(this->get_logger(), "Armor Detector Node has been started.");
        }

        ~ArmorDetectorNode() override {
            running_.store(false);
            if (worker_.joinable()) worker_.join();
        }

    private:
        // ---------------- 工具函数 ----------------
        template<typename T>
        T get_required_param(const std::string& name) {
            T value;
            if (!this->get_parameter(name, value)) {
                RCLCPP_ERROR(this->get_logger(), "Required parameter '%s' not found in YAML!", name.c_str());
                throw std::runtime_error("Missing required parameter: " + name);
            }
            return value;
        }

        // ---------------- 分类器加载 ----------------
        void reload_classifier_impl_(const std::string& onnx_path) {
            if (onnx_path.empty()) {
                classifier_.reset();
                if (detector_) detector_->set_classifier(nullptr);
                RCLCPP_WARN(this->get_logger(), "[Classifier] Disabled (cls_model_file is empty).");
                return;
            }
            try {
                classifier_ = std::make_shared<NumberClassifier>(onnx_path, cv::Size(20, 28));
                if (detector_) detector_->set_classifier(classifier_);
                RCLCPP_INFO(this->get_logger(), "[Classifier] Loaded ONNX: %s",
                            onnx_path.c_str());
            } catch (const std::exception& e) {
                classifier_.reset();
                if (detector_) detector_->set_classifier(nullptr);
                RCLCPP_ERROR(this->get_logger(), "[Classifier] Load failed: %s", e.what());
            }
        }

        // ---------------- 图像回调 (生产者) ----------------
        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
            cv_bridge::CvImageConstPtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            }
            catch (const std::exception& e) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                    "cv_bridge toCvShare failed: %s", e.what());
                return;
            }

            {
                // 使用花括号限制锁的生命周期，赋值完立即解锁
                std::lock_guard<std::mutex> lock(frame_mtx_);
                latest_frame_ = cv_ptr;
            }

            // [核心修改] 发出信号，唤醒正在 wait 的处理线程
            new_frame_cv_.notify_one();
        }
        // ---------------- 相机内参回调 ----------------
        void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(caminfo_mtx_);
            latest_caminfo_ = msg;
        }

        // ---------------- 主处理线程 (消费者) ----------------
        void processing_loop() {
            using clock = std::chrono::steady_clock;

            // 颜色定义保持不变
            const std::string GREEN = "\033[32m";
            const std::string CYAN  = "\033[96m";
            const std::string PINK  = "\033[38;5;218m";
            const std::string RESET = "\033[0m";

            auto last_print = clock::now();

            // 周期内状态
            bool had_detection_in_period = false;
            std::vector<rm_interfaces::msg::ArmorInfo> last_detected_armors;

            // FPS 统计
            int frame_count = 0;
            auto last_fps_time = clock::now();
            double current_fps = 0.0;

            while (rclcpp::ok() && running_.load()) {

                cv_bridge::CvImageConstPtr frame_ptr;

                // ==================== 核心修改区域 Start ====================
                {
                    // 必须使用 unique_lock 配合 condition_variable
                    std::unique_lock<std::mutex> lock(frame_mtx_);

                    // 线程在此处“死等”，直到被唤醒且满足条件 (有新帧 OR 程序退出)
                    new_frame_cv_.wait(lock, [this] {
                        return this->latest_frame_ != nullptr || !this->running_.load();
                    });

                    // 如果是因为程序退出被唤醒，则跳出循环
                    if (!running_.load()) break;

                    // 取出数据
                    frame_ptr = latest_frame_;

                    // [关键一步] 取走数据后，把指针置空！
                    // 这确保了下一轮循环如果相机还没来新图，wait 会继续卡住，
                    // 从而彻底根除了“重复识别同一帧”的问题。
                    latest_frame_ = nullptr;
                }
                // ==================== 核心修改区域 End ====================

                // 拿到数据后开始干活，这部分逻辑不用变
                frame_count++;

                // 获取最新的 CameraInfo (非阻塞，取当前最新的即可)
                sensor_msgs::msg::CameraInfo::SharedPtr caminfo_ptr;
                { std::lock_guard<std::mutex> lock(caminfo_mtx_); caminfo_ptr = latest_caminfo_; }

                if (frame_ptr->image.empty()) continue;
                cv::Mat frame = frame_ptr->image;

                // -------- 下面是原有的识别与发布逻辑 (保持不变) --------
                cv::Mat crop, bin, result, img_armor, img_armor_processed;
                std::vector<Armor> armors;
                bool detection_error = false;
                try {
                    armors = detector_->detect_armors(frame);
                    // 注意：display() 内部如果涉及耗时绘图，建议仅在调试时开启
                    std::tie(crop, bin, result, img_armor, img_armor_processed) = detector_->display();
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
                    detection_error = true;
                }

                rm_interfaces::msg::ArmorsMsg armors_msg;
                armors_msg.header.stamp = this->get_clock()->now();
                armors_msg.header.frame_id = "camera_frame";

                if (!detection_error && !armors.empty()) {
                    had_detection_in_period = true;
                    last_detected_armors.clear();

                    for (const auto& armor : armors) {
                        rm_interfaces::msg::ArmorInfo armor_info;
                        armor_info.armor_id = armor.armor_id;

                        // 这里的 processArmorCorners 可能会耗时，需注意优化
                        auto [dx, dy, dz, yaw] = pnp_->processArmorCorners(
                            caminfo_ptr, use_geometric_center_, frame, armor, armor.armor_id);

                        armor_info.dx = dx; armor_info.dy = dy; armor_info.dz = dz;
                        armor_info.yaw = yaw;
                        // armor_info.rz = rz;
                        last_detected_armors.push_back(armor_info);
                        armors_msg.armors.push_back(armor_info);
                    }
                }

                publisher_armors_->publish(armors_msg);

                // 图片发布建议加上 display_mode_ 判断，否则带宽压力巨大
                if (display_mode_) {
                    if (!crop.empty())
                        publisher_crop_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", crop).toImageMsg());
                    if (!bin.empty())
                        publisher_bin_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", bin).toImageMsg());
                    if (!result.empty())
                        publisher_result_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result).toImageMsg());
                    if (!img_armor.empty())
                        publisher_armor_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img_armor).toImageMsg());
                    if (!img_armor_processed.empty())
                        publisher_armor_processed_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", img_armor_processed).toImageMsg());
                }

                // -------- 打印节流逻辑 (保持不变) --------
                int pp_ms = print_period_ms_.load();
                auto now = clock::now();
                if (pp_ms <= 0 ||
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() >= pp_ms) {

                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
                    if (elapsed > 0) {
                        current_fps = frame_count * 1000.0 / elapsed;
                    }
                    frame_count = 0;
                    last_fps_time = now;

                    if (had_detection_in_period) {
                        for (const auto& armor_info : last_detected_armors) {
                            RCLCPP_INFO(this->get_logger(),
                                "发布 %sID:%s %s%d%s | %sdx:%.2f%s | %sdy:%.2f%s | %sdz:%.2f%s | %syaw:%.2f%s",
                                CYAN.c_str(), RESET.c_str(),
                                GREEN.c_str(), armor_info.armor_id, RESET.c_str(),
                                CYAN.c_str(), armor_info.dx, RESET.c_str(),
                                CYAN.c_str(), armor_info.dy, RESET.c_str(),
                                CYAN.c_str(), armor_info.dz, RESET.c_str(),
                                CYAN.c_str(), armor_info.yaw, RESET.c_str()
                            );
                        }
                    } else {
                        RCLCPP_INFO(this->get_logger(), "%sNo armors detected%s", PINK.c_str(), RESET.c_str());
                    }

                    RCLCPP_WARN(this->get_logger(), "[FPS_detector] %.1f", current_fps);

                    had_detection_in_period = false;
                    last_detected_armors.clear();
                    last_print = now;
                }
            }
        }

        // ---------------- 动态参数回调 ----------------
        rcl_interfaces::msg::SetParametersResult
        parameters_callback(const std::vector<rclcpp::Parameter>& parameters) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true; result.reason = "success";

            std::optional<std::string> new_model_file;

            for (const auto& param : parameters) {
                const auto& name = param.get_name();
                if (name == "cls_model_file") {
                    new_model_file = param.as_string();
                } else if (name == "light_area_min") { detector_->update_light_area_min(param.as_int());
                } else if (name == "light_h_w_ratio") { detector_->update_light_h_w_ratio(param.as_double());
                } else if (name == "light_angle_min") { detector_->update_light_angle_min(param.as_int());
                } else if (name == "light_angle_max") { detector_->update_light_angle_max(param.as_int());
                } else if (name == "light_red_ratio") { detector_->update_light_red_ratio(param.as_double());
                } else if (name == "light_blue_ratio") { detector_->update_light_blue_ratio(param.as_double());
                } else if (name == "height_rate_tol") { detector_->update_height_rate_tol(param.as_double());
                } else if (name == "height_multiplier_min") { detector_->update_height_multiplier_min(param.as_double());
                } else if (name == "height_multiplier_max") { detector_->update_height_multiplier_max(param.as_double());
                } else if (name == "binary_val") { detector_->update_binary_val(param.as_int());
                } else if (name == "detect_color") { detector_->update_detect_color(param.as_int());
                } else if (name == "display_mode") { detector_->update_display_mode(param.as_bool()); display_mode_ = param.as_bool();
                } else if (name == "print_period_ms") { print_period_ms_.store(param.as_int());
                } else if (name == "roi_crop") { detector_->update_roi_crop(param.as_bool());
                } else if (name == "roi_scale") { detector_->update_roi_scale(param.as_double());
                }
            }

            if (new_model_file) {
                std::string pkg_share = ament_index_cpp::get_package_share_directory("rm_detector");
                std::string new_path = cls_model_path_;
                if (new_model_file) {
                    new_path = (std::filesystem::path(pkg_share) / "model" / *new_model_file).string();
                }
                reload_classifier_impl_(new_path);
            }
            return result;
        }

    private:
        // ---- ROS 通道
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr        sub_image_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr   sub_camera_info_;
        rclcpp::Publisher<rm_interfaces::msg::ArmorsMsg>::SharedPtr  publisher_armors_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_result_img_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_armor_img_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_armor_processed_img_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_bin_img_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_crop_img_;
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_handle_;

        // ---- 模块实例
        std::shared_ptr<ArmorDetector> detector_;
        std::shared_ptr<PNP>           pnp_;
        std::shared_ptr<NumberClassifier> classifier_;

        // ---- 参数缓存
        std::string cls_model_path_;
        int detect_color_;
        bool display_mode_;
        int binary_val_;
        bool use_geometric_center_;
        std::atomic<int> print_period_ms_{1000};
        std::condition_variable new_frame_cv_;

        // ---- 缓存与线程
        std::mutex frame_mtx_;
        cv_bridge::CvImageConstPtr latest_frame_;
        std::mutex caminfo_mtx_;
        sensor_msgs::msg::CameraInfo::SharedPtr latest_caminfo_;
        std::thread worker_;
        std::atomic<bool> running_{false};
    };

}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DT46_VISION::ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
