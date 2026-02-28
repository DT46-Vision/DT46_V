#include "rm_detector/detector.hpp"

namespace DT46_VISION{
    // 计算两个点之间的距离
    double calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
        return cv::norm(p1 - p2);
    }

    // 调整宽高和角度的函数
    std::pair<cv::Size2f, double> adjust(const cv::Size2f& w_h, double angle) {
        float w = w_h.width;
        float h = w_h.height;

        if (w > h) {
            std::swap(w, h);
            if (angle >= 0) {
                angle -= 90;
            } else {
                angle += 90;
            }
        }

        return std::make_pair(cv::Size2f(w, h), angle);
    }

    // 将角度转换为斜率的函数
    double angle_to_slope(double angle_degrees) {
        double angle_radians = angle_degrees * M_PI / 180.0;
        return std::tan(angle_radians);
    }

    // Light 类的构造函数实现
    Light::Light(const cv::Point2f& up, const cv::Point2f& down, int color)
        : up(up), down(down), color(color) {
        cx = static_cast<int>(std::abs(up.x - down.x) / 2 + std::min(up.x, down.x));
        cy = static_cast<int>(std::abs(up.y - down.y) / 2 + std::min(up.y, down.y));
        height = calculate_distance(up, down);
    }

    Armor::Armor(float height_multiplier, const Light& light1, const Light& light2, NumberClassifier::Result res)
        : height_multiplier(height_multiplier), light1_up(light1.up), light1_down(light1.down),
        light2_up(light2.up), light2_down(light2.down),
        res(res) {

        color = light1.color;
        armor_id = get_id();
    }

    int Armor::get_id() const {
        if(color == 0){
            if(res.class_id == 0){
                return 6;
            }
            if(res.class_id == 1){
                return 8;
            }
            if(res.class_id == 2){
                return 11;
            }
        }
        else if(color == 1){
            if(res.class_id == 0){
                return 0;
            }
            if(res.class_id == 1){
                return 2;
            }
            if(res.class_id == 2){
                return 5;
            }
        }
        return -1;
    }

    // ArmorDetector 类的实现
    ArmorDetector::ArmorDetector(int detect_color, bool display_mode, int binary_val, const Params& params)
        : binary_val(binary_val), color(detect_color), display_mode(display_mode), params(params) {}

    void ArmorDetector::update_roi_crop(bool new_roi_crop_val) {
        params.roi_crop = new_roi_crop_val;
    }

    void ArmorDetector::update_roi_scale(double new_roi_scale_val) {
        params.roi_scale = std::clamp(new_roi_scale_val, 0.1, 1.0);
    }
    void ArmorDetector::update_light_area_min(int new_light_area_min) {
        params.light_area_min = new_light_area_min;
    }

    void ArmorDetector::update_light_h_w_ratio(double new_light_h_w_ratio) {
        params.light_h_w_ratio = new_light_h_w_ratio;
    }

    void ArmorDetector::update_light_angle_min(int new_light_angle_min) {
        params.light_angle_min = new_light_angle_min;
    }

    void ArmorDetector::update_light_angle_max(int new_light_angle_max) {
        params.light_angle_max = new_light_angle_max;
    }

    void ArmorDetector::update_light_red_ratio(float new_light_red_ratio) {
        params.light_red_ratio = new_light_red_ratio;
    }

    void ArmorDetector::update_light_blue_ratio(float new_light_blue_ratio) {
        params.light_blue_ratio = new_light_blue_ratio;
    }

    void ArmorDetector::update_height_rate_tol(float new_height_rate_tol) {
        params.height_rate_tol = new_height_rate_tol;
    }

    void ArmorDetector::update_height_multiplier_min(float new_height_multiplier_min) {
        params.height_multiplier_min = new_height_multiplier_min;
    }

    void ArmorDetector::update_height_multiplier_max(float new_height_multiplier_max) {
        params.height_multiplier_max = new_height_multiplier_max;
    }

    void ArmorDetector::update_binary_val(int new_binary_val) {
        binary_val = new_binary_val;
    }

    void ArmorDetector::update_detect_color(int new_color) {
        color = new_color;
    }

    void ArmorDetector::update_display_mode(bool new_display_mode) {
        display_mode = new_display_mode;
    }

    cv::Mat ArmorDetector::process(const cv::Mat& img_input) {
        if (img_input.empty()) {
            std::cerr << "Input image is empty!" << std::endl;
            return cv::Mat();
        }
        img = img_input.clone();
        
        cv::Mat gray_img;
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, img_binary, binary_val, 255, cv::THRESH_BINARY);
        return img_binary;
    }

    std::vector<Light> ArmorDetector::find_lights(const cv::Mat& img_binary_input) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::RotatedRect> is_lights;
        std::vector<Light> lights_found;

        cv::findContours(img_binary_input, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) >= params.light_area_min) {
                cv::RotatedRect min_rect = cv::minAreaRect(contour);
                cv::Size2f w_h = min_rect.size;
                double angle = min_rect.angle;
                std::tie(w_h, angle) = adjust(w_h, angle);
                if (w_h.height / w_h.width < params.light_h_w_ratio) {
                    continue;
                }
                if (angle >= params.light_angle_min && angle <= params.light_angle_max) {
                    cv::RotatedRect rect(min_rect.center, w_h, static_cast<float>(angle));
                    is_lights.push_back(rect);
                }
            }
        }

        for (const auto& rect : is_lights) { // 这里的定义上下是遵守opencv规定的坐标系来描述的，y轴是竖直向下的，x轴是水平向右的
            cv::Point2f box[4];
            rect.points(box);

            int up_x = static_cast<int>(std::abs(box[0].x - box[3].x) / 2 + std::min(box[0].x, box[3].x));
            int up_y = static_cast<int>(std::abs(box[0].y - box[3].y) / 2 + std::min(box[0].y, box[3].y));
            cv::Point2f up(up_x, up_y);

            int down_x = static_cast<int>(std::abs(box[1].x - box[2].x) / 2 + std::min(box[1].x, box[2].x));
            int down_y = static_cast<int>(std::abs(box[1].y - box[2].y) / 2 + std::min(box[1].y, box[2].y));
            cv::Point2f down(down_x, down_y);

            int length = static_cast<int>(calculate_distance(up, down));

            if (length <= 0) continue;  // 避免空 ROI

            cv::Mat roi(1, length, CV_8UC3, cv::Scalar(0, 0, 0));

            for (int i = 0; i < length; ++i) {
                float t = static_cast<float>(i) / length;
                int current_x = static_cast<int>(up_x + (down_x - up_x) * t);
                int current_y = static_cast<int>(up_y + (down_y - up_y) * t);

                if (current_x >= 0 && current_x < img.cols && current_y >= 0 && current_y < img.rows) {
                    roi.at<cv::Vec3b>(0, i) = img.at<cv::Vec3b>(current_y, current_x);
                }
            }

            int sum_b = cv::sum(roi)[0];
            int sum_r = cv::sum(roi)[2];

            if ((color == 1 || color == 2) && sum_b > sum_r * params.light_blue_ratio) {
                lights_found.push_back(Light(up, down, 1));
            } else if ((color == 0 || color == 2) && sum_r > sum_b * params.light_red_ratio) {
                lights_found.push_back(Light(up, down, 0));
            }
        }

        lights = lights_found;
        return lights;
    }

    std::pair<cv::Point2f, cv::Point2f> ArmorDetector::stretch_point(const cv::Point2f& up, const cv::Point2f& down, float ratio) const {
        cv::Point2f center = (up + down) * 0.5f;
        cv::Point2f v = down - up;
        float h = cv::norm(v);
        if (h < 1e-3) {
            return {up, down};  // 避免除零
        }

        float scale = (1.0f * ratio) * 0.5f;  // 上下各拉一半
        cv::Point2f v_scaled = v * scale;

        return {center - v_scaled, center + v_scaled};
    }

    NumberClassifier::Result ArmorDetector::get_armor_result(const Light& light1, const Light& light2) {
        NumberClassifier::Result res;

        auto l1 = stretch_point(light1.up, light1.down, 2.22f);
        auto l2 = stretch_point(light2.up, light2.down, 2.22f);

        auto armor_down = stretch_point(l1.second, l2.second, 0.66f);
        auto armor_up = stretch_point(l1.first, l2.first, 0.66f);

        cv::Point2f src_armor_pts[4] = {
            armor_up.first,     // left-up
            armor_up.second,    // right-up
            armor_down.first,   // left-down
            armor_down.second,  // right-down
        };

        cv::Mat H = cv::getPerspectiveTransform(src_armor_pts, dst_armor_pts);
        cv::Mat roi;
        cv::warpPerspective(img, roi, H, cv::Size(89, 125));
        img_armor = roi;

        if (!roi.empty() && classifier_) {
            cv::Mat gray, binary;

            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

            // --- Otsu 阈值二值化 ---
            cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

            img_armor_processed = binary;
            res = classifier_->classify(binary);
        }
        return res;
    }


    std::pair<NumberClassifier::Result, float> ArmorDetector::is_close(const Light& light1, const Light& light2) {
        NumberClassifier::Result res;
        // 计算公共变量
        float height_max, height_min;
        if (std::max(light1.height, light2.height) == light1.height){
            height_max = light1.height;
            height_min = light2.height;
        }
        else{
            height_max = light2.height;
            height_min = light1.height;
        }

        float height_rate = height_max / height_min;
        // 如果高度比例不符合，直接返回
        if (height_rate >= params.height_rate_tol) {
            return {res, 0.f};
        }

        float height = (height_max + height_min) / 2;

        double distance = calculate_distance({static_cast<float>(light1.cx), static_cast<float>(light1.cy)}, {static_cast<float>(light2.cx), static_cast<float>(light2.cy)});

        if ((distance > height * params.height_multiplier_min && distance < height * params.height_multiplier_max)
            || (distance > height * 1.56f * params.height_multiplier_min && distance < height * 1.76f * params.height_multiplier_max)
            ) {
                res = get_armor_result(light1, light2);
        }
        return {res, distance / height};
    }

    std::vector<Armor> ArmorDetector::is_armor(const std::vector<Light>& lights) {
        std::vector<Armor> armors_found;

        if (lights.size() < 2) {
            return armors_found;
        }

        // 按中心 x 排序（左 → 右）
        std::vector<Light> sorted_lights = lights;
        std::sort(sorted_lights.begin(), sorted_lights.end(),
                [](const Light& a, const Light& b) {
                    return a.cx < b.cx;
                });

        // 标记哪些灯条已经被使用过
        std::vector<bool> used(sorted_lights.size(), false);

        // 全对遍历
        for (size_t i = 0; i < sorted_lights.size(); i++) {
            if (used[i]) continue; // 已被匹配就跳过

            for (size_t j = i + 1; j < sorted_lights.size(); j++) {
                if (used[j]) continue;
                if (sorted_lights[i].color != sorted_lights[j].color) continue;

                NumberClassifier::Result res;
                float height_multiplier;
                std::tie(res, height_multiplier) = is_close(sorted_lights[i], sorted_lights[j]);

                // 只要通过几何验证就是一个装甲板
                if (res.class_id >= 0 && res.class_id < 3) {
                    armors_found.emplace_back(height_multiplier, sorted_lights[i], sorted_lights[j], res);
                    used[i] = used[j] = true; // 标记两个灯条已用
                    break; // i 已经匹配成功，不再尝试别的 j
                }
            }
        }

        armors = armors_found;
        return armors;
    }

    cv::Mat ArmorDetector::draw_rect(cv::Mat img_draw) {
        // 获取图像尺寸
        int img_height = img_draw.rows;
        int img_width = img_draw.cols;

        // 计算矩形高度（图像高度的10%）
        int rect_height = img_height * 0.1;

        // 根据高宽比计算矩形宽度
        float h_w_ratio = params.light_h_w_ratio;
        int rect_width = static_cast<int>(rect_height / h_w_ratio);

        // 确保矩形宽度不超过图像宽度，留10像素边距
        rect_width = std::min(rect_width, img_width - 10);

        // 设置矩形顶部距离图像顶部10像素的偏移量
        int top_offset = 10;

        // 定义矩形位置（居中）
        cv::Point top_left((img_width - rect_width) / 2, top_offset);
        cv::Point bottom_right((img_width + rect_width) / 2, top_offset + rect_height);

        // 绘制绿色矩形（BGR格式，绿色(0, 255, 0)，线宽2）
        cv::rectangle(img_draw, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);

        float height_rate_tol = params.height_rate_tol;

        // 竖线高度
        int line_height = static_cast<int>(rect_height / height_rate_tol);

        // 矩形底部 y 坐标
        int rect_bottom_y = bottom_right.y;

        // 左边竖线
        int left_line_x = top_left.x - rect_width;
        cv::Point left_bottom(left_line_x, rect_bottom_y);
        cv::Point left_top(left_line_x, rect_bottom_y - line_height);
        cv::line(img_draw, left_bottom, left_top, cv::Scalar(0, 255, 0), 2);

        // 右边竖线
        int right_line_x = bottom_right.x + rect_width;
        cv::Point right_bottom(right_line_x, rect_bottom_y);
        cv::Point right_top(right_line_x, rect_bottom_y - line_height);
        cv::line(img_draw, right_bottom, right_top, cv::Scalar(0, 255, 0), 2);

        return img_draw;
    }


    cv::Mat ArmorDetector::draw_lights(cv::Mat img_draw) {
        for (const auto& light : lights) {
            if (light.color == 0) {
                cv::line(img_draw, light.up, light.down, cv::Scalar(0, 100, 255), 1);
                cv::circle(img_draw, cv::Point(light.cx, light.cy), 1, cv::Scalar(255, 0, 0), -1);
            } else if (light.color == 1) {
                cv::line(img_draw, light.up, light.down, cv::Scalar(200, 71, 90), 1);
                cv::circle(img_draw, cv::Point(light.cx, light.cy), 1, cv::Scalar(0, 0, 255), -1);
            }
        }
        return img_draw;
    }

    cv::Mat ArmorDetector::draw_armors(cv::Mat img_draw) {
        for (const auto& armor : armors) {
            cv::Scalar bgColor;
            // 绘制交叉线
            if (armor.color == 0) {
                cv::line(img_draw, armor.light1_up, armor.light2_down, cv::Scalar(128, 0, 128), 1);
                cv::line(img_draw, armor.light2_up, armor.light1_down, cv::Scalar(128, 0, 128), 1);
                bgColor = cv::Scalar(255, 255, 0); // 背景
            } else if (armor.color == 1) {
                cv::line(img_draw, armor.light1_up, armor.light2_down, cv::Scalar(255, 255, 0), 1);
                cv::line(img_draw, armor.light2_up, armor.light1_down, cv::Scalar(255, 255, 0), 1);
                bgColor = cv::Scalar(128, 0, 128); // 背景
            }

            // 设置文本参数
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 0.5;
            int thickness = 2;
            cv::Scalar textColor(193, 182, 255); // 粉色 BGR，接近 \033[38;5;218m
            int lineHeight = 15;                 // 每行间距
            int padding = 2;                     // 背景框内边距

            // 文本内容（带格式化）
            std::vector<std::string> texts = {
                std::to_string(armor.armor_id),
                cv::format("%.2f", armor.height_multiplier),   // ⭐ 新增 height_multiplier
                std::to_string(armor.res.class_id),
                cv::format("%.2f", armor.res.confidence)       // ⭐ 格式化 confidence
            };

            // 计算文本区域（整体高度和最大宽度）
            int maxWidth = 0;
            int totalHeight = lineHeight * texts.size();
            for (const auto& text : texts) {
                cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, nullptr);
                maxWidth = std::max(maxWidth, textSize.width);
            }

            // 文本起始位置（light1_down 上方）
            cv::Point textOrg = armor.light1_down;
            textOrg.y -= totalHeight + padding; // 上移到 light1_down 上方

            // 绘制背景框
            cv::Point bgTopLeft(textOrg.x - padding, textOrg.y - padding);
            cv::Point bgBottomRight(textOrg.x + maxWidth + padding, textOrg.y + totalHeight + padding);
            cv::rectangle(img_draw, bgTopLeft, bgBottomRight, bgColor, cv::FILLED);

            // 绘制多行文本
            for (size_t i = 0; i < texts.size(); ++i) {
                cv::Point linePos(textOrg.x, textOrg.y + lineHeight * (i + 1));
                cv::putText(img_draw, texts[i], linePos, fontFace, fontScale, textColor, thickness);
            }
        }
        return img_draw;
    }


    cv::Mat ArmorDetector::draw_img() {
    cv::Mat img_draw = original_img.clone();  // 用原图作为底图
    if (img_draw.empty()) {
        img_draw = img.clone();  // 回退
    }

    // 绘制裁剪区域蓝色边缘
    if (params.roi_crop && !last_roi.empty()) {
        cv::rectangle(img_draw, last_roi, cv::Scalar(255, 0, 0), 2);  // 蓝线，粗细 2
    }

    // 原有绘制
    
    img_draw = draw_rect(img_draw);
    img_draw = draw_lights(img_draw);
    img_drawn = draw_armors(img_draw);

    return img_drawn;
}

    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> ArmorDetector::display() {
        if (display_mode == true) {
            img_drawn = draw_img();
            return std::make_tuple(img_binary, img_drawn, img_armor, img_armor_processed);
        } else if (display_mode == false) {
            return std::make_tuple(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)));
        } else {
            std::cerr << "Invalid display mode" << std::endl;
            return std::make_tuple(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)), cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)));
        }
    }

    std::vector<Armor> ArmorDetector::detect_armors(const cv::Mat& img_input) {
        if (img_input.empty()){
            return {};
        }

        original_img = img_input.clone();
    
        //判断 roi 局部模式

        cv::Mat working_img = img_input;     // img_input ---> working_img
        cv::Rect roi_rect(0, 0, img_input.cols, img_input.rows);
        bool using_crop = false;

        if (params.roi_crop && params.roi_scale > 0.0 && params.roi_scale <1.0){
            int crop_w = static_cast<int>(img_input.cols * params.roi_scale);
            int crop_h = static_cast<int>(img_input.rows * params.roi_scale);
            int center_x = img_input.cols / 2;
            int center_y = img_input.rows / 2;

            int x = std::max(0,center_x - crop_w / 2);
            int y = std::max(0,center_y - crop_h / 2);
            crop_w = std::min(img_input.cols - x , crop_w);
            crop_h = std::min(img_input.rows - y, crop_h);
            
            roi_rect = cv::Rect(x, y, crop_w, crop_h);

            if (roi_rect.area() >= 100 * 100){  //最小面积阈值
                working_img = img_input(roi_rect).clone();
                using_crop = true;
            }
        }

        img = working_img;
        img_binary = process(working_img);
        lights = find_lights(img_binary);
        armors = is_armor(lights);

        if (using_crop) {
        int offset_x = roi_rect.x;
        int offset_y = roi_rect.y;

        for (auto& light : lights) {
            light.up.x    += offset_x;
            light.up.y    += offset_y;
            light.down.x  += offset_x;
            light.down.y  += offset_y;
            light.cx      += offset_x;
            light.cy      += offset_y;
        }
    

        for (auto& armor : armors) {
            armor.light1_up.x   += offset_x;   armor.light1_up.y   += offset_y;
            armor.light1_down.x += offset_x;   armor.light1_down.y += offset_y;
            armor.light2_up.x   += offset_x;   armor.light2_up.y   += offset_y;
            armor.light2_down.x += offset_x;   armor.light2_down.y += offset_y;
        }
    }

        last_roi = roi_rect;  // 记录给 draw_img 使用

        return armors;
    }
} // namespace DT46_VISION