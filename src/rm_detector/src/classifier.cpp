#include "rm_detector/classifier.hpp"

namespace DT46_VISION {

    NumberClassifier::NumberClassifier(const std::string& onnx_path,
                                       cv::Size input_size)
        : input_size_(input_size)
    {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        if (net_.empty()) {
            throw std::runtime_error("NumberClassifier: failed to load ONNX model: " + onnx_path);
        }

        // ---- DNN 加速配置 ----
        try {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (...) {
            try {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
            } catch (...) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }
        bin_f_.create(input_size_, CV_8UC1);
    }

    NumberClassifier::Result NumberClassifier::classify(const cv::Mat& armor_img)
    {
        Result out;
        if (armor_img.empty()) return out;

        cv::resize(armor_img, bin_f_, input_size_, 0, 0, cv::INTER_AREA);

        // 转 blob 并归一化（单通道也可以直接 blobFromImage）
        cv::Mat blob = cv::dnn::blobFromImage(bin_f_, 1.0 / 255.0, input_size_, cv::Scalar(), false, false, CV_32F);

        // 前向推理
        net_.setInput(blob);
        cv::Mat logits = net_.forward(); // shape: 1xC

        // 取最大值/类别
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(logits, nullptr, &confidence, nullptr, &classIdPoint);

        out.class_id = classIdPoint.x;
        out.confidence = static_cast<float>(confidence);  // 仍然是 logit / 未 softmax 的值

        return out;
    }

} // namespace DT46_VISION
