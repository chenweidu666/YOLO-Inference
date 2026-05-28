#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "preprocess.h"
#include "postprocess.h"
#include "onnx_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [model_path] [device]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "/home/test/Downloads/yolov5s.onnx";
    std::string device = (argc > 3) ? argv[3] : "cpu";

    std::cout << "========================================" << std::endl;
    std::cout << "YOLOv5s C++ ONNX Inference Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "[MAIN] Image loaded: " << image_path << std::endl;
    std::cout << "[MAIN] Image size: " << image.cols << "x" << image.rows << std::endl;

    std::cout << "[MAIN] Loading ONNX model: " << model_path << std::endl;
    ONNXInference infer(model_path, device);

    LetterBoxInfo info;
    std::cout << "[MAIN] Preprocessing (C++): letterbox + normalize + HWC2CHW..." << std::endl;
    std::vector<float> input_data = preprocess(image, info);

    std::cout << "[MAIN] LetterBox info: ratio=" << std::fixed << std::setprecision(3) << info.ratio
              << ", pad_w=" << info.pad_w << ", pad_h=" << info.pad_h << std::endl;
    std::cout << "[MAIN] Input tensor size: " << input_data.size() << " floats (3x640x640)" << std::endl;

    std::cout << "[MAIN] Running inference (C++ via ONNX)..." << std::endl;
    std::vector<float> output_data = infer.run_inference(input_data, 1, 3, 640, 640);

    std::cout << "[MAIN] Output tensor size: " << output_data.size() << " floats" << std::endl;

    std::cout << "[MAIN] Postprocessing (C++): decode + NMS + scale..." << std::endl;
    int num_classes = 80;
    int num_outputs_per_pred = 5 + num_classes;
    int num_predictions = output_data.size() / num_outputs_per_pred;
    
    std::cout << "[MAIN] Calculated: num_predictions=" << num_predictions 
              << ", num_classes=" << num_classes << std::endl;

    std::vector<Detection> detections = postprocess(output_data.data(), num_predictions, num_classes, 0.25f, 0.45f);

    std::cout << "[MAIN] Scaling detections back to original image coordinates..." << std::endl;
    auto scaled_dets = scale_detections(detections, info.ratio, info.pad_w, info.pad_h,
                                       info.orig_w, info.orig_h);

    std::cout << "[MAIN] Drawing detections on image..." << std::endl;
    cv::Mat result = image.clone();
    draw_detections(result, scaled_dets);

    std::string output_path = "output/result.jpg";
    cv::imwrite(output_path, result);
    std::cout << "========================================" << std::endl;
    std::cout << "Saved result to " << output_path << std::endl;
    std::cout << "Final detected objects: " << scaled_dets.size() << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
