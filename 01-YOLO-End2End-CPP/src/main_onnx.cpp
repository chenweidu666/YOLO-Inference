#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

#include "preprocess.h"
#include "postprocess.h"
#include "onnx_inference.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [model_path] [device]" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string model_path = (argc > 2) ? argv[2] : "../00-Models/yolov5su_fp32.onnx";
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
    // Output: [1, 84, 8400] where 84 = 4(bbox) + 80(classes), no separate obj_conf
    int num_outputs_per_pred = 4 + num_classes;
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

    // Generate output path following same structure as Python project
    std::string model_filename = fs::path(model_path).filename().string();
    std::string model_arch = "yolov5su";
    std::string precision = "fp32";

    // Parse model filename: yolov5su_fp32.onnx -> arch=yolov5su, precision=fp32
    size_t last_underscore = model_filename.rfind('_');
    if (last_underscore != std::string::npos) {
        size_t dot_pos = model_filename.rfind('.');
        model_arch = model_filename.substr(0, last_underscore);
        precision = model_filename.substr(last_underscore + 1, dot_pos - last_underscore - 1);
    }

    std::string image_filename = fs::path(image_path).stem().string();
    std::string output_dir = "outputs/" + model_arch + "/" + precision + "/";
    fs::create_directories(output_dir);
    std::string output_path = output_dir + image_filename + "_result.jpg";

    cv::imwrite(output_path, result);
    std::cout << "Saved image to " << output_path << std::endl;

    // Save JSON results
    std::string json_path = output_dir + image_filename + "_result.json";
    std::ofstream json_file(json_path);
    json_file << "{\n  \"count\": " << scaled_dets.size() << ",\n  \"detections\": [\n";
    for (size_t i = 0; i < scaled_dets.size(); ++i) {
        const auto& d = scaled_dets[i];
        json_file << "    {\n";
        json_file << "      \"bbox\": [" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << "],\n";
        json_file << "      \"confidence\": " << d.confidence << ",\n";
        json_file << "      \"class_id\": " << d.class_id << "\n";
        json_file << "    }";
        if (i < scaled_dets.size() - 1) json_file << ",";
        json_file << "\n";
    }
    json_file << "  ]\n}\n";
    json_file.close();
    std::cout << "Saved JSON to " << json_path << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "Final detected objects: " << scaled_dets.size() << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
