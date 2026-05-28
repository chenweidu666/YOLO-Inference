#include "onnx_inference.h"
#include <stdexcept>
#include <iostream>

ONNXInference::ONNXInference(const std::string& model_path, const std::string& device) 
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5"), session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {
    // Constructor implementation
}

std::vector<float> ONNXInference::run_inference(const std::vector<float>& input_data, 
                                                int batch_size, int channels, int height, int width) {
    // Create input tensor
    std::vector<int64_t> input_shape{batch_size, channels, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_data.data()), 
        input_data.size(), 
        input_shape.data(), 
        input_shape.size()
    );

    // Get input and output names
    auto input_names_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_names_ptr.get();

    auto output_names_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_names_ptr.get();

    // Prepare for inference
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                      input_names, 
                                      &input_tensor, 
                                      1, 
                                      output_names, 
                                      1);

    // Get output data
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    
    // Get output shape to determine size
    std::vector<int64_t> output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }

    // Print output shape for debugging
    std::cout << "Output tensor shape: ";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << "x";
    }
    std::cout << std::endl;

    std::vector<float> result(floatarr, floatarr + output_size);
    return result;
}
