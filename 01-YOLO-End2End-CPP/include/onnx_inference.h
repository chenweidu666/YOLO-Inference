#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <memory>

class ONNXInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

public:
    ONNXInference(const std::string& model_path, const std::string& device = "cpu");
    
    std::vector<float> run_inference(const std::vector<float>& input_data, 
                                     int batch_size, int channels, int height, int width);
};

#endif
