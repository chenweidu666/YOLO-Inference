#ifndef OV_INFERENCE_H
#define OV_INFERENCE_H

#include <openvino/openvino.hpp>
#include <vector>
#include <string>

class OVInference {
private:
    ov::Core core;
    ov::CompiledModel compiled;
    ov::InferRequest infer_request;

public:
    OVInference(const std::string& model_path, const std::string& device = "CPU");

    std::vector<float> run_inference(const std::vector<float>& input_data,
                                     int batch_size, int channels, int height, int width);
};

#endif
