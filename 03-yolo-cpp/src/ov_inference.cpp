#include "ov_inference.h"
#include <stdexcept>
#include <iostream>

OVInference::OVInference(const std::string& model_path, const std::string& device)
    : core()
{
    std::cout << "[OVInference] Loading model: " << model_path << std::endl;
    auto model = core.read_model(model_path);
    std::cout << "[OVInference] Compiling for device: " << device << std::endl;
    compiled = core.compile_model(model, device);
    infer_request = compiled.create_infer_request();
    std::cout << "[OVInference] Model loaded and compiled successfully!" << std::endl;
}

std::vector<float> OVInference::run_inference(const std::vector<float>& input_data,
                                                int batch_size, int channels, int height, int width)
{
    ov::Shape input_shape{(size_t)batch_size, (size_t)channels, (size_t)height, (size_t)width};
    auto input_tensor = ov::Tensor(ov::element::f32, input_shape, const_cast<float*>(input_data.data()));
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output_tensor = infer_request.get_output_tensor();
    float* output_data = output_tensor.data<float>();
    size_t output_size = output_tensor.get_size();
    std::vector<float> result(output_data, output_data + output_size);
    return result;
}
