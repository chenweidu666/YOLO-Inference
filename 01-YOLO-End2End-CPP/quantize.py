import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationDataReader
import cv2
import numpy as np
import os


class YOLOv5CalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder, input_size=640):
        self.image_folder = image_folder
        self.input_size = input_size
        self.image_paths = []
        for f in os.listdir(image_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(image_folder, f))
        self.image_paths = self.image_paths[:10]  # Use first 10 images for calibration
        self.current_index = 0
        self.input_name = None

    def get_next(self):
        if self.current_index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.current_index]
        self.current_index += 1

        # Preprocess image - letterbox + normalize + HWC2CHW
        img = cv2.imread(img_path)
        if img is None:
            return self.get_next()

        # Letterbox
        h, w = img.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        padded = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Normalize and HWC2CHW
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_trans = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(img_trans, axis=0).astype(np.float32)

        if self.input_name is None:
            model = onnx.load(model_fp32)
            self.input_name = model.graph.input[0].name

        return {self.input_name: input_tensor}


if __name__ == "__main__":
    model_fp32 = "data/yolov5su.onnx"
    model_int8 = "data/yolov5su_int8.onnx"

    print(f"Loading FP32 model: {model_fp32}")

    # Create calibration data reader
    dr = YOLOv5CalibrationDataReader("data/")

    # Quantize
    print("Running static quantization...")
    quantize_static(
        model_input=model_fp32,
        model_output=model_int8,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        optimize_model=True
    )

    print(f"INT8 model saved to: {model_int8}")

    # Verify model
    print("\nVerifying INT8 model...")
    model = onnx.load(model_int8)
    onnx.checker.check_model(model)
    print("Model verification passed!")

    # Print sizes
    size_fp32 = os.path.getsize(model_fp32)
    size_int8 = os.path.getsize(model_int8)
    print(f"\nFP32 model size: {size_fp32 / 1024 / 1024:.2f} MB")
    print(f"INT8 model size: {size_int8 / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {size_fp32 / size_int8:.2f}x")
