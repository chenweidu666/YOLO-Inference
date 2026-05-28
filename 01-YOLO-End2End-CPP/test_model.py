import cv2
import numpy as np
import onnxruntime as ort


def preprocess(img_path, target_size=640):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    padded = cv2.copyMakeBorder(resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_trans = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_trans, axis=0)


if __name__ == "__main__":
    session = ort.InferenceSession("data/yolov5su.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"Output: {output_name}, shape: {session.get_outputs()[0].shape}")

    input_data = preprocess("data/zidane.jpg")
    outputs = session.run([output_name], {input_name: input_data})
    output = outputs[0]

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output min: {output.min()}, max: {output.max()}, mean: {output.mean()}")

    # Print first few values
    print("\nFirst 20 values:")
    print(output.flatten()[:20])

    # Check the 5th channel (objectness)
    obj_conf = output[0, 4, :]
    print(f"\nObjectness min: {obj_conf.min()}, max: {obj_conf.max()}, mean: {obj_conf.mean()}")
    print(f"Objectness > 0.5: {(obj_conf > 0.5).sum()}")
    print(f"Objectness > 0.1: {(obj_conf > 0.1).sum()}")
