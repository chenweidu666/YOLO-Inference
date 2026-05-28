"""
Dynamic INT8 quantization on opset13 model for CPU compatibility
Only quantizes weights, preserves activation values.
"""
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    # Use opset 13 model
    input_path = "01-models/yolov5su_fp32_opset13.onnx"
    output_path = "01-models/yolov5su_int8_cpu.onnx"
    
    if not os.path.exists(input_path):
        print(f"Creating opset13 model first...")
        m = onnx.load("01-models/yolov5su_fp32.onnx")
        m.opset_import[0].version = 13
        onnx.save(m, input_path)
    
    print(f"Dynamic INT8 quantization (opsed 13)...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.2f} MB")
    
    # Verify
    import onnxruntime as ort
    try:
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        print(f"  ✅ CPU EP loads successfully!")
    except Exception as e:
        print(f"  ❌ {e}")
        return
    
    # Quick accuracy test
    import cv2
    import numpy as np
    image = cv2.imread("00-data/zidane.jpg")
    h, w = image.shape[:2]
    ratio = min(640/w, 640/h)
    new_w, new_h = int(round(w*ratio)), int(round(h*ratio))
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.full((640,640,3), 114, dtype=np.uint8)
    sx, sy = (640-new_w)//2, (640-new_h)//2
    canvas[sy:sy+new_h, sx:sx+new_w] = resized
    rgb = canvas[:,:,::-1].transpose(2,0,1).astype(np.float32)/255.0
    tensor = np.expand_dims(rgb, axis=0)
    
    outputs = session.run(None, {'images': tensor})
    pred = np.transpose(outputs[0][0])
    scores = pred[:, 4:]
    max_scores = np.sort(np.max(scores, axis=1))[-5:]
    print(f"  Top-5 max class scores: {max_scores}")


if __name__ == "__main__":
    main()
