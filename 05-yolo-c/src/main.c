#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "openvino/c/openvino.h"
#include "preprocess.h"
#include "postprocess.h"

extern unsigned char* cv_imread_bgr(const char* path, int* w, int* h);
extern void cv_imwrite_jpg(const char* path, const unsigned char* data, int w, int h);
extern void cv_draw_detections(unsigned char* img, int img_w, int img_h,
                               const DetectionResult* det);

#define OV_CHECK(stmt) do { \
    ov_status_e _st = (stmt); \
    if (_st != OK) { \
        fprintf(stderr, "OV ERROR: %s (%s:%d)\n", ov_get_error_info(_st), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.jpg> [model.onnx] [device]\n", argv[0]);
        return 1;
    }

    const char* image_path = argv[1];
    const char* model_path = (argc > 2) ? argv[2] : "../01-models/yolov5su_fp32.onnx";
    const char* device = (argc > 3) ? argv[3] : "CPU";

    ov_version_t version = {0};
    OV_CHECK(ov_get_openvino_version(&version));

    printf("========================================\n");
    printf("YOLOv5su C OpenVINO Inference\n");
    printf("OpenVINO: %s\n", version.buildNumber ? version.buildNumber : "?");
    printf("========================================\n");

    int src_w, src_h;
    unsigned char* src_img = cv_imread_bgr(image_path, &src_w, &src_h);
    if (!src_img) {
        fprintf(stderr, "Failed to load: %s\n", image_path);
        return 1;
    }
    printf("[1/5] Image: %dx%d\n", src_w, src_h);

    const int TARGET = 640;
    LetterBoxInfo info;
    float* input_tensor = (float*)malloc(3 * TARGET * TARGET * sizeof(float));
    preprocess(src_img, src_w, src_h, input_tensor, &info, TARGET);
    printf("[2/5] Preprocess: ratio=%.3f pad=(%d,%d)\n", info.ratio, info.pad_x, info.pad_y);

    double t0 = now_ms();
    ov_core_t* core = NULL;
    ov_compiled_model_t* compiled = NULL;
    ov_infer_request_t* infer_req = NULL;
    ov_tensor_t* input_ov = NULL;

    OV_CHECK(ov_core_create(&core));
    OV_CHECK(ov_core_compile_model_from_file(core, model_path, device, 0, &compiled));
    printf("[3/5] Model loaded: %.1fms (%s, %s)\n", now_ms() - t0, model_path, device);

    OV_CHECK(ov_compiled_model_create_infer_request(compiled, &infer_req));

    int64_t dims[] = {1, 3, TARGET, TARGET};
    ov_shape_t shape = {0};
    OV_CHECK(ov_shape_create(4, dims, &shape));
    OV_CHECK(ov_tensor_create_from_host_ptr(F32, shape, input_tensor, &input_ov));
    OV_CHECK(ov_infer_request_set_input_tensor(infer_req, input_ov));

    double t_infer_start = now_ms();
    OV_CHECK(ov_infer_request_infer(infer_req));
    printf("[4/5] Inference: %.1fms\n", now_ms() - t_infer_start);

    ov_tensor_t* output_ov = NULL;
    OV_CHECK(ov_infer_request_get_output_tensor(infer_req, &output_ov));

    void* output_data = NULL;
    OV_CHECK(ov_tensor_data(output_ov, &output_data));

    DetectionResult det = postprocess((const float*)output_data, 8400, 80, 0.25f, 0.45f);
    scale_detections(&det, info.ratio, info.pad_x, info.pad_y, info.orig_w, info.orig_h);
    printf("[5/5] Detections: %d\n", det.count);
    for (int i = 0; i < det.count; i++) {
        Detection* d = &det.items[i];
        printf("  [%d] %s %.3f (%.0f,%.0f,%.0f,%.0f)\n",
               i, coco_name(d->class_id), d->score,
               d->x1, d->y1, d->x2, d->y2);
    }

    cv_draw_detections(src_img, src_w, src_h, &det);
    const char* out_path = "output.jpg";
    cv_imwrite_jpg(out_path, src_img, src_w, src_h);
    printf("Saved: %s\n", out_path);

    ov_shape_free(&shape);
    ov_tensor_free(input_ov);
    ov_infer_request_free(infer_req);
    ov_compiled_model_free(compiled);
    ov_core_free(core);
    ov_version_free(&version);
    free(input_tensor);
    free(src_img);

    printf("Total: %.1fms\n", now_ms() - t0);
    return 0;
}
