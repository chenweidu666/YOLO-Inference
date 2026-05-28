#include "postprocess.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static const char* COCO_NAMES[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

static float sigmoid_c(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float box_iou(float ax1, float ay1, float ax2, float ay2,
                     float bx1, float by1, float bx2, float by2) {
    float ix1 = ax1 > bx1 ? ax1 : bx1;
    float iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2;
    float iy2 = ay2 < by2 ? ay2 : by2;
    float iw = ix2 - ix1; if (iw < 0) iw = 0;
    float ih = iy2 - iy1; if (ih < 0) ih = 0;
    float inter = iw * ih;
    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);
    if (area_a + area_b - inter <= 0) return 0;
    return inter / (area_a + area_b - inter);
}

DetectionResult postprocess(const float* predictions, int num_anchors,
                            int num_classes, float conf_thresh, float iou_thresh) {
    DetectionResult result;
    result.count = 0;

    // predictions layout: [84, 8400] — 4 bbox channels + 80 cls channels
    // Each anchor i: cx=predictions[0*num_anchors+i], cy=predictions[1*num_anchors+i],
    //                w=predictions[2*num_anchors+i], h=predictions[3*num_anchors+i]
    //                cls_k=predictions[(4+k)*num_anchors+i]
    typedef struct { float x1, y1, x2, y2, score; int cls; } Cand;
    Cand* candidates = (Cand*)malloc(num_anchors * sizeof(Cand));
    int n_cand = 0;

    for (int i = 0; i < num_anchors; i++) {
        float cx = predictions[0 * num_anchors + i];
        float cy = predictions[1 * num_anchors + i];
        float w  = predictions[2 * num_anchors + i];
        float h  = predictions[3 * num_anchors + i];

        // Find max class score
        float max_score = -1.0f;
        int max_cls = 0;
        for (int k = 0; k < num_classes; k++) {
            float s = predictions[(4 + k) * num_anchors + i];
            if (s > max_score) { max_score = s; max_cls = k; }
        }

        if (max_score > conf_thresh) {
            candidates[n_cand].x1 = cx - w / 2;
            candidates[n_cand].y1 = cy - h / 2;
            candidates[n_cand].x2 = cx + w / 2;
            candidates[n_cand].y2 = cy + h / 2;
            candidates[n_cand].score = max_score;
            candidates[n_cand].cls = max_cls;
            n_cand++;
        }
    }

    // NMS
    int* order = (int*)malloc(n_cand * sizeof(int));
    for (int i = 0; i < n_cand; i++) order[i] = i;
    // Sort by score descending (simple bubble for small N)
    for (int i = 0; i < n_cand - 1; i++) {
        for (int j = i + 1; j < n_cand; j++) {
            if (candidates[order[j]].score > candidates[order[i]].score) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }
    char* suppressed = (char*)calloc(n_cand, 1);
    for (int i = 0; i < n_cand && result.count < MAX_DETECTIONS; i++) {
        if (suppressed[order[i]]) continue;
        for (int j = i + 1; j < n_cand; j++) {
            if (suppressed[order[j]]) continue;
            if (candidates[order[i]].cls != candidates[order[j]].cls) continue;
            float iou_val = box_iou(
                candidates[order[i]].x1, candidates[order[i]].y1,
                candidates[order[i]].x2, candidates[order[i]].y2,
                candidates[order[j]].x1, candidates[order[j]].y1,
                candidates[order[j]].x2, candidates[order[j]].y2);
            if (iou_val > iou_thresh) suppressed[order[j]] = 1;
        }
        if (!suppressed[order[i]]) {
            Detection* d = &result.items[result.count++];
            d->x1 = candidates[order[i]].x1;
            d->y1 = candidates[order[i]].y1;
            d->x2 = candidates[order[i]].x2;
            d->y2 = candidates[order[i]].y2;
            d->score = candidates[order[i]].score;
            d->class_id = candidates[order[i]].cls;
        }
    }
    free(candidates); free(order); free(suppressed);
    return result;
}

void scale_detections(DetectionResult* result, float ratio, int pad_x, int pad_y,
                      int orig_w, int orig_h) {
    for (int i = 0; i < result->count; i++) {
        Detection* d = &result->items[i];
        d->x1 = (d->x1 - pad_x) / ratio;
        d->y1 = (d->y1 - pad_y) / ratio;
        d->x2 = (d->x2 - pad_x) / ratio;
        d->y2 = (d->y2 - pad_y) / ratio;
        // Clamp
        if (d->x1 < 0) d->x1 = 0;
        if (d->y1 < 0) d->y1 = 0;
        if (d->x2 > orig_w) d->x2 = orig_w;
        if (d->y2 > orig_h) d->y2 = orig_h;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

const char* coco_name(int id) {
    if (id >= 0 && id < 80) return COCO_NAMES[id];
    return "unknown";
}

#ifdef __cplusplus
}
#endif
