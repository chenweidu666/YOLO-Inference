#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#define MAX_DETECTIONS 64

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int class_id;
} Detection;

typedef struct {
    Detection items[MAX_DETECTIONS];
    int count;
} DetectionResult;

DetectionResult postprocess(const float* predictions, int num_anchors,
                            int num_classes, float conf_thresh, float iou_thresh);

void scale_detections(DetectionResult* result, float ratio, int pad_x, int pad_y,
                      int orig_w, int orig_h);

#ifdef __cplusplus
extern "C" {
#endif

const char* coco_name(int id);

#ifdef __cplusplus
}
#endif

#endif
