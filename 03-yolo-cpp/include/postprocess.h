#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

std::vector<Detection> postprocess(
    const float* predictions, int num_predictions, int num_classes,
    float conf_thresh = 0.25f, float iou_thresh = 0.45f, float min_size = 3.0f);

std::vector<Detection> scale_detections(const std::vector<Detection>& dets,
                                         float ratio, int pad_w, int pad_h,
                                         int orig_w, int orig_h);

std::vector<Detection> filter_small_boxes(const std::vector<Detection>& dets, float min_size, int img_size = 640);

void draw_detections(cv::Mat& image, const std::vector<Detection>& dets);

#endif
