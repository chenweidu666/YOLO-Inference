#include "postprocess.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>

static float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static const char* coco_names[] = {
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

struct BoxWithClass {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

static float iou(const BoxWithClass& a, const BoxWithClass& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) return 0.0f;
    return inter_area / union_area;
}

std::vector<Detection> postprocess(
    const float* predictions, int num_predictions, int num_classes,
    float conf_thresh, float iou_thresh, float min_size) {

    std::cout << "[POSTPROCESS] Input: num_predictions=" << num_predictions
              << ", num_classes=" << num_classes
              << ", conf_thresh=" << conf_thresh
              << ", iou_thresh=" << iou_thresh
              << ", min_size=" << min_size << std::endl;

    // Output shape is [1, 84, 8400], 84 = 4(bbox) + 80(classes), no obj_conf
    const int num_outputs = 4 + num_classes;

    // Print first few raw predictions to understand data layout
    std::cout << "[POSTPROCESS] Raw prediction samples (first 5 anchors):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        // predictions[col][row] format - col is 0-83, row is 0-8399
        float cx = predictions[i + 0 * num_predictions];
        float cy = predictions[i + 1 * num_predictions];
        float w = predictions[i + 2 * num_predictions];
        float h = predictions[i + 3 * num_predictions];
        std::cout << "  Anchor " << i << ": cx=" << std::fixed << std::setprecision(3) << cx
                  << ", cy=" << cy << ", w=" << w << ", h=" << h;
        std::cout << ", cls_0=" << predictions[i + 4 * num_predictions] << std::endl;
    }

    std::vector<BoxWithClass> candidates;

    for (int i = 0; i < num_predictions; ++i) {
        float cx = predictions[i + 0 * num_predictions];
        float cy = predictions[i + 1 * num_predictions];
        float w = predictions[i + 2 * num_predictions];
        float h = predictions[i + 3 * num_predictions];

        // YOLOv5su output: [cx, cy, w, h, cls_0..cls_79] (84 values), no obj_conf
        // Class scores are already sigmoided (range 0-1)
        int best_class = -1;
        float best_score = 0.0f;

        for (int c = 0; c < num_classes; ++c) {
            float cls_score = predictions[i + (4 + c) * num_predictions];
            if (cls_score > best_score) {
                best_score = cls_score;
                best_class = c;
            }
        }

        if (best_score >= conf_thresh) {
            BoxWithClass box;
            box.x1 = cx - w / 2.0f;
            box.y1 = cy - h / 2.0f;
            box.x2 = cx + w / 2.0f;
            box.y2 = cy + h / 2.0f;
            box.score = best_score;
            box.class_id = best_class;
            candidates.push_back(box);
        }
    }

    std::cout << "[POSTPROCESS] Total predictions scanned: " << num_predictions << std::endl;
    std::cout << "[POSTPROCESS] Candidates passing confidence threshold: " << candidates.size() << std::endl;

    if (candidates.empty()) {
        std::cout << "[POSTPROCESS] No candidates found, returning empty detections" << std::endl;
        return {};
    }

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const BoxWithClass& a, const BoxWithClass& b) {
                  return a.score > b.score;
              });

    std::cout << "[POSTPROCESS] Top 5 scores:" << std::endl;
    for (size_t i = 0; i < std::min(candidates.size(), (size_t)5); ++i) {
        std::cout << "  #" << i << ": class=" << candidates[i].class_id
                  << " (" << (candidates[i].class_id < 80 ? coco_names[candidates[i].class_id] : "unknown") << ")"
                  << ", score=" << std::fixed << std::setprecision(3) << candidates[i].score
                  << ", box=[" << candidates[i].x1 << "," << candidates[i].y1 
                  << "," << candidates[i].x2 << "," << candidates[i].y2 << "]"
                  << " (w=" << (candidates[i].x2-candidates[i].x1) << ", h=" << (candidates[i].y2-candidates[i].y1) << ")"
                  << std::endl;
    }

    // Standard NMS
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> result;
    int suppressed_count = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;

        float box_w = candidates[i].x2 - candidates[i].x1;
        float box_h = candidates[i].y2 - candidates[i].y1;
        if (box_w < min_size || box_h < min_size) {
            suppressed[i] = true;
            suppressed_count++;
            continue;
        }

        Detection det;
        det.x1 = candidates[i].x1;
        det.y1 = candidates[i].y1;
        det.x2 = candidates[i].x2;
        det.y2 = candidates[i].y2;
        det.confidence = candidates[i].score;
        det.class_id = candidates[i].class_id;
        result.push_back(det);

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;
            float w_j = candidates[j].x2 - candidates[j].x1;
            float h_j = candidates[j].y2 - candidates[j].y1;
            if (w_j < min_size || h_j < min_size) {
                suppressed[j] = true;
                suppressed_count++;
                continue;
            }
            if (candidates[j].class_id != candidates[i].class_id) continue;
            float current_iou = iou(candidates[i], candidates[j]);
            if (current_iou > iou_thresh) {
                suppressed[j] = true;
                suppressed_count++;
            }
        }
    }

    std::cout << "[POSTPROCESS] NMS complete: suppressed " << suppressed_count 
              << " boxes, remaining " << result.size() << " detections" << std::endl;

    return result;
}

std::vector<Detection> scale_detections(const std::vector<Detection>& dets,
                                         float ratio, int pad_w, int pad_h,
                                         int orig_w, int orig_h) {
    std::cout << "[SCALE] Scaling " << dets.size() << " detections: ratio=" 
              << std::fixed << std::setprecision(3) << ratio
              << ", pad_w=" << pad_w << ", pad_h=" << pad_h
              << ", orig_w=" << orig_w << ", orig_h=" << orig_h << std::endl;

    std::vector<Detection> scaled;
    for (const auto& det : dets) {
        Detection s;
        s.class_id = det.class_id;
        s.confidence = det.confidence;

        s.x1 = (det.x1 - pad_w) / ratio;
        s.y1 = (det.y1 - pad_h) / ratio;
        s.x2 = (det.x2 - pad_w) / ratio;
        s.y2 = (det.y2 - pad_h) / ratio;

        s.x1 = std::max(0.0f, std::min(s.x1, (float)orig_w));
        s.y1 = std::max(0.0f, std::min(s.y1, (float)orig_h));
        s.x2 = std::max(0.0f, std::min(s.x2, (float)orig_w));
        s.y2 = std::max(0.0f, std::min(s.y2, (float)orig_h));

        scaled.push_back(s);
    }
    std::cout << "[SCALE] Done. Scaled detections: " << scaled.size() << std::endl;
    return scaled;
}

std::vector<Detection> filter_small_boxes(const std::vector<Detection>& dets, float min_size, int img_size) {
    std::vector<Detection> filtered;
    for (const auto& d : dets) {
        float w = d.x2 - d.x1;
        float h = d.y2 - d.y1;
        if (w >= min_size && h >= min_size) {
            filtered.push_back(d);
        }
    }
    std::cout << "[FILTER] filter_small_boxes: min_size=" << min_size 
              << ", before=" << dets.size() << ", after=" << filtered.size() << std::endl;
    return filtered;
}

static cv::Scalar get_color(int class_id) {
    static cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255),
        cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128), cv::Scalar(128, 128, 0),
        cv::Scalar(255, 128, 0), cv::Scalar(128, 0, 255), cv::Scalar(0, 255, 128),
    };
    return colors[class_id % 12];
}

void draw_detections(cv::Mat& image, const std::vector<Detection>& dets) {
    std::cout << "[DRAW] Drawing " << dets.size() << " detections on image" << std::endl;
    for (const auto& det : dets) {
        cv::Scalar color = get_color(det.class_id);

        cv::rectangle(image,
                      cv::Point((int)det.x1, (int)det.y1),
                      cv::Point((int)det.x2, (int)det.y2),
                      color, 2);

        const char* label = (det.class_id < 80) ? coco_names[det.class_id] : "unknown";
        std::string text = std::string(label) + " " + std::to_string((int)(det.confidence * 100)) + "%";

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int top = std::max((int)det.y1, textSize.height + 4);

        cv::rectangle(image,
                      cv::Point((int)det.x1, top - textSize.height - 4),
                      cv::Point((int)det.x1 + textSize.width, top),
                      color, -1);

        cv::putText(image, text,
                    cv::Point((int)det.x1, top - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}