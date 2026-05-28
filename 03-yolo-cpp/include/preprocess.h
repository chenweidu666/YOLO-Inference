#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <vector>

struct LetterBoxInfo {
    float ratio;
    int pad_w;
    int pad_h;
    int orig_w;
    int orig_h;
};

LetterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_size = 640);

std::vector<float> preprocess(const cv::Mat& image, LetterBoxInfo& info, int target_size = 640);

#endif
