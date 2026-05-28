#include "preprocess.h"

LetterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_size) {
    LetterBoxInfo info;
    info.orig_w = src.cols;
    info.orig_h = src.rows;

    float ratio = std::min((float)target_size / src.rows, (float)target_size / src.cols);
    info.ratio = ratio;

    int new_w = (int)round(src.cols * ratio);
    int new_h = (int)round(src.rows * ratio);

    int pad_w = target_size - new_w;
    int pad_h = target_size - new_h;

    info.pad_w = pad_w / 2;
    info.pad_h = pad_h / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    dst = cv::Mat(target_size, target_size, resized.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(dst(cv::Rect(pad_w / 2, pad_h / 2, new_w, new_h)));

    return info;
}

std::vector<float> preprocess(const cv::Mat& image, LetterBoxInfo& info, int target_size) {
    cv::Mat letterboxed;
    info = letterbox(image, letterboxed, target_size);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32FC3, 1.0/255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> output;
    output.reserve(target_size * target_size * 3);

    for (int c = 0; c < 3; ++c) {
        const float* ptr = channels[c].ptr<float>();
        for (int i = 0; i < target_size * target_size; ++i) {
            output.push_back(ptr[i]);
        }
    }

    return output;
}
