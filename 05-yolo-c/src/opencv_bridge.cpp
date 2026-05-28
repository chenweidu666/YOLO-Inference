/* OpenCV C++ → C bridge for image I/O and drawing */
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "postprocess.h"
#include <stdlib.h>

extern "C" {

unsigned char* cv_imread_bgr(const char* path, int* w, int* h) {
    cv::Mat img = cv::imread(path);
    if (img.empty()) return NULL;
    *w = img.cols;
    *h = img.rows;
    unsigned char* out = (unsigned char*)malloc(img.rows * img.cols * 3);
    memcpy(out, img.data, img.rows * img.cols * 3);
    return out;
}

void cv_imwrite_jpg(const char* path, const unsigned char* data, int w, int h) {
    cv::Mat img(h, w, CV_8UC3, (void*)data);
    cv::imwrite(path, img);
}

void cv_draw_detections(unsigned char* img_data, int img_w, int img_h,
                        const DetectionResult* det) {
    cv::Mat img(img_h, img_w, CV_8UC3, img_data);
    for (int i = 0; i < det->count; i++) {
        const Detection* d = &det->items[i];
        cv::Scalar color(0, 255, 0);
        cv::rectangle(img, cv::Point((int)d->x1, (int)d->y1),
                           cv::Point((int)d->x2, (int)d->y2), color, 2);
        char label[128];
        snprintf(label, sizeof(label), "%s %.2f", coco_name(d->class_id), d->score);
        cv::putText(img, label, cv::Point((int)d->x1, (int)d->y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

} /* extern "C" */
