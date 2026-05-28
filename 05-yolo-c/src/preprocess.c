#include "preprocess.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static unsigned char* create_letterbox(const unsigned char* src, int src_w, int src_h,
                                        int target, int* out_w, int* out_h,
                                        LetterBoxInfo* info) {
    float ratio = (float)target / src_w;
    if ((float)target / src_h < ratio)
        ratio = (float)target / src_h;

    int new_w = (int)roundf(src_w * ratio);
    int new_h = (int)roundf(src_h * ratio);
    int pad_x = (target - new_w) / 2;
    int pad_y = (target - new_h) / 2;

    info->ratio = ratio;
    info->pad_x = pad_x;
    info->pad_y = pad_y;
    info->orig_w = src_w;
    info->orig_h = src_h;

    // Allocate target canvas filled with 114
    unsigned char* canvas = (unsigned char*)malloc(target * target * 3);
    memset(canvas, 114, target * target * 3);

    // Bilinear resize inline + paste into canvas
    for (int y = 0; y < new_h; y++) {
        float src_y = y / ratio;
        int sy0 = (int)floorf(src_y);
        int sy1 = sy0 < src_h - 1 ? sy0 + 1 : sy0;
        float wy1 = src_y - sy0;

        for (int x = 0; x < new_w; x++) {
            float src_x = x / ratio;
            int sx0 = (int)floorf(src_x);
            int sx1 = sx0 < src_w - 1 ? sx0 + 1 : sx0;
            float wx1 = src_x - sx0;

            for (int c = 0; c < 3; c++) {
                float v00 = src[(sy0 * src_w + sx0) * 3 + c];
                float v10 = src[(sy1 * src_w + sx0) * 3 + c];
                float v01 = src[(sy0 * src_w + sx1) * 3 + c];
                float v11 = src[(sy1 * src_w + sx1) * 3 + c];
                float v = v00 * (1 - wx1) * (1 - wy1) + v01 * wx1 * (1 - wy1)
                        + v10 * (1 - wx1) * wy1 + v11 * wx1 * wy1;
                int dst_idx = ((y + pad_y) * target + (x + pad_x)) * 3 + c;
                canvas[dst_idx] = (unsigned char)(v + 0.5f);
            }
        }
    }
    *out_w = target;
    *out_h = target;
    return canvas;
}

void preprocess(const unsigned char* src_data, int src_w, int src_h,
                float* out, LetterBoxInfo* info, int target_size) {
    int letter_w, letter_h;
    unsigned char* letterboxed = create_letterbox(src_data, src_w, src_h,
                                                   target_size, &letter_w, &letter_h, info);

    // BGR→RGB + normalize + HWC→CHW
    int total = target_size * target_size;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < total; i++) {
            // letterboxed is BGR, channel order: B=0, G=1, R=2
            // We want RGB output: R→ch0, G→ch1, B→ch2
            int src_c = 2 - c;  // BGR→RGB swap
            out[c * total + i] = letterboxed[i * 3 + src_c] / 255.0f;
        }
    }
    free(letterboxed);
}
