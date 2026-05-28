#ifndef PREPROCESS_H
#define PREPROCESS_H

typedef struct {
    float ratio;
    int pad_x;
    int pad_y;
    int orig_w;
    int orig_h;
} LetterBoxInfo;

/**
 * LetterBox resize + normalize + HWC→CHW
 * src: BGR uint8 image
 * out: float array of size 3*640*640 (CHW, normalized to [0,1])
 */
void preprocess(const unsigned char* src_data, int src_w, int src_h,
                float* out, LetterBoxInfo* info, int target_size);

#endif
