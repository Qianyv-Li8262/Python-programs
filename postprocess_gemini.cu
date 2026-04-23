
// 1. 提取高光 Kernel
extern "C" __global__
void extract_bright_kernel(
    const float* __restrict__ accum,
    float* __restrict__ bright_out,
    int w, int h, float frames, float threshold
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    int pid = y * w + x;
    int c_idx = pid * 3;
    
    float r = accum[c_idx] / frames;
    float g = accum[c_idx+1] / frames;
    float b = accum[c_idx+2] / frames;

    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    if (luma > threshold) {
        bright_out[c_idx]   = r;
        bright_out[c_idx+1] = g;
        bright_out[c_idx+2] = b;
    } else {
        bright_out[c_idx]   = 0.0f;
        bright_out[c_idx+1] = 0.0f;
        bright_out[c_idx+2] = 0.0f;
    }
}

// 2. 横向高斯模糊 Kernel
extern "C" __global__
void blur_x_kernel(
    const float* __restrict__ in_img,
    float* __restrict__ out_img,
    int w, int h, int radius, float sigma
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum_r = 0, sum_g = 0, sum_b = 0, sum_w = 0;
    float two_sigma_sq = 2.0f * sigma * sigma;

    for(int d = -radius; d <= radius; d++) {
        int nx = min(max(x + d, 0), w - 1); // 边界钳制
        int n_idx = (y * w + nx) * 3;
        
        float weight = __expf(-(float)(d * d) / two_sigma_sq);
        sum_r += in_img[n_idx] * weight;
        sum_g += in_img[n_idx+1] * weight;
        sum_b += in_img[n_idx+2] * weight;
        sum_w += weight;
    }

    int pid = (y * w + x) * 3;
    out_img[pid]   = sum_r / sum_w;
    out_img[pid+1] = sum_g / sum_w;
    out_img[pid+2] = sum_b / sum_w;
}

// 3. Fused Kernel: 纵向模糊 + 光晕合成 + 色调映射输出
extern "C" __global__
void blur_y_fuse_postprocess_kernel(
    const float* __restrict__ accum,
    const float* __restrict__ blur_x_in,
    unsigned char* __restrict__ pbo_out,
    int w, int h, int radius, float sigma, 
    float frames, float bloom_strength
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // --- A. 计算纵向模糊 (最终 Bloom 像素) ---
    float sum_r = 0, sum_g = 0, sum_b = 0, sum_w = 0;
    float two_sigma_sq = 2.0f * sigma * sigma;

    for(int d = -radius; d <= radius; d++) {
        int ny = min(max(y + d, 0), h - 1);
        int n_idx = (ny * w + x) * 3;
        
        float weight = __expf(-(float)(d * d) / two_sigma_sq);
        sum_r += blur_x_in[n_idx] * weight;
        sum_g += blur_x_in[n_idx+1] * weight;
        sum_b += blur_x_in[n_idx+2] * weight;
        sum_w += weight;
    }
    float bloom_r = sum_r / sum_w;
    float bloom_g = sum_g / sum_w;
    float bloom_b = sum_b / sum_w;

    // --- B. 混合原图 (HDR 空间) ---
    int pid = y * w + x;
    int c_idx = pid * 3;
    
    float r = accum[c_idx] / frames + bloom_r * bloom_strength;
    float g = accum[c_idx+1] / frames + bloom_g * bloom_strength;
    float b = accum[c_idx+2] / frames + bloom_b * bloom_strength;

    // --- C. 原汁原味的色调映射 (曝光 + ACES) ---
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float contrast = 1.03f;
    float factor = (contrast * (luma - 0.5f) + 0.5f) / (luma + 1e-5f);
    if(luma < 0.9f) {
        r *= factor; g *= factor; b *= factor;
    }
    float black_level = 0.03f; 
    r = fmaxf(0.0f, r - black_level);
    g = fmaxf(0.0f, g - black_level);
    b = fmaxf(0.0f, b - black_level);

    float exposure = 0.35f;
    r *= exposure; g *= exposure; b *= exposure;

    float a_c = 2.51f, b_c = 0.03f, c_c = 2.43f, d_c = 0.59f, e_c = 0.14f;
    r = (r * (a_c * r + b_c)) / (r * (c_c * r + d_c) + e_c);
    g = (g * (a_c * g + b_c)) / (g * (c_c * g + d_c) + e_c);
    b = (b * (a_c * b + b_c)) / (b * (c_c * b + d_c) + e_c);
    
    r = __powf(r, 0.4545f) * 255.0f;
    g = __powf(g, 0.4545f) * 255.0f;
    b = __powf(b, 0.4545f) * 255.0f;

    // --- D. 写入 PBO ---
    int out_idx = pid * 4;
    pbo_out[out_idx + 0] = (unsigned char)fmaxf(0.0f, fminf(r, 255.0f));
    pbo_out[out_idx + 1] = (unsigned char)fmaxf(0.0f, fminf(g, 255.0f));
    pbo_out[out_idx + 2] = (unsigned char)fmaxf(0.0f, fminf(b, 255.0f));
    pbo_out[out_idx + 3] = 255;
}