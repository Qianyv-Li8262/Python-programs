
extern "C" __global__
void postprocess_kernel(
    const float* __restrict__ accum,
    unsigned char* __restrict__ pbo_out,
    int total_pixels, int frames
){
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= total_pixels) return;
    int r_idx = pid * 3;
    int g_idx = pid * 3 + 1;
    int b_idx = pid * 3 + 2;
    float inv_frames = 1.0f / (float)frames;
    float r = accum[r_idx] * inv_frames;
    float g = accum[g_idx] * inv_frames;
    float b = accum[b_idx] * inv_frames;


    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float contrast = 1.03f; // 增强 15% 的微对比度
    float factor = (contrast * (luma - 0.5f) + 0.5f) / (luma + 1e-5f);
    
    // 只有当亮度不是极高时才锐化，防止白色过曝区出现黑点   
    if(luma < 0.9f) {
        r *= factor; g *= factor; b *= factor;
    }
    float black_level = 0.03f; 
    r = fmaxf(0.0f, r - black_level);
    g = fmaxf(0.0f, g - black_level);
    b = fmaxf(0.0f, b - black_level);


    float exposure = 0.4f;
    r *= exposure; g *= exposure; b *= exposure;
    //float exposure = 1.2f;
    //r *= exposure; g *= exposure; b *= exposure;

    // 3. ACES Filmic Tone Mapping 
    float a = 2.51f, b_c = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    r = (r * (a * r + b_c)) / (r * (c * r + d) + e);
    g = (g * (a * g + b_c)) / (g * (c * g + d) + e);
    b = (b * (a * b + b_c)) / (b * (c * b + d) + e);
    
    r = __powf(r, 0.4545f) * 255.0f;
    g = __powf(g, 0.4545f) * 255.0f;
    b = __powf(b, 0.4545f) * 255.0f;
    r = fmaxf(0.0f, fminf(r, 255.0f));
    g = fmaxf(0.0f, fminf(g, 255.0f));
    b = fmaxf(0.0f, fminf(b, 255.0f));





    
    // Output RGBA for OpenGL (BGR->RGB swap: accum is RGB already)
    int out_idx = pid * 4;
    pbo_out[out_idx + 0] = (unsigned char)r;
    pbo_out[out_idx + 1] = (unsigned char)g;
    pbo_out[out_idx + 2] = (unsigned char)b;
    pbo_out[out_idx + 3] = 255;
}
