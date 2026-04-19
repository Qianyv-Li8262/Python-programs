import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
import glfw
from zero_copy_window import ZeroCopyWindow
def create_texture_object(img_cp):
    h, w, c = img_cp.shape
    bytes_per_pixel = 16 
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :3] = img_cp
    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D, 
        arr=rgba,                  
        chDesc=ch_fmt,  
        width=w,
        height=h,
        pitchInBytes=pitch_bytes
    )
    tex_ptr = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeWrap, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj, rgba
# 1. 使用 OpenCV 读取图片
img_bgr = cv2.imread('eso0932a.jpg')
# img_bgr = cv2.imread('test_img2.bmp')

# 2. OpenCV 默认是 BGR 通道顺序，我们需要转成 RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. 归一化到 [0.0, 1.0] 并转为 float32
# img_float = img_rgb.astype(np.float32) 
img_float = img_rgb.astype(np.float32) / 255.0

# 4. 将数据传送到 GPU (转为 CuPy 数组)
img_cp = cp.array(img_float)
tex_handle, _internal_storage = create_texture_object(img_cp) # img要在显卡里面

with open("blackhole_kernel.cu", "r", encoding="utf-8") as f:
    cuda_source = f.read()


module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))


trace_rays_kernel = module.get_function("blackholekernel")

postprocess_source=r'''
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
'''

postprocess_kernel = cp.RawKernel(postprocess_source, 'postprocess_kernel', options=('-use_fast_math',))

print('kernel complied')
w,h=1024,1024
window=ZeroCopyWindow(w,h,'try')
# current_frame_float=window.map_pbo()
frame_intermediate_result=cp.empty((h * w * 3), dtype=cp.float32)
accum=cp.empty((h * w * 3), dtype=cp.float32)
block_x,block_y=32,32
grid_x=w//block_x+1 if w%block_x!=0 else w//block_x
grid_y=h//block_y+1 if h%block_y!=0 else h//block_y
print(grid_x)
tot_pixels=w*h
frames=1
# t0=time.time()
# trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
# (frame_intermediate_result, cp.uint64(tex_handle.ptr),cp.float32(10),cp.float32(0),cp.float32(0)   ,cp.float32(-0.91651),cp.float32(0.4),cp.float32(0)
#     ,cp.float32(0.4),cp.float32(0.91651),cp.float32(0)   ,cp.float32(0),cp.float32(0),cp.float32(1)   ,cp.int32(1024),cp.int32(1024),
#         cp.float32(2),cp.float32(2),cp.float32(0.5)  ,cp.float32(0.1),cp.int32(5000)))

# postprocess_kernel((cp.int32(1024),),(cp.int32(1024),),(frame_intermediate_result,current_frame_float,tot_pixels,frames))

# print('start.')
# window.unmap_and_draw()
# print('ended.')
# t1=time.time()
# print(f'{1/(t1-t0)}FPS')
# flagg=True
while not window.should_close():
    current_frame_float=window.map_pbo()
    # glfw.wait_events()  # 使用 wait_events 而不是 poll_events，这样画面静止时不占用 CPU
    trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
    (frame_intermediate_result, cp.uint64(tex_handle.ptr),cp.float32(10),cp.float32(0),cp.float32(0)   ,cp.float32(-0.91651),cp.float32(0.4),cp.float32(0)
    ,cp.float32(0.4),cp.float32(0.91651),cp.float32(0)   ,cp.float32(0),cp.float32(0),cp.float32(1)   ,cp.int32(1024),cp.int32(1024),
        cp.float32(2),cp.float32(2),cp.float32(0.5)  ,cp.float32(0.1),cp.int32(5000)))
    accum = accum +frame_intermediate_result
    frames+=1
    postprocess_kernel((cp.int32(1024),),(cp.int32(1024),),(accum,current_frame_float,tot_pixels,frames))
    window.unmap_and_draw()

window.destroy()
print('Done.')