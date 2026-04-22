import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
import glfw
from cupyx.scipy.ndimage import gaussian_filter
import os
from zero_copy_window import ZeroCopyWindow
def create_texture_object(img_cp,num_of_channels):
    h, w, c = img_cp.shape
    bytes_per_pixel = 16 
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :num_of_channels] = img_cp
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
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType,
        normalizedCoords=1
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj, rgba




base_path = os.path.dirname(os.path.abspath(__file__))
img_file_path = os.path.join(base_path, 'eso0932a.tif')#改图片
img_bgr = cv2.imread(img_file_path)




if img_bgr is None:
    print(f"错误：无法在路径 {img_file_path} 找到背景图片！")
    print("请检查图片文件名是否正确，或者图片是否在文件夹中。")
    exit() 

img_bgr = cv2.imread(img_file_path)


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0
img_cp = cp.array(img_float)
# img_cp = gaussian_filter(img_cp, sigma=0.8, axes=(0, 1)) 
tex_handle, _internal_storage = create_texture_object(img_cp,3)

physlut_file_path = os.path.join(base_path, 'disk_lut.npy')#改图片
lut_phys= cp.load(physlut_file_path).astype(cp.float32)

tex_handle_lut,____=create_texture_object(lut_phys,4)

colorlut_file_path = os.path.join(base_path, 'color_lut.npy')
lut_color= cp.load(colorlut_file_path).astype(cp.float32)

tex_handle_color,____=create_texture_object(lut_color,3)

kernel_path = os.path.join(base_path, "blackholekernel3.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
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


    float exposure = 1.3f;
    r *= exposure; g *= exposure; b *= exposure;
    //float exposure = 1.2f;
    //r *= exposure; g *= exposure; b *= exposure;

    // 3. ACES Filmic Tone Mapping (保留亮度时的色彩饱和度)
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
'''

postprocess_kernel = cp.RawKernel(postprocess_source, 'postprocess_kernel', options=('-use_fast_math',))

print('kernel complied')

def apply_bloom(image_gpu, threshold=1.0, blur_radius=15, bloom_strength=0.8):
    """
    对 GPU 图像应用 Bloom 效果
    
    Args:
        image_gpu: CuPy 数组，形状 (H, W, 4)，RGBA 格式，值域 [0, 255]
        threshold: 亮度阈值（0-255），超过此值的像素参与 Bloom
        blur_radius: 高斯模糊半径（像素）
        bloom_strength: Bloom 强度（0-1）
    """
    # 1. 转换为 float 并归一化到 [0, 1]
    img_float = image_gpu.astype(cp.float32) / 255.0
    
    # 2. 提取亮部（luminance > threshold）
    luminance = 0.2126 * img_float[:, :, 0] + 0.7152 * img_float[:, :, 1] + 0.0722 * img_float[:, :, 2]
    bright_mask = (luminance > threshold / 255.0).astype(cp.float32)
    
    # 3. 提取亮部颜色
    bright_colors = cp.zeros_like(img_float[:, :, :3])
    for i in range(3):
        bright_colors[:, :, i] = img_float[:, :, i] * bright_mask
    
    # 4. 对亮部进行高斯模糊
    blurred = cp.zeros_like(bright_colors)
    for i in range(3):
        blurred[:, :, i] = gaussian_filter(bright_colors[:, :, i], sigma=blur_radius)
    
    # 5. 叠加 Bloom 到原图
    result = img_float[:, :, :3] + blurred * bloom_strength
    result = cp.clip(result, 0.0, 1.0)
    
    # 6. 转回 uint8
    image_gpu[:, :, :3] = (result * 255.0).astype(cp.uint8)
    
    return image_gpu

# 超参数！


w,h=2048,2048
r=20.0
th=0
cam_pos = np.array([r*np.cos(th),r*np.sin(th), 0.0], dtype=np.float32)
cam_yaw = np.pi
cam_pitch = 0.0
cam_roll = 0.0 

move_speed = 0.025
turn_speed = 0.01
focus_speed=0.01
jitnum=1

focal_length=1.0




window=ZeroCopyWindow(w,h,'try')
frame_intermediate_result=cp.empty((h * w * 3), dtype=cp.float32)
accum=cp.zeros((h * w * 3), dtype=cp.float32)
block_x,block_y=32,32
grid_x=w//block_x+1 if w%block_x!=0 else w//block_x
grid_y=h//block_y+1 if h%block_y!=0 else h//block_y
print(grid_x)
tot_pixels=w*h
frames=1
world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
fwd_z = np.sin(cam_pitch)
fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
fwd /= np.linalg.norm(fwd)
right0 = np.cross(fwd, world_up)
right_norm = np.linalg.norm(right0)
if right_norm > 1e-6:
    right0 /= right_norm
else:
    right0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
up0 = np.cross(right0, fwd)
up0 /= np.linalg.norm(up0)
right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)


while not window.should_close():
    current_frame_float = window.map_pbo()
    # th+=0.001
    # cam_pos = np.array([r*np.cos(th),r*np.sin(th), 0.0], dtype=np.float32)
    camera_moved = False
    
    if glfw.KEY_W in window.key_pressed:
        cam_pos += fwd * move_speed
        camera_moved = True
    if glfw.KEY_S in window.key_pressed:
        cam_pos -= fwd * move_speed
        camera_moved = True
    if glfw.KEY_D in window.key_pressed:
        cam_pos -= right * move_speed
        camera_moved = True
    if glfw.KEY_A in window.key_pressed:
        cam_pos += right * move_speed
        camera_moved = True
    if glfw.KEY_UP in window.key_pressed:
        cam_pos += up * move_speed 
        camera_moved = True
    if glfw.KEY_DOWN in window.key_pressed:
        cam_pos -= up * move_speed
        camera_moved = True
    if glfw.KEY_E in window.key_pressed:
        cam_yaw += turn_speed
        camera_moved = True
    if glfw.KEY_Q in window.key_pressed:
        cam_yaw -= turn_speed
        camera_moved = True
    if glfw.KEY_R in window.key_pressed: 
        cam_pitch += turn_speed
        camera_moved = True
    if glfw.KEY_F in window.key_pressed:  
        cam_pitch -= turn_speed
        camera_moved = True
    if glfw.KEY_Z in window.key_pressed:  
        cam_roll -= turn_speed
        camera_moved = True
    if glfw.KEY_C in window.key_pressed: 
        cam_roll += turn_speed
        camera_moved = True
    if glfw.KEY_G in window.key_pressed: 
        focal_length -= focus_speed
        camera_moved = True
    if glfw.KEY_T in window.key_pressed: 
        focal_length += focus_speed
        camera_moved = True
    if camera_moved:
        accum.fill(0)
        frames = 1
        cam_pitch = np.clip(cam_pitch, -np.pi/2 + 0.001, np.pi/2 - 0.001)
        fwd_x = np.cos(cam_yaw) * np.cos(cam_pitch)
        fwd_y = np.sin(cam_yaw) * np.cos(cam_pitch)
        fwd_z = np.sin(cam_pitch)
        fwd = np.array([fwd_x, fwd_y, fwd_z], dtype=np.float32)
        fwd /= np.linalg.norm(fwd)
        right0 = np.cross(fwd, world_up)
        right_norm = np.linalg.norm(right0)
        if right_norm > 1e-6:
            right0 /= right_norm
        else:
            right0 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up0 = np.cross(right0, fwd)
        up0 /= np.linalg.norm(up0)
        right = right0 * np.cos(cam_roll) + up0 * np.sin(cam_roll)
        up = up0 * np.cos(cam_roll) - right0 * np.sin(cam_roll)

    trace_rays_kernel((grid_x, grid_y,), (block_x, block_y,), 
        (frame_intermediate_result, cp.uint64(tex_handle.ptr),cp.uint64(tex_handle_lut.ptr),cp.uint64(tex_handle_color.ptr),
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
         cp.int32(w), cp.int32(h),
         cp.float32(2), cp.float32(2), cp.float32(focal_length), cp.float32(0.1), cp.int32(5000), cp.int32(jitnum),cp.int32(frames)))
    
    accum = accum + frame_intermediate_result
    postprocess_kernel((cp.int32(tot_pixels//1024+1 if tot_pixels%1024!=0 else tot_pixels//1024),),(cp.int32(1024),),(accum, current_frame_float, tot_pixels, frames))
    frames += 1
#     current_frame_uint8 = current_frame_float.view(cp.uint8).reshape((h, w, 4))
#     current_frame_float = apply_bloom(
#     current_frame_uint8, 
#     threshold=200,        # 亮度阈值（0-255）
#     blur_radius=20,       # 模糊半径（像素）
#     bloom_strength=0.6    # Bloom 强度
# )
    window.unmap_and_draw()


window.destroy()
print('Done.')