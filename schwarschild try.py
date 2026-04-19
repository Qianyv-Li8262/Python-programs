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

# img_bgr = cv2.imread('eso0932a.jpg')
img_bgr = cv2.imread('eso0932a.tif')
# img_bgr = cv2.imread('test_img2.bmp')


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_float = img_rgb.astype(np.float32) / 255.0
img_cp = cp.array(img_float)
tex_handle, _internal_storage = create_texture_object(img_cp)

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
w,h=2048,2048
window=ZeroCopyWindow(w,h,'try')
# current_frame_float=window.map_pbo()
frame_intermediate_result=cp.empty((h * w * 3), dtype=cp.float32)
accum=cp.zeros((h * w * 3), dtype=cp.float32)
block_x,block_y=32,32
grid_x=w//block_x+1 if w%block_x!=0 else w//block_x
grid_y=h//block_y+1 if h%block_y!=0 else h//block_y
print(grid_x)
tot_pixels=w*h
frames=1


cam_pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)
cam_yaw = np.pi
cam_pitch = 0.0
cam_roll = 0.0 

move_speed = 0.025
turn_speed = 0.01


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
        (frame_intermediate_result, cp.uint64(tex_handle.ptr),
         cp.float32(cam_pos[0]), cp.float32(cam_pos[1]), cp.float32(cam_pos[2]),
         cp.float32(fwd[0]), cp.float32(fwd[1]), cp.float32(fwd[2]),
         cp.float32(right[0]), cp.float32(right[1]), cp.float32(right[2]),
         cp.float32(up[0]), cp.float32(up[1]), cp.float32(up[2]),
         cp.int32(w), cp.int32(h),
         cp.float32(2), cp.float32(2), cp.float32(0.5), cp.float32(0.1), cp.int32(5000), cp.int32(5)))
    
    accum = accum + frame_intermediate_result
    postprocess_kernel((cp.int32(4096),),(cp.int32(1024),),(accum, current_frame_float, tot_pixels, frames))
    frames += 1
    window.unmap_and_draw()

window.destroy()
print('Done.')