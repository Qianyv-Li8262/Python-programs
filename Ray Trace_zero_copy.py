import numpy as np
import cupy as cp
import cv2
import time
import ctypes
import glfw
from OpenGL.GL import *
from cupy.cuda import texture
from cupy.cuda import runtime

print('import succeed.')

# ============ 找到 cudart DLL ============
import os, glob
cupy_cuda_path = os.path.join(os.path.dirname(cp.__file__), 'cuda')
dll_candidates = glob.glob(os.path.join(cupy_cuda_path, 'bin', 'cudart64_*.dll'))
if not dll_candidates:
    dll_candidates = glob.glob(os.path.join(cupy_cuda_path, '..', '**', 'cudart64_*.dll'), recursive=True)
if not dll_candidates:
    # fallback: search in PATH
    import shutil
    for p in os.environ.get('PATH','').split(';'):
        dll_candidates += glob.glob(os.path.join(p, 'cudart64_*.dll'))
if dll_candidates:
    cudart = ctypes.cdll.LoadLibrary(dll_candidates[0])
    print(f'Loaded cudart: {dll_candidates[0]}')
else:
    # last resort: try cupy's internal handle
    cudart = ctypes.CDLL('cudart64_12.dll')
    print('Loaded cudart64_12.dll from system PATH')

# ============ 光学参数 ============
nr=1.5168; ng=1.52; nb=1.5168

class sphericalLens:
    def __init__(self, r1, r2, d, rm):
        self.r1=float(r1); self.r2=float(r2); self.d=float(d); self.rm=float(rm)
        self.signr1=float(np.sign(r1)); self.signr2=float(np.sign(r2))
        self.zc1=float(-d/2+np.sign(r1)*np.sqrt(r1**2-rm**2))
        self.zc2=float(d/2-np.sign(r2)*np.sqrt(r2**2-rm**2))

l1 = sphericalLens(r1=10, r2=10, d=0.2, rm=1.1)
lenses=[l1]; numlenses=len(lenses); max_lenses=1; z_len1=24
w,h=2048,2048; pw,ph=10.0,10.0; numm=128; total_threads=w*h*3
z_Object=30; resx,resy=512,512; z_step=0.1; sizex,sizey=10,10

if numlenses>max_lenses: raise ValueError('透镜数量过多')

r1_arr=cp.asarray([l.r1 for l in lenses],dtype=np.float32)
r2_arr=cp.asarray([l.r2 for l in lenses],dtype=np.float32)
z_arr=cp.asarray([z_len1],dtype=np.float32)
d_arr=cp.asarray([l.d for l in lenses],dtype=np.float32)
rm_arr=cp.asarray([l.rm for l in lenses],dtype=np.float32)
sgnr1_arr=cp.asarray([l.signr1 for l in lenses],dtype=np.float32)
sgnr2_arr=cp.asarray([l.signr2 for l in lenses],dtype=np.float32)
zc1_arr=cp.asarray([l.zc1 for l in lenses],dtype=np.float32)
zc2_arr=cp.asarray([l.zc2 for l in lenses],dtype=np.float32)
n_arr=cp.asarray([nr,ng,nb],dtype=np.float32)

# ============ CUDA 源码 ============
cuda_source_code=r'''
__device__ unsigned int hash_rng(unsigned int seed) {
    seed ^= seed >> 16; seed *= 0x7feb352dU;
    seed ^= seed >> 15; seed *= 0x846ca68bU;
    seed ^= seed >> 16; return seed;
}
__device__ float rand_float(unsigned int& seed) {
    seed = hash_rng(seed);
    return (float)seed * 2.3283064365386963e-10f;
}
extern "C" __global__
void render_kernel(
float* __restrict__ final_img, cudaTextureObject_t tex_obj,
const float* __restrict__ r1_arr, const float* __restrict__ r2_arr,
const float* __restrict__ z_arr, const float* __restrict__ d_arr,
const float* __restrict__ rm_arr, const float* __restrict__ sgnr1_arr,
const float* __restrict__ sgnr2_arr, const float* __restrict__ zc1_arr,
const float* __restrict__ zc2_arr, const float* __restrict__ n_arr,
float z_obj,float sizex,float sizey,int resx,int resy,int w,int h,
float pw,float ph,int numm,int total_threads,int numlenses,int frames
){
    const int MAX_LENSES = 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    int pixel_idx = tid / 3;
    int px = pixel_idx % w;
    int py = pixel_idx / w;
    int colorid = tid % 3;
    float n = __ldg(&n_arr[colorid]);
    float inv_w = 1.0f/(float)w; float inv_h = 1.0f/(float)h;
    float inv_sizex = 1.0f/sizex; float inv_sizey = 1.0f/sizey;
    float nn = n*n; float nninv = 1.0f/nn;
    float phys_x = ((float)px*inv_w - 0.5f)*pw;
    float phys_y = (0.5f - (float)py*inv_h)*ph;
    float rm1 = __ldg(&rm_arr[0]); float rmsq = rm1*rm1;
    float initial_z_lens = __ldg(&z_arr[0]);
    float color_sum = 0.0f; float4 pixel;
    float tx,ty,t_obj,fx,fy;
    unsigned int seed = hash_rng(pixel_idx+1+hash_rng(frames));
    float l_zc1[MAX_LENSES],l_zc2[MAX_LENSES],l_r1[MAX_LENSES],l_r2[MAX_LENSES];
    float l_rm[MAX_LENSES],l_s1[MAX_LENSES],l_s2[MAX_LENSES],l_z[MAX_LENSES];
    for(int i=0;i<numlenses&&i<MAX_LENSES;++i){
        l_zc1[i]=__ldg(&zc1_arr[i]); l_zc2[i]=__ldg(&zc2_arr[i]);
        l_r1[i]=__ldg(&r1_arr[i]); l_r2[i]=__ldg(&r2_arr[i]);
        l_rm[i]=__ldg(&rm_arr[i]); l_s1[i]=__ldg(&sgnr1_arr[i]);
        l_s2[i]=__ldg(&sgnr2_arr[i]); l_z[i]=__ldg(&z_arr[i]);
    }
    for(int m=0;m<numm;++m){
        float u=rand_float(seed); float v=rand_float(seed);
        float r_lens=rm1*sqrtf(u); float th=6.28318530718f*v;
        float sin_th,cos_th; sincosf(th,&sin_th,&cos_th);
        float lx=r_lens*cos_th; float ly=r_lens*sin_th;
        float dx=lx-phys_x; float dy=ly-phys_y; float dz=initial_z_lens;
        float inv_d_norm=rsqrtf(dx*dx+dy*dy+dz*dz+1e-10f);
        dx*=inv_d_norm; dy*=inv_d_norm; dz*=inv_d_norm;
        float current_z_lens=initial_z_lens; float ox,oy,oz;
        for(int lensid=0;lensid<numlenses;++lensid){
            float zc1=l_zc1[lensid]; float zc2=l_zc2[lensid];
            float r1=l_r1[lensid]; float r2=l_r2[lensid];
            float r1sq=r1*r1; float r2sq=r2*r2;
            float signr1=l_s1[lensid]; float signr2=l_s2[lensid];
            float rm=l_rm[lensid]; rmsq=rm*rm;
            ox=(lensid==0)?phys_x:ox; oy=(lensid==0)?phys_y:oy;
            oz=-current_z_lens; float ocz=oz-zc1;
            float b=ox*dx+oy*dy+ocz*dz; float c=ox*ox+oy*oy+ocz*ocz-r1sq;
            float delta=b*b-c; if(delta<0.0f) goto next_ray;
            float t_hit1=-b-signr1*sqrtf(delta);
            float x1=ox+t_hit1*dx; float y1=oy+t_hit1*dy; float z1=oz+t_hit1*dz;
            if((x1*x1+y1*y1)>rmsq) goto next_ray;
            if((x1*x1+y1*y1)>1.0f) goto next_ray;
            float temp1=z1-zc1;
            float inv_n_norm=rsqrtf(x1*x1+y1*y1+temp1*temp1);
            float nvx=x1*inv_n_norm; float nvy=y1*inv_n_norm; float nvz=temp1*inv_n_norm;
            float nv=nvx*dx+nvy*dy+nvz*dz;
            if(nv<0.0f){nvx=-nvx;nvy=-nvy;nvz=-nvz;nv=-nv;}
            float x=nv*nv+nn-1.0f; if(x<0.0f) goto next_ray;
            float k=-nv+sqrtf(x);
            float d1x=dx+k*nvx; float d1y=dy+k*nvy; float d1z=dz+k*nvz;
            float inv_d1_norm=rsqrtf(d1x*d1x+d1y*d1y+d1z*d1z+1e-10f);
            d1x*=inv_d1_norm; d1y*=inv_d1_norm; d1z*=inv_d1_norm;
            float ocz2=z1-zc2;
            float bb=x1*d1x+y1*d1y+ocz2*d1z;
            float cc=x1*x1+y1*y1+ocz2*ocz2-r2sq;
            float ddelta=bb*bb-cc; if(ddelta<0.0f) goto next_ray;
            float t_hit2=-bb+signr2*sqrtf(ddelta);
            float x2=x1+t_hit2*d1x; float y2=y1+t_hit2*d1y; float z2=z1+t_hit2*d1z;
            float n2x=x2; float n2y=y2; float n2z=z2-zc2;
            float inv_n2_norm=rsqrtf(n2x*n2x+n2y*n2y+n2z*n2z+1e-10f);
            n2x*=inv_n2_norm; n2y*=inv_n2_norm; n2z*=inv_n2_norm;
            float nv2=n2x*d1x+n2y*d1y+n2z*d1z;
            if(nv2<0.0f){n2x=-n2x;n2y=-n2y;n2z=-n2z;nv2=-nv2;}
            float tmp2=nv2*nv2+nninv-1.0f; if(tmp2<0.0f) goto next_ray;
            float k2=-nv2+sqrtf(tmp2);
            float d2x=d1x+k2*n2x; float d2y=d1y+k2*n2y; float d2z=d1z+k2*n2z;
            float inv_d2_norm=rsqrtf(d2x*d2x+d2y*d2y+d2z*d2z+1e-10f);
            d2x*=inv_d2_norm; d2y*=inv_d2_norm; d2z*=inv_d2_norm;
            ox=x2; oy=y2; dx=d2x; dy=d2y; dz=d2z;
            if(lensid+1<numlenses){float zzz=__ldg(&z_arr[lensid+1]);current_z_lens=zzz-z2;}
            else{current_z_lens=z2;}
        }
        t_obj=(z_obj-current_z_lens)/dz; if(t_obj<=0.0f) goto next_ray;
        fx=ox+t_obj*dx; fy=oy+t_obj*dy;
        tx=(fx*inv_sizex+0.5f)*(float)resx;
        ty=(0.5f-fy*inv_sizey)*(float)resy;
        if(tx>0&&ty>0&&tx<(float)resx&&ty<(float)resy){
            pixel=tex2D<float4>(tex_obj,tx,ty);
            if(colorid==0) color_sum+=pixel.x;
            else if(colorid==1) color_sum+=pixel.y;
            else color_sum+=pixel.z;
        }
        next_ray:;
    }
    final_img[tid] = color_sum / (float)numm;
}
'''

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

render_kernel = cp.RawKernel(cuda_source_code, 'render_kernel', options=('-use_fast_math',))
postprocess_kernel = cp.RawKernel(postprocess_source, 'postprocess_kernel', options=('-use_fast_math',))
print('kernels compiled.')

# ============ 测试图生成 ============
def generate_complex_test_chart(resy, resx):
    chart = np.zeros((resy, resx, 3), dtype=np.float32)
    center = (resx // 2, resy // 2)
    for r in range(0, max(resx, resy), 40):
        cv2.circle(chart, center, r, (0.2, 0.2, 0.2), 1)
    for angle in range(0, 360, 15):
        rad = np.deg2rad(angle)
        p2 = (int(center[0]+1000*np.cos(rad)), int(center[1]+1000*np.sin(rad)))
        cv2.line(chart, center, p2, (0.15, 0.15, 0.15), 1)
    cv2.circle(chart, center, 180, (1.0, 0, 0), 2)
    cv2.circle(chart, center, 170, (0, 1.0, 0), 2)
    cv2.circle(chart, center, 160, (0, 0, 1.0), 2)
    num_spokes = 36; star_radius = 80
    for i in range(num_spokes):
        sa = i*(360/num_spokes); ea = (i+0.5)*(360/num_spokes)
        pts = np.array([center,
            (center[0]+star_radius*np.cos(np.deg2rad(sa)), center[1]+star_radius*np.sin(np.deg2rad(sa))),
            (center[0]+star_radius*np.cos(np.deg2rad(ea)), center[1]+star_radius*np.sin(np.deg2rad(ea)))], np.int32)
        cv2.fillPoly(chart, [pts], (1.0, 1.0, 1.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(chart, "RTX 5070", (20, 40), font, 0.7, (0.8,0.8,0.8), 1)
    cv2.putText(chart, "OPTICAL_TEST_V2", (resx-180, resy-20), font, 0.5, (1.0,0.8,0), 1)
    for x in range(100):
        val = x/100.0
        cv2.line(chart, (resx//2-50+x, resy//2+100), (resx//2-50+x, resy//2+120), (val,val,val), 1)
    for i in range(5):
        dist = i*6
        cv2.line(chart, (50+dist, 100), (50+dist, 200), (1,1,1), 1)
        cv2.line(chart, (resx-100, 100+dist), (resx-50, 100+dist), (1,1,1), 1)
    return chart

img_cpu = generate_complex_test_chart(resy, resx)
img = cp.asarray(img_cpu)

def create_texture_object(img_cp):
    hh, ww, c = img_cp.shape
    bytes_per_pixel = 16; alignment = 256
    pitch_bytes = ((ww*bytes_per_pixel+alignment-1)//alignment)*alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((hh, padded_w, 4), dtype=cp.float32)
    rgba[:, :ww, :3] = img_cp
    ch_fmt = texture.ChannelFormatDescriptor(32,32,32,32, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(runtime.cudaResourceTypePitch2D, arr=rgba, chDesc=ch_fmt, width=ww, height=hh, pitchInBytes=pitch_bytes)
    tex_ptr = texture.TextureDescriptor(addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeBorder),
        borderColors=(0.0,0.0,0.0,0.0), filterMode=runtime.cudaFilterModeLinear, readMode=runtime.cudaReadModeElementType)
    return texture.TextureObject(res_ptr, tex_ptr), rgba

tex_handle, _internal_storage = create_texture_object(img)

# ============ CUDA-GL interop 辅助函数 ============
# cudaGraphicsRegisterFlags
cudaGraphicsRegisterFlagsNone = 0
cudaGraphicsRegisterFlagsWriteDiscard = 2
cudaGraphicsMapFlagsWriteDiscard = 2

class cudaGraphicsResource_p(ctypes.c_void_p):
    pass

def cuda_check(err, msg=""):
    if err != 0:
        raise RuntimeError(f"CUDA error {err}: {msg}")

def register_gl_buffer(gl_buffer_id):
    resource = cudaGraphicsResource_p()
    err = cudart.cudaGraphicsGLRegisterBuffer(
        ctypes.byref(resource),
        ctypes.c_uint(gl_buffer_id),
        ctypes.c_uint(cudaGraphicsRegisterFlagsWriteDiscard))
    cuda_check(err, "cudaGraphicsGLRegisterBuffer")
    return resource

def map_resource(resource):
    err = cudart.cudaGraphicsMapResources(ctypes.c_int(1), ctypes.byref(resource), ctypes.c_void_p(0))
    cuda_check(err, "cudaGraphicsMapResources")

def get_mapped_pointer(resource):
    dev_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    err = cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(size), resource)
    cuda_check(err, "cudaGraphicsResourceGetMappedPointer")
    return dev_ptr.value, size.value

def unmap_resource(resource):
    err = cudart.cudaGraphicsUnmapResources(ctypes.c_int(1), ctypes.byref(resource), ctypes.c_void_p(0))
    cuda_check(err, "cudaGraphicsUnmapResources")

# ============ GLFW + OpenGL 初始化 ============
if not glfw.init():
    raise RuntimeError("Failed to init GLFW")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

window = glfw.create_window(w, h, "Lens Rendering (Zero-Copy)", None, None)
if not window:
    glfw.terminate(); raise RuntimeError("Failed to create GLFW window")
glfw.make_context_current(window)
glfw.swap_interval(0)  # 不限帧率

# 创建 OpenGL PBO
pbo_size = w * h * 4  # RGBA uint8
pbo_id = glGenBuffers(1)
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
glBufferData(GL_PIXEL_UNPACK_BUFFER, pbo_size, None, GL_DYNAMIC_DRAW)
glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

# 创建 OpenGL 纹理
gl_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, gl_tex)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
glBindTexture(GL_TEXTURE_2D, 0)

# 注册 PBO 到 CUDA
cuda_pbo_resource = register_gl_buffer(int(pbo_id))
print("OpenGL + CUDA interop initialized. Zero-copy ready!")

# ============ 渲染缓冲区 ============
accumulated_buffer = cp.zeros((h*w*3), dtype=cp.float32)
current_frame_float = cp.zeros((h*w*3), dtype=cp.float32)
threads_per_block = 256
blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
pp_threads = w * h
pp_blocks = (pp_threads + 255) // 256

# 键盘状态
key_pressed = {}
def key_callback(win, key, scancode, action, mods):
    if action == glfw.PRESS:
        key_pressed[key] = True
    elif action == glfw.RELEASE:
        key_pressed.pop(key, None)
glfw.set_key_callback(window, key_callback)

print("Controls: [W/S] move object, [D/E] move lens, [Shift] 5x speed, [ESC] quit")

frames = 0
running = True

# ============ 主循环 ============
while running and not glfw.window_should_close(window):
    t0 = time.time()

    # --- 光线追踪 kernel ---
    render_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (current_frame_float, tex_handle.ptr,
         r1_arr,r2_arr,z_arr,d_arr,rm_arr,sgnr1_arr,sgnr2_arr,zc1_arr,zc2_arr,n_arr,
         np.float32(z_Object),np.float32(sizex),np.float32(sizey),
         np.int32(resx),np.int32(resy),np.int32(w),np.int32(h),
         np.float32(pw),np.float32(ph),np.int32(numm),np.int32(total_threads),
         np.int32(numlenses),np.int32(frames)))

    accumulated_buffer += current_frame_float
    frames += 1

    # --- 零拷贝: CUDA 直接写入 OpenGL PBO ---
    map_resource(cuda_pbo_resource)
    pbo_dev_ptr, pbo_dev_size = get_mapped_pointer(cuda_pbo_resource)

    # 用 CuPy 包装 PBO 的 GPU 指针 (无拷贝)
    pbo_mem = cp.cuda.UnownedMemory(pbo_dev_ptr, pbo_dev_size, owner=None)
    pbo_memptr = cp.cuda.MemoryPointer(pbo_mem, 0)
    pbo_cupy = cp.ndarray(pbo_dev_size, dtype=cp.uint8, memptr=pbo_memptr)

    # 后处理 kernel: accum → gamma → uint8 RGBA → PBO
    postprocess_kernel(
        (pp_blocks,), (256,),
        (accumulated_buffer, pbo_cupy, np.int32(pp_threads), np.int32(frames)))

    # cp.cuda.Stream.null.synchronize()
    unmap_resource(cuda_pbo_resource)

    # --- OpenGL: PBO → 纹理 → 全屏四边形 ---
    glClear(GL_COLOR_BUFFER_BIT)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
    glBindTexture(GL_TEXTURE_2D, gl_tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0,1); glVertex2f(-1,-1)
    glTexCoord2f(1,1); glVertex2f(1,-1)
    glTexCoord2f(1,0); glVertex2f(1,1)
    glTexCoord2f(0,0); glVertex2f(-1,1)
    glEnd()
    glDisable(GL_TEXTURE_2D)

    glfw.swap_buffers(window)
    glfw.poll_events()

    # --- 键盘处理 ---
    reset = False
    spd = 5*z_step if glfw.KEY_LEFT_SHIFT in key_pressed or glfw.KEY_RIGHT_SHIFT in key_pressed else z_step
    if glfw.KEY_W in key_pressed:
        z_Object += spd; reset = True; print(f"z_Object = {z_Object:.1f}")
    if glfw.KEY_S in key_pressed:
        z_Object -= spd; reset = True; print(f"z_Object = {z_Object:.1f}")
    if glfw.KEY_E in key_pressed:
        z_len1 += spd; z_arr=cp.asarray([z_len1],dtype=np.float32); reset=True; print(f"z_len1 = {z_len1:.1f}")
    if glfw.KEY_D in key_pressed:
        z_len1 -= spd; z_arr=cp.asarray([z_len1],dtype=np.float32); reset=True; print(f"z_len1 = {z_len1:.1f}")
    if glfw.KEY_ESCAPE in key_pressed:
        running = False
    if reset:
        accumulated_buffer.fill(0); frames = 0

    t1 = time.time()
    dt = t1 - t0 + 1e-9
    glfw.set_window_title(window, f"Lens Rendering (Zero-Copy) | {1/dt:.1f} FPS | z={z_Object:.1f} | frames={frames}")

glfw.terminate()
print("Done.")
