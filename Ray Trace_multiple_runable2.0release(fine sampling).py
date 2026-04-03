import numpy as np
import cupy as cp
import cv2
import time
print('import succeed.')
nr=1.5
ng=1.5
nb=1.5
class sphericalLens:
    def __init__(self, r1, r2, z, d, rm):
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.d = float(d)
        self.rm = float(rm)
        self.signr1 = float(np.sign(r1))
        self.signr2 = float(np.sign(r2))
        self.z = float(z)
        self.zc1 = float(-d/2 + np.sign(r1)*np.sqrt(r1**2 - rm**2))
        self.zc2 = float(d/2 - np.sign(r2)*np.sqrt(r2**2 - rm**2))
l1 = sphericalLens(r1=10, r2=10, z=90, d=0, rm=3) 

w, h = 500, 500
pw, ph = 27.0, 27.0
numm = 512
total_threads = w * h * 3 

from scipy.stats import qmc
sampler = qmc.Sobol(d=2, scramble=False)  
samples = sampler.random(w * h * numm).astype(np.float32)  
u_arr = cp.asarray(samples[:, 0])
v_arr = cp.asarray(samples[:, 1])
del samples
z_Object = 8.9        
resx,resy = 512,512
z_step = 0.1             
sizex,sizey = 10,10

lenses=[l1]
numlenses =len(lenses)
r1_arr=cp.asarray([l.r1 for l in lenses],dtype=np.float32)
r2_arr=cp.asarray([l.r2 for l in lenses],dtype=np.float32)
z_arr=cp.asarray([l.z for l in lenses],dtype=np.float32)
d_arr=cp.asarray([l.d for l in lenses],dtype=np.float32)
rm_arr=cp.asarray([l.rm for l in lenses],dtype=np.float32)
sgnr1_arr=cp.asarray([l.signr1 for l in lenses],dtype=np.float32)
sgnr2_arr=cp.asarray([l.signr2 for l in lenses],dtype=np.float32)
zc1_arr=cp.asarray([l.zc1 for l in lenses],dtype=np.float32)
zc2_arr=cp.asarray([l.zc2 for l in lenses],dtype=np.float32)
n_arr=cp.asarray([nr,ng,nb],dtype=np.float32)

cuda_source_code='''
extern "C" __global__
void render_kernel(
unsigned char* __restrict__ final_img,
const float* __restrict__ img,
const float* __restrict__ u_arr,
const float* __restrict__ v_arr,
const float* __restrict__ r1_arr,
const float* __restrict__ r2_arr,
const float* __restrict__ z_arr,
const float* __restrict__ d_arr,
const float* __restrict__ rm_arr,
const float* __restrict__ sgnr1_arr,
const float* __restrict__ sgnr2_arr,
const float* __restrict__ zc1_arr,
const float* __restrict__ zc2_arr,
const float* __restrict__ n_arr,
float z_obj,float sizex,float sizey,int resx,int resy,int w,int h,float pw,float ph,int numm,int total_threads,int numlenses
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    
    int pixel_idx = tid / 3; 
    int px = pixel_idx % w;     
    int py = pixel_idx / w;
    int colorid = tid % 3;

    float n = __ldg(&n_arr[colorid]);
    float inv_w = 1.0f / (float)w;
    float inv_h = 1.0f / (float)h;
    float inv_sizex = 1.0f / sizex;
    float inv_sizey = 1.0f / sizey;
    float nn = n * n;
    float nninv = 1.0f / nn;

    float phys_x = ((float)px * inv_w - 0.5f) * pw;
    float phys_y = (0.5f - (float)py * inv_h) * ph;

    float rm1 = __ldg(&rm_arr[0]);
    float rmsq = rm1*rm1;
    float initial_z_lens = __ldg(&z_arr[0]);

    float color_sum = 0.0f;

    for(int m = 0; m < numm; ++m){
        int sample_idx = pixel_idx * numm + m;
        float u = __ldg(&u_arr[sample_idx]);
        float v = __ldg(&v_arr[sample_idx]);

        float r_lens = rm1 * sqrtf(u);
        float th = 6.28318530718f * v; 
        float sin_th, cos_th;
        sincosf(th, &sin_th, &cos_th);
        float lx = r_lens * cos_th;
        float ly = r_lens * sin_th;

        float dx = lx - phys_x;
        float dy = ly - phys_y;
        float dz = initial_z_lens;

        float inv_d_norm = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-10f);
        dx *= inv_d_norm; dy *= inv_d_norm; dz *= inv_d_norm;
        
        float current_z_lens = initial_z_lens;
        float ox, oy, oz;

        for (int lensid = 0; lensid < numlenses; ++lensid){
            float zc1 = __ldg(&zc1_arr[lensid]);
            float zc2 = __ldg(&zc2_arr[lensid]);
            float r1 = __ldg(&r1_arr[lensid]);
            float r2 = __ldg(&r2_arr[lensid]);
            float r1sq = r1 * r1;
            float r2sq = r2 * r2;
            float signr1 = __ldg(&sgnr1_arr[lensid]);
            float signr2 = __ldg(&sgnr2_arr[lensid]);
            float rm = __ldg(&rm_arr[lensid]);
            float rmsq = rm * rm;

            ox = (lensid == 0) ? phys_x : ox;
            oy = (lensid == 0) ? phys_y : oy;
            oz = -current_z_lens;
            float ocz = oz - zc1;
            
            float b = ox * dx + oy * dy + ocz * dz;
            float c = ox * ox + oy * oy + ocz * ocz - r1sq;

            float delta = b * b - c;
            if ( delta < 0.0f ) goto next_ray;

            float t_hit1 = -b - signr1 * sqrtf(delta);
            float x1 = ox + t_hit1 * dx;
            float y1 = oy + t_hit1 * dy;
            float z1 = oz + t_hit1 * dz;
            if ((x1 * x1 + y1 * y1) > rmsq) goto next_ray;

            float temp1 = z1 - zc1;
            float inv_n_norm = rsqrtf(x1 * x1 + y1 * y1 + temp1 * temp1);
            float nvx = x1 * inv_n_norm;
            float nvy = y1 * inv_n_norm;
            float nvz = temp1 * inv_n_norm;
            float nv = nvx * dx + nvy * dy + nvz * dz;
            if (nv<0.0f){ nvx=-nvx; nvy=-nvy; nvz=-nvz; nv=-nv; }
            
            float x = nv * nv + nn - 1.0f;
            if ( x < 0.0f ) goto next_ray;
            float k = -nv + sqrtf(x);
            float d1x = dx + k * nvx;
            float d1y = dy + k * nvy;
            float d1z = dz + k * nvz;
            float inv_d1_norm = rsqrtf(d1x * d1x + d1y * d1y + d1z * d1z + 1e-10f);
            d1x *= inv_d1_norm; d1y *= inv_d1_norm; d1z *= inv_d1_norm;

            float ocz2 = z1 - zc2;
            float bb = x1 * d1x + y1 * d1y + ocz2 * d1z;
            float cc = x1 * x1 + y1 * y1 + ocz2 * ocz2 - r2sq;
            float ddelta = bb * bb - cc;
            if (ddelta<0.0f) goto next_ray;
            
            float t_hit2 = -bb + signr2 * sqrtf(ddelta);
            float x2 = x1 + t_hit2 * d1x;
            float y2 = y1 + t_hit2 * d1y;
            float z2 = z1 + t_hit2 * d1z;

            float n2x = x2;
            float n2y = y2;
            float n2z = z2 - zc2;
            float inv_n2_norm = rsqrtf(n2x*n2x + n2y*n2y + n2z*n2z + 1e-10f);
            n2x *= inv_n2_norm; n2y *= inv_n2_norm; n2z *= inv_n2_norm;

            float nv2 = n2x * d1x + n2y * d1y + n2z * d1z;
            if (nv2 < 0.0f) { n2x = -n2x; n2y = -n2y; n2z = -n2z; nv2 = -nv2; }
            float tmp2 = nv2 * nv2 + nninv - 1.0f;
            if (tmp2 < 0.0f) goto next_ray; 

            float k2 = -nv2 + sqrtf(tmp2);
            float d2x = d1x + k2 * n2x;
            float d2y = d1y + k2 * n2y;
            float d2z = d1z + k2 * n2z;
            
            float inv_d2_norm = rsqrtf(d2x*d2x + d2y*d2y + d2z*d2z + 1e-10f);
            d2x *= inv_d2_norm; d2y *= inv_d2_norm; d2z *= inv_d2_norm;

            ox = x2;
            oy = y2;
            dx = d2x;
            dy = d2y;
            dz = d2z;

            if (lensid + 1 < numlenses) {
                float zzz = __ldg(&z_arr[lensid+1]);
                current_z_lens = zzz - z2;
            } else {
                current_z_lens = z2;
            }
        }

        float t_obj = (z_obj - current_z_lens) / dz;
        if (t_obj <= 0.0f) goto next_ray;

        float fx = ox + t_obj * dx;
        float fy = oy + t_obj * dy;

        int ix = (int)((fx * inv_sizex + 0.5f) * resx);
        int iy = (int)((0.5f - fy * inv_sizey) * resy);

        if (ix >= 0 && ix < resx && iy >= 0 && iy < resy) {
            int img_idx = (iy * resx + ix) * 3;
            color_sum += __ldg(&img[img_idx + colorid]);
        }

    next_ray:;
    }
    float avg_color = color_sum / (float)numm;
    float gamma_corrected = powf(avg_color, 0.454545f);
    float scaled = gamma_corrected * 255.0f;
    scaled = fmaxf(0.0f, fminf(scaled, 255.0f));
    final_img[tid] = (unsigned char)scaled;
}
'''
render_fused_kernel = cp.RawKernel(cuda_source_code, 'render_kernel',options=('-use_fast_math',))
print('defs ended')
img_cpu = np.zeros((resy, resx, 3), dtype=np.float32)
cv2.putText(img_cpu, ". . .", (190, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (1.0, 1.0, 0.0), 8)
img = cp.asarray(img_cpu)
final_img = cp.zeros((w * h * 3), dtype=cp.uint8)
threads_per_block = 256
blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
cv2.namedWindow("Lens Rendering")
running = True
print("Controls: [W] move forward, [ESC] quit")
while running:
    t0 = time.time()
    render_fused_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (final_img, img, u_arr,v_arr,
         r1_arr,r2_arr,z_arr,d_arr,rm_arr,sgnr1_arr,sgnr2_arr,zc1_arr,zc2_arr,n_arr,np.float32(z_Object),np.float32(sizex),np.float32(sizey),np.int32(resx),np.int32(resy),np.int32(w),np.int32(h),
         np.float32(pw),np.float32(ph),np.int32(numm),np.int32(total_threads),np.int32(numlenses))
    )

    t1 = time.time()
    print(f'Render time (z={z_Object:.1f}): {t1-t0:.4f}s')
    render_img = cp.asnumpy(final_img).reshape((h, w, 3))
    render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Lens Rendering", render_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        z_Object += z_step
        print(f"Moving forward → z_Object = {z_Object:.1f}")
    elif key == ord('s'):
        z_Object -= z_step
        print(f"Moving backward → z_Object = {z_Object:.1f}")
    elif key == 27: 
        running = False
        print("Exiting...")

cv2.destroyAllWindows()