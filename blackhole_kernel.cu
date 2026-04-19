__device__ __forceinline__ float3 normalize(float3 v){
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x*inv_norm , v.y*inv_norm , v.z*inv_norm);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

extern "C" __global__
void blackholekernel(
float* __restrict__ raw_img,
cudaTextureObject_t tex_obj,
const float cam_pos_x,
const float cam_pos_y,
const float cam_pos_z,
const float fwd_x,
const float fwd_y,
const float fwd_z,
const float right_x,
const float right_y,
const float right_z,
const float up_x,
const float up_y,
const float up_z,
const int imgwidth,const int imgheight,
const float physwidth,const float physheight,
const float focal_length,const float step,const int maxstep

){



//打包成向量
float3 cam_pos=make_float3(cam_pos_x,cam_pos_y,cam_pos_z);
float r = length(cam_pos);
float3 fwd = make_float3(fwd_x,fwd_y,fwd_z);
float3 right = make_float3(right_x,right_y,right_z);
float3 up = make_float3(up_x,up_y,up_z);



int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

if( pixel_idx >= imgwidth || pixel_idy >= imgheight ) return;

float physical_x = (((float)pixel_idx+0.5f)/(float)imgwidth - 0.5f) * physwidth;
float physical_y = (((float)pixel_idy+0.5f)/(float)imgheight - 0.5f) * physheight;

float3 d = make_float3(
    fwd.x * focal_length - right.x * physical_x - up.x * physical_y,
    fwd.y * focal_length - right.y * physical_x - up.y * physical_y,
    fwd.z * focal_length - right.z * physical_x - up.z * physical_y
    );
// 出射单位向量
d=normalize(d);
float u=1.0f/(2.0f * r);
float upl = 1.0f+u;
float umi = 1.0f-u;
float n=upl*upl*upl/umi;
float3 p = d * n;
bool flag = true;


for (int s = 0 ; s < maxstep && flag ; ++s){
    //step 1
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
float g = -1.0f/(r*r*r*umi*umi*umi)*upl*(2.0f-u);
float uu=1.0f/(upl*upl*upl*upl);
float3 k11 = p * uu;
float3 k12 = g * cam_pos;

//自适应步长

        // float n_current = upl * upl * upl / umi;
        // // 限制真实空间的移动步幅：越靠近视界 (0.5) 步幅越小，防止“瞬移过冲”
        // float spatial_step = step * fminf(5.0f, fmaxf(0.05f, r - 0.505f)); 
        // // 将空间步长转换为 RK4 需要的参数步长
        // float current_step = spatial_step / n_current; 

// float current_step = step * fmaxf(0.02f, (r - 0.55f)); // 离黑洞越近，步长越小
float current_step = step * fminf(10.0f, fmaxf(0.05f, (r - 0.55f))); 

//step 2
float3 pos_tmp=cam_pos+(current_step/2.0f)*k11;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -1.0f/(r*r*r*umi*umi*umi)*upl*(2.0f-u);
uu=1.0f/(upl*upl*upl*upl);
float3 k21 = (p+(current_step/2.0f)*k12)*uu;
float3 k22 = pos_tmp * g;

//step 3
pos_tmp=cam_pos+(current_step/2.0f)*k21;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -1.0f/(r*r*r*umi*umi*umi)*upl*(2.0f-u);
uu=1.0f/(upl*upl*upl*upl);
float3 k31 = (p+(current_step/2.0f)*k22)*uu;
float3 k32 = pos_tmp * g;

//step 4
pos_tmp=cam_pos+ current_step*k31;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -1.0f/(r*r*r*umi*umi*umi)*upl*(2.0f-u);
uu=1.0f/(upl*upl*upl*upl);
float3 k41 = (p + current_step*k32)*uu;
float3 k42 = pos_tmp * g;

//concatenate
cam_pos = cam_pos+(current_step/6.0f)*(k11+2.0f*k21+2.0f*k31+k41);
p = p+(current_step/6.0f)*(k12+2.0f*k22+2.0f*k32+k42);
r = length(cam_pos);

if(r<0.55f || r>50.0f || isnan(r)){flag = false;}
}
// if (r < 0.501f) {
//     // 掉进黑洞，涂黑
//     raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 0] = 0.0f;
//     raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 1] = 0.0f;
//     raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 2] = 0.0f;

// } else {
// float3 final_dir = normalize(p);

//     float phi = atan2f(final_dir.z, final_dir.x); 
//     float theta = asinf(final_dir.y);

//     float u = (phi + 3.14159265f) * 0.1591549f;
//     float v = (theta + 1.57079633f) * 0.3183099f;
    

//     // v = 1.0f - v; 
    
//     // 5. 使用 CUDA 硬件纹理采样 (tex2D)
//     // tex2D 会自动处理双线性插值和边界环绕
//     float4 color = tex2D<float4>(tex_obj, u, v);
    
//     // 6. 写入显存 (假设 raw_img 是 float 类型的 RGB 数组)
//     int pixel_index = (pixel_idy * imgwidth + pixel_idx) * 3;
//     raw_img[pixel_index + 0] = color.x; // R
//     raw_img[pixel_index + 1] = color.y; // G
//     raw_img[pixel_index + 2] = color.z; // B
if (r >=0.55f && !isnan(r)) {
    // 掉进黑洞，涂黑


float3 final_dir = normalize(p);

    float phi = atan2f(final_dir.y, -final_dir.x); 
    float theta = asinf(-final_dir.z);

    float tex_u = phi*0.1591549f+0.5f;
    float tex_v = theta* 0.3183099f+0.5f;
    

    // v = 1.0f - v; 
    
    // 5. 使用 CUDA 硬件纹理采样 (tex2D)
    // tex2D 会自动处理双线性插值和边界环绕
    float4 color = tex2D<float4>(tex_obj, tex_u, tex_v);
    
    // 6. 写入显存 (假设 raw_img 是 float 类型的 RGB 数组)
    int pixel_index = (pixel_idy * imgwidth + pixel_idx) * 3;
    raw_img[pixel_index + 0] = color.x; // R
    raw_img[pixel_index + 1] = color.y; // G
    raw_img[pixel_index + 2] = color.z; // B




} else {
    raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 0] = 0.0f;
    raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 1] = 0.0f;
    raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 2] = 0.0f;
}
}