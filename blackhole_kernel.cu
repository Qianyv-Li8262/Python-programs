__device__ __forceinline__ float3 normalize(float3 v){
    float inv_norm = __rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
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
    return __sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
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
const float focal_length,const float step

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
float u=__frcp_rn(2.0f * r);
float upl = 1.0f+u;
float umi = 1.0f-u;
float n=upl*upl*upl/umi;
float3 p = d * n;
bool flag = true;


for (int s = 0 ; s < maxstep && flag ; ++s){
float g = -__frcp_rn(r*r*r*umi*umi*umi)*upl*upl*upl*upl*upl*(2.0f-u);
float3 k11 = p;
float3 k12 = g * cam_pos;

float3 k21 = p+(step/2.0f)*k12;
float3 pos_tmp=cam_pos+(step/2.0f)*k11;
r = length(pos_tmp);
u=__frcp_rn(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -__frcp_rn(r*r*r*umi*umi*umi)*upl*upl*upl*upl*upl*(2.0f-u);
float3 k22 = pos_tmp * g;

float3 k31 = p+(step/2.0f)*k22;
pos_tmp=cam_pos+(step/2.0f)*k21;
r = length(pos_tmp);
u=__frcp_rn(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -__frcp_rn(r*r*r*umi*umi*umi)*upl*upl*upl*upl*upl*(2.0f-u);
float3 k32 = pos_tmp * g;

float3 k41 = p + step*k32;
pos_tmp=cam_pos+ step*k31;
r = length(pos_tmp);
u=__frcp_rn(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
g = -__frcp_rn(r*r*r*umi*umi*umi)*upl*upl*upl*upl*upl*(2.0f-u);
float3 k42 = pos_tmp * g;


cam_pos = cam_pos+(step/6.0f)*(k11+2.0f*k21+2.0f*k31+k41);
p = p+(step/6.0f)*(k12+2.0f*k22+2.0f*k32+k42);
r = length(pos_tmp);
if(r<0.501f || r>50.0f){flag = false};
}

}