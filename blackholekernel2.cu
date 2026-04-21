__device__ __forceinline__ float3 normalize(float3 v){
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x*inv_norm , v.y*inv_norm , v.z*inv_norm);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,a.w +b.w);
}

__device__ __forceinline__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float4 operator*(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s,a.w * s);
}
__device__ __forceinline__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __forceinline__ float rand_float(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)seed / 4294967296.0f; // 归一化到 [0, 1)
}

// ============ 体积渲染吸积盘 - 新增函数 ============

// 简单的3D噪声函数（用于模拟湍流和不均匀性）
__device__ float noise3D(float3 p) {
    // 简化的伪随机噪声，基于位置的哈希
    float3 i = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 f = make_float3(p.x - i.x, p.y - i.y, p.z - i.z);
    
    // 平滑插值
    f.x = f.x * f.x * (3.0f - 2.0f * f.x);
    f.y = f.y * f.y * (3.0f - 2.0f * f.y);
    f.z = f.z * f.z * (3.0f - 2.0f * f.z);
    
    // 简单哈希
    float n = i.x + i.y * 57.0f + i.z * 113.0f;
    float hash = sinf(n * 43758.5453f);
    return hash * 0.5f + 0.5f; // [0, 1]
}


// 黑体辐射颜色近似（温度 -> RGB）
__device__ float3 temperature_to_color(float temp) {
    // temp: 温度（开尔文），范围 1000K - 40000K
    // 修正版 Tanner Helland 算法
    
    // 注意：原算法的拟合公式是基于 temp / 100.0f 计算的！
    float t = temp / 100.0f; 
    float r, g, b;
    
    // --- Red ---
    if (t <= 66.0f) {
        r = 1.0f;
    } else {
        r = 1.292936186f * powf(t - 60.0f, -0.1332047592f);
        r = fminf(1.0f, fmaxf(0.0f, r));
    }
    
    // --- Green ---
    if (t <= 66.0f) {
        g = 0.39008157876f * logf(t) - 0.631841444f;
        g = fminf(1.0f, fmaxf(0.0f, g));
    } else {
        g = 1.129890861f * powf(t - 60.0f, -0.0755148492f);
        g = fminf(1.0f, fmaxf(0.0f, g));
    }
    
    // --- Blue ---
    if (t >= 66.0f) {
        b = 1.0f;
    } else if (t <= 19.0f) { // 原算法中低于 1900K 蓝光为 0
        b = 0.0f;
    } else {
        b = 0.543206789f * logf(t - 10.0f) - 1.196254089f;
        b = fminf(1.0f, fmaxf(0.0f, b));
    }
            r = fminf(1.0f, fmaxf(0.0f, r));
    g = fminf(1.0f, fmaxf(0.0f, g));
    b = fminf(1.0f, fmaxf(0.0f, b));
    return make_float3(r, g, b);
}






// 吸积盘密度函数
__device__ float disk_density(float3 pos, float r_disk) {
    // 参数说明：
    // pos: 当前位置
    // r_disk: 到旋转轴的距离 sqrt(x^2 + y^2)
    
    // 1. 垂直方向：高斯衰减（模拟盘的厚度）
    float z_scale = 0.15f;  // 厚度参数，越小盘越薄
    float vertical_density = expf(-fabsf(pos.z) / z_scale);
    
    // 2. 径向方向：幂律衰减（内圈密度高，外圈密度低）
    float radial_density = powf(r_disk / 3.0f, -1.5f);
    
    // 3. 添加湍流噪声（可选，增加真实感）
    float noise_scale = 3.0f;
    float turbulence = noise3D(make_float3(pos.x * noise_scale, pos.y * noise_scale, pos.z * noise_scale));
    turbulence = 0.7f + 0.6f * turbulence; // [0.7, 1.3]
    
    // 4. 螺旋结构（可选）
    float phi = atan2f(pos.y, pos.x);
    float spiral_arms = 3.0f; // 螺旋臂数量
    float spiral_pattern = 1.0f + 0.3f * sinf(spiral_arms * phi - r_disk * 0.5f);
    
    // 组合所有因素
    float density = vertical_density * radial_density * turbulence * spiral_pattern;
    
    // 密度缩放因子（控制整体不透明度）
    density *= 0.6f;
    
    return fmaxf(0.0f, density);
}

// 吸积盘温度函数
__device__ float disk_temperature(float r_disk) {
    // 标准薄盘温度分布：T ∝ r^(-3/4)
    // 内圈高温（蓝白色），外圈低温（橙红色）
    
    float T0 = 4000.0f; // 参考温度（开尔文）
    float r0 = 3.0f;    // 参考半径
    
    float temp = T0 * powf(r_disk / r0, -0.75f);
    
    // 限制温度范围
    temp = fminf(12000.0f, fmaxf(2000.0f, temp));
    
    return temp;
}

// 计算吸积盘在某点的发射颜色和强度
__device__ float4 disk_emission(float3 pos, float r_disk) {
    // 1. 计算温度
    float temp = disk_temperature(r_disk);
    
    // 2. 温度转颜色
    float3 color = temperature_to_color(temp);
    
    // 3. 发射强度（内圈更亮）
    float intensity = 1.5f / (r_disk - 0.9f);
    intensity = fminf(3.0f, intensity);
    
    // 4. 添加一些变化（模拟热点）
    float noise_val = noise3D(make_float3(pos.x * 2.0f, pos.y * 2.0f, pos.z * 2.0f));
    intensity *= (0.8f + 0.4f * noise_val);
    
    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
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
const float focal_length,const float step,const int maxstep,const int jitternum

){



//打包成向量
// float3 cam_pos=make_float3(cam_pos_x,cam_pos_y,cam_pos_z);
// float r = length(cam_pos);
float3 fwd = make_float3(fwd_x,fwd_y,fwd_z);
float3 right = make_float3(right_x,right_y,right_z);
float3 up = make_float3(up_x,up_y,up_z);



int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

if( pixel_idx >= imgwidth || pixel_idy >= imgheight ) return;
float4 buffer=make_float4(0.0f,0.0f,0.0f,0.0f);
// float jitterx = rand_float((unsigned int)pixel_idx);
// float jittery = rand_float((unsigned int)pixel_idy);
// float physical_x = (((float)pixel_idx+jitterx)/(float)imgwidth - 0.5f) * physwidth;
// float physical_y = (((float)pixel_idy+jittery)/(float)imgheight - 0.5f) * physheight;
float jitterx;
float jittery;
float physical_x;
float physical_y;
for(int i = 0;i < jitternum;++i){

jitterx = rand_float((unsigned int)pixel_idx+i);
jittery = rand_float((unsigned int)pixel_idy+i+12345);
physical_x = (((float)pixel_idx+jitterx)/(float)imgwidth - 0.5f) * physwidth;
physical_y = (((float)pixel_idy+jittery)/(float)imgheight - 0.5f) * physheight;
float3 cam_pos=make_float3(cam_pos_x,cam_pos_y,cam_pos_z);
float r = length(cam_pos);
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
float4 accumulated_color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
// bool hit_disk=false;


for (int s = 0 ; s < maxstep && flag ; ++s){
    float3 prev_pos = cam_pos;

    
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

// ============ 体积渲染吸积盘（新方法）============
// 计算到旋转轴的距离
float r_disk = sqrtf(cam_pos.x * cam_pos.x + cam_pos.y * cam_pos.y);

// 检查是否在吸积盘体积内
// 参数可调：内半径 1.5，外半径 8.0，厚度范围约 ±0.5
if (r_disk > 1.5f && r_disk < 8.0f && fabsf(cam_pos.z) < 0.5f) {
    // 1. 计算该点的密度
    float density = disk_density(cam_pos, r_disk);
    
    // 2. 计算发射颜色（基于温度）
    float4 emission = disk_emission(cam_pos, r_disk);
    
    // 3. 体积渲染累积（Beer-Lambert定律）
    // step_opacity: 这一小段路径的不透明度
    float step_opacity = density * current_step * 0.3f; // 0.3是调节因子
    step_opacity = fminf(step_opacity, 1.0f); // 限制最大值
    
    // 4. 前向累积（front-to-back compositing）
    float transmittance = 1.0f - accumulated_color.w; // 剩余透射率
    accumulated_color.x += emission.x * step_opacity * transmittance;
    accumulated_color.y += emission.y * step_opacity * transmittance;
    accumulated_color.z += emission.z * step_opacity * transmittance;
    accumulated_color.w += step_opacity * transmittance;
    
    // 5. 提前终止（如果已经完全不透明）
    if (accumulated_color.w > 0.99f) {
        flag = false;
    }
}

// 终止条件：掉入黑洞、飞出边界、或数值异常
if(r<0.55f || r>50.0f || isnan(r)) {flag = false;}
}

float4 color;
if (r >=0.55f && !isnan(r)) {



float3 final_dir = normalize(p);

    float phi = atan2f(final_dir.y, -final_dir.x); 
    float theta = asinf(-final_dir.z);

    float tex_u = phi*0.1591549f+0.5f;
    float tex_v = theta* 0.3183099f+0.5f;
    

    // v = 1.0f - v; 
    
    // 5. 使用 CUDA 硬件纹理采样 (tex2D)
    // tex2D 会自动处理双线性插值和边界环绕
    float4 bkgd = tex2D<float4>(tex_obj, tex_u, tex_v);
    color = accumulated_color + bkgd * (1.0f - accumulated_color.w);
    // // 6. 写入显存 (假设 raw_img 是 float 类型的 RGB 数组)
    // int pixel_index = (pixel_idy * imgwidth + pixel_idx) * 3;
    // raw_img[pixel_index + 0] = color.x; // R
    // raw_img[pixel_index + 1] = color.y; // G
    // raw_img[pixel_index + 2] = color.z; // B




} else {
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 0] = 0.0f;
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 1] = 0.0f;
    // raw_img[(pixel_idy * imgwidth + pixel_idx) * 3 + 2] = 0.0f;
color = accumulated_color + make_float4(0.0f, 0.0f, 0.0f, 1.0f) * (1.0f - accumulated_color.w);
}

buffer = buffer+color;
}
buffer = buffer*(1.0f/(float)jitternum);
    int pixel_index = (pixel_idy * imgwidth + pixel_idx) * 3;
    raw_img[pixel_index + 0] = buffer.x; // R
    raw_img[pixel_index + 1] = buffer.y; // G
    raw_img[pixel_index + 2] = buffer.z; // B
}
