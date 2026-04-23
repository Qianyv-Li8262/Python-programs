__device__ __forceinline__ float3 normalize(float3 v){
    float inv_norm = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x*inv_norm , v.y*inv_norm , v.z*inv_norm);
}

__device__ __forceinline__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
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

// 取小数部分
__device__ __forceinline__ float fract(float x) { 
    return x - floorf(x); 
}


__device__ float hash11(float n) {
    return fract(sinf(n) * 43758.5453123f);
}


__device__ float noise1D(float x) {
    float i = floorf(x);
    float f = fract(x);
    

    float u = f * f * (3.0f - 2.0f * f);
    

    return hash11(i) * (1.0f - u) + hash11(i + 1.0f) * u;
}

__device__ float fbm_sharp(float r) {
    float v = 0.0f;
    float amp = 1.0f;       // 初始振幅
    float freq = 1.0f;      // 初始频率
    float max_amp = 0.0f;   // 用于累加理论最大值，进行严格归一化
    
    // 叠加 5 阶细节 (Octaves)
    for(int i = 0; i < 5; i++) {
        // 1. 获取平滑噪声，并重新映射到 [-1.0, 1.0]
        float n = noise1D(r * freq) * 2.0f - 1.0f;
        
        // 2. 绝对值制造“锐利折角”，翻转制造“亮光尖峰” (Ridged 操作)
        n = 1.0f - fabsf(n); 
        
        // 3. 可选：平方操作，让亮的环变窄变锐利，暗的缝隙变宽
        n = n * n; 
        
        // 累加到总值
        v += n * amp;
        
        // 记录此时可能达到的最大值
        max_amp += amp; 
        
        // 频率翻倍 (Lacunarity)，振幅减半 (Gain)
        freq *= 2.0f;  
        amp *= 0.5f;   
    }
    
    // 4. 完美归一化：除以理论最大值，强行锁死在 [0.0, 1.0]
    return v / max_amp; 
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
    float z_scale = 0.04f;  // 厚度参数，越小盘越薄
    float vertical_density = expf(-fabsf(pos.z) / z_scale);
    
    // 2. 径向方向：幂律衰减（内圈密度高，外圈密度低）
    float radial_density = powf((r_disk-1.4f) / 0.1f, -0.3f);

    float sharp_noise = fbm_sharp(r_disk * 1.5f); 
    
    // 关键：利用 pow() 调整对比度
    // pow(x, 1.0): 正常的环
    // pow(x, 3.0): 极细且极其锐利的亮环，背景是黑缝隙
    float rings = powf(sharp_noise, 2.0f); 
    // // 4. 螺旋结构（可选）
    // float phi = atan2f(pos.y, pos.x);
    // float spiral_arms = 3.0f; // 螺旋臂数量
    // float spiral_pattern = 1.0f + 0.3f * sinf(spiral_arms * phi - r_disk * 0.5f);
    float spiral_pattern=1.0f;
    // 组合所有因素
    float density = vertical_density * radial_density * rings * spiral_pattern;
    
    // 密度缩放因子（控制整体不透明度）
    density *= 1.2f;
    
    return fmaxf(0.0f, density);
}

// 吸积盘温度函数
__device__ float disk_temperature(float r_disk) {
    // 标准薄盘温度分布：T ∝ r^(-3/4)
    // 内圈高温（蓝白色），外圈低温（橙红色）
    
    float T0 = 13000.0f; // 参考温度（开尔文）
    // float r0 = 3.0f;    // 参考半径
    
    float temp = T0 *powf(r_disk/2.0f,-2.0f)*powf(1.0f-sqrtf(1.5f/r_disk),0.25f);
    
    // // 限制温度范围
    temp = fminf(2500.0f, fmaxf(1000.0f, temp));
    // float temp=1500.0f;
    return temp;
}

// 计算吸积盘在某点的发射颜色和强度
__device__ float4 disk_emission(float temp,float intensity,cudaTextureObject_t lut_color,float g) {


    float4 color = tex2D<float4>(lut_color,(temp-1000.0f)/20000.0f,0.5f);
    
    // 3. 发射强度（内圈更亮）
    // float intensity = 10.0f*powf(4/(r_disk-1.3f),2.0f);
    // intensity = fminf(3.0f, intensity);
    // intensity = fminf(20.0f, fmaxf(0.0f, intensity));

    
    return make_float4(color.x * intensity, color.y * intensity, color.z * intensity, 1.0f);
}

extern "C" __global__
void blackholekernel(
float* __restrict__ raw_img,
cudaTextureObject_t tex_obj,
cudaTextureObject_t lut_physics,
cudaTextureObject_t lut_color,
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
const float focal_length,const float step,const int maxstep,const int jitternum,const int frames

){




float3 fwd = make_float3(fwd_x,fwd_y,fwd_z);
float3 right = make_float3(right_x,right_y,right_z);
float3 up = make_float3(up_x,up_y,up_z);



int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;

if( pixel_idx >= imgwidth || pixel_idy >= imgheight ) return;
float4 buffer=make_float4(0.0f,0.0f,0.0f,0.0f);


float jitterx;
float jittery;
float physical_x;
float physical_y;
for(int i = 0;i < jitternum;++i){

jitterx = rand_float((unsigned int)pixel_idx+i+frames);
jittery = rand_float((unsigned int)pixel_idy+i+12345+frames);
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

float rmhalf = r-0.5f;
float g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
float uplsq=upl*upl;
float uu=1.0f/(uplsq*uplsq);
float3 k11 = p * uu;
float3 k12 = g * cam_pos;

//自适应步长

// float current_step = step * fminf(10.0f, fmaxf(0.05f, (r - 0.95f))); 



// if (r > 1.2f && r < 18.0f && fabsf(cam_pos.z) < 2.0f) {

//     float z_factor = fabsf(cam_pos.z) / 2.0f;
    

//     float multiplier = 0.05f + 0.15f * (z_factor * z_factor); 
    
//     current_step *= multiplier;
// }

bool in_disk_volume = (r > 4.8f && r < 18.0f && fabsf(cam_pos.z) < 2.0f); 
// in_disk_volume 为 true 时(1.0)，应用 0.05f，否则为 1.0f
float zone_multiplier = in_disk_volume ? (0.05f + 0.15f * (cam_pos.z * cam_pos.z * 0.25f)) : 1.0f;
float current_step = step * fminf(10.0f, fmaxf(0.05f, r - 0.54f)) * zone_multiplier;

// if (r > 1.4f && r < 17.0f && fabsf(cam_pos.z) < 0.7f){
//     current_step *=0.05f;
// }

//step 2
float3 pos_tmp=cam_pos+(current_step/2.0f)*k11;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
rmhalf = r-0.5f;
g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
uplsq=upl*upl;
uu=1.0f/(uplsq*uplsq);
float3 k21 = (p+(current_step/2.0f)*k12)*uu;
float3 k22 = pos_tmp * g;

//step 3
pos_tmp=cam_pos+(current_step/2.0f)*k21;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
rmhalf = r-0.5f;
g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
uplsq=upl*upl;
uu=1.0f/(uplsq*uplsq);
float3 k31 = (p+(current_step/2.0f)*k22)*uu;
float3 k32 = pos_tmp * g;

//step 4
pos_tmp=cam_pos+ current_step*k31;
r = length(pos_tmp);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
rmhalf = r-0.5f;
g = -upl*(2.0f-u)/(rmhalf*rmhalf*rmhalf);
uplsq=upl*upl;
uu=1.0f/(uplsq*uplsq);
float3 k41 = (p + current_step*k32)*uu;
float3 k42 = pos_tmp * g;

//concatenate
cam_pos = cam_pos+(current_step/6.0f)*(k11+2.0f*k21+2.0f*k31+k41);
p = p+(current_step/6.0f)*(k12+2.0f*k22+2.0f*k32+k42);
r = length(cam_pos);
u=1.0f/(2.0f * r);
upl = 1.0f+u;
umi = 1.0f-u;
float3 temp = make_float3((cam_pos.x+prev_pos.x)/2.0f,(cam_pos.y+prev_pos.y)/2.0f,0.0f);
// float r_disk = sqrtf(cam_pos.x * cam_pos.x + cam_pos.y * cam_pos.y);
float r_disk_sq = temp.x * temp.x + temp.y * temp.y;
bool indisk = (r_disk_sq > 24.4974f && r_disk_sq < 272.25f && fabsf(cam_pos.z) < 0.5f);

// if (__any_sync(0xFFFFFFFF, indisk)) {
if (indisk) {
float r_disk=sqrtf(r_disk_sq);
    float4 parameters = tex2D<float4>(lut_physics,(r_disk-4.9495f)/11.5505f,fabsf(cam_pos.z)/0.5f);
    
    

    float4 emission = disk_emission(parameters.y,parameters.z,lut_color,1.0f);
    
    float ravg = (length(prev_pos)+r)/2.0f;
    float uuu=1.0f+1.0f/(2.0f*ravg);
    // float intensity_factor = fminf(1.0f, parameters.z * 5.0f); 
    // float S = 0.3f; 
    // float x = fmaxf(0.0f, fminf(1.0f, parameters.z / S));
    // float intensity_factor = x * x * (3.0f - 2.0f * x);
    float k = 2.0f; 
    float intensity_factor = 1.0f - __expf(-(k * parameters.z)*(k * parameters.z));
    float step_opacity = parameters.x * 1.7f*uuu*uuu*length(cam_pos-prev_pos)* intensity_factor;
    step_opacity = fminf(step_opacity, 1.0f);
    

    float transmittance = 1.0f - accumulated_color.w;
    accumulated_color.x += emission.x * step_opacity * transmittance;
    accumulated_color.y += emission.y * step_opacity * transmittance;
    accumulated_color.z += emission.z * step_opacity * transmittance;
    accumulated_color.w += step_opacity * transmittance;
    

    if (accumulated_color.w > 0.99f) {
        flag = false;
    }
}

// }


// 终止条件：掉入黑洞、飞出边界、或数值异常
if(r<0.55f || r>70.0f ) {flag = false;}
}

float4 color;
if (r >=0.55f && !isnan(r)) {



float3 final_dir = normalize(p);

    float phi = atan2f(final_dir.y, -final_dir.x); 
    float theta = asinf(-final_dir.z);

    float tex_u = phi*0.1591549f+0.5f;
    float tex_v = theta* 0.3183099f+0.5f;
    


    float4 bkgd = tex2D<float4>(tex_obj, tex_u, tex_v);
    color = accumulated_color + bkgd * (1.0f - accumulated_color.w);



// bkgd.x = powf(bkgd.x, 2.2f);
// bkgd.y = powf(bkgd.y, 2.2f);
// bkgd.z = powf(bkgd.z, 2.2f);

// float contrast = 1.5f; 
// bkgd.x = powf(bkgd.x, contrast);
// bkgd.y = powf(bkgd.y, contrast);
// bkgd.z = powf(bkgd.z, contrast);


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
