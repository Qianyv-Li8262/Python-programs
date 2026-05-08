#define EPS 0.0025
#define BLOCKSIZE 256
#include <cuda_pipeline.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
// auto test_pipe = cuda::make_pipeline<cuda::thread_scope_block>();
__device__ __forceinline__ float getdistance(float xme,float yme,float xit,float yit){
    float xx = xme - xit;
    float yy = yme - yit;
    return sqrtf(xx*xx+yy*yy);
}
__device__ __forceinline__ float3 getaccel(float m2,float xme,float yme,float xit,float yit){
    float xx = xme - xit;
    float yy = yme - yit;
    float invdis=rsqrtf(xx*xx+yy*yy+EPS*EPS);
    // float diss_soft = dis*dis + EPS * EPS;
    float F = m2 *invdis*invdis*invdis;
    return make_float3(-F*xx,-F*yy,0.0f);
}
extern "C"
__global__ void nbodystep(
    float* __restrict__ posx,
    float* __restrict__ posy,
    float* __restrict__ velx,
    float* __restrict__ vely,
    const float* __restrict__ mass,
    const int n,const float dt
)
{




        namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();

    // 2. 在共享内存中声明 Pipeline 的状态对象
    // cuda::thread_scope_block 表示作用域是整个 Block
    // 2 表示我们要用双缓冲 (Double Buffering，2个 stage)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;

    // 3. 正确初始化 Pipeline！把 block 和 状态指针 传进去
    auto pipe = cuda::make_pipeline(block, &shared_state);

int tid = blockIdx.x*blockDim.x+threadIdx.x;
if(tid>=n) return;
float x=posx[tid];
float y=posy[tid];
float xdot=velx[tid];
float ydot=vely[tid];
float xddot = 0.0f;
float yddot = 0.0f;
// float m=mass[tid];

__shared__ float4 pos_and_vel[BLOCKSIZE];
__shared__ float masss[BLOCKSIZE];
// #pragma unroll
for(int tile=0;tile<gridDim.x;++tile)
{
int idx = tile * blockDim.x + threadIdx.x;
if (idx<n)
    {
    float xx=posx[idx];
    float yy=posy[idx];
    // float xxdot=velx[idx];
    // float yydot=vely[idx];
    pos_and_vel[threadIdx.x]=make_float4(xx,yy,0.0f,0.0f);
    masss[threadIdx.x] = mass[idx];
    } else 
    {
    pos_and_vel[threadIdx.x]=make_float4(0.0f,0.0f,0.0f,0.0f);
    masss[threadIdx.x] = 0.0f;
    }
    __syncthreads();
#pragma unroll 16
for(int i=0;i<blockDim.x;++i)
    {
    int j = tile*blockDim.x + i;
    if(tid==j) continue;
    if(j>=n) continue;
    float4 temp = pos_and_vel[i];
    float mm = masss[i];
    float3 accel = getaccel(mm,x,y,temp.x,temp.y);
    xddot+=accel.x;
    yddot+=accel.y;
    }
__syncthreads();
}
xdot += xddot * dt;
ydot += yddot * dt;
x += xdot * dt;
y += ydot * dt;
posx[tid]=x;
posy[tid]=y;
velx[tid]=xdot;
vely[tid]=ydot;

}


extern "C"
__global__ void render_bodies(
    unsigned char* __restrict__ image,      // 输出图像 RGBA
    const float* __restrict__ posx,  // 天体物理 x 坐标
    const float* __restrict__ posy,  // 天体物理 y 坐标
    int n,                           // 天体数量
    int width,                       // 图像宽度 (像素)
    int height,                      // 图像高度 (像素)
    float xmin,                      // 物理世界 x 范围
    float xmax,
    float ymin,
    float ymax,
    float radius)                    // 绘制半径 (像素)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // 像素中心坐标 (0.5 偏移保证抗锯齿视觉)
    float px = x + 0.5f;
    float py = y + 0.5f;

    // 映射到物理坐标：
    // 物理 x 从左到右线性映射
    float fx = xmin + (px / (float)width)  * (xmax - xmin);
    // 物理 y：图像 y 向下，物理 y 轴向上，所以需要翻转：
    // 图像顶部 (py=0) 对应 ymax，底部对应 ymin
    float fy = ymax - (py / (float)height) * (ymax - ymin);

    float r2 = radius * radius;
    uchar4 color = make_uchar4(0, 0, 0, 255); // 黑色背景

    for (int i = 0; i < n; ++i) {
        float dx = fx - posx[i];
        float dy = fy - posy[i];
        if (dx * dx + dy * dy <= r2) {
            color = make_uchar4(255, 255, 255, 255); // 白色
            break;  // 找到一个天体就跳出，可以加速
        }
    }

    image[(y * width + x)*4+0] = color.x;
    image[(y * width + x)*4+1] = color.y;
    image[(y * width + x)*4+2] = color.z;
    image[(y * width + x)*4+3] = color.w;
}




// --- 1. 清空背景的 Kernel ---
extern "C"
__global__ void clear_background(unsigned char* __restrict__ image, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;
    image[idx + 0] = 0;   // R
    image[idx + 1] = 0;   // G
    image[idx + 2] = 0;   // B
    image[idx + 3] = 255; // A
}

// --- 2. 极速渲染星星的 Kernel ---
extern "C"
__global__ void render_bodies_fast(
    unsigned char* __restrict__ image,
    const float* __restrict__ posx,
    const float* __restrict__ posy,
    int n,
    int width,
    int height,
    float xmin,
    float xmax,
    float ymin,
    float ymax)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float x = posx[tid];
    float y = posy[tid];

    // 如果星星飞出了视口，就不画它
    if (x < xmin || x > xmax || y < ymin || y > ymax) return;

    // 物理坐标映射到像素坐标 (之前逻辑的逆向)
    int px = (int)((x - xmin) / (xmax - xmin) * width);
    int py = (int)((ymax - y) / (ymax - ymin) * height);

    // 确保像素不越界
    if (px >= 0 && px < width && py >= 0 && py < height) {
        int idx = (py * width + px) * 4;
        image[idx + 0] = 255; // R
        image[idx + 1] = 255; // G
        image[idx + 2] = 255; // B
        // Alpha 通道在 clear_background 已经是 255 了，这里不用动
    }
}