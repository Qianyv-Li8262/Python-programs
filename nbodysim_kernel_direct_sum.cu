#define EPS 0.05
#define BLOCKSIZE 256
__device__ __forceinline__ float getdistance(float xme,float yme,float xit,float yit){
    float xx = xme - xit;
    float yy = yme - yit;
    return sqrtf(xx*xx+yy*yy);
}
__device__ __forceinline__ float3 getaccel(float m2,float xme,float yme,float xit,float yit){
    float xx = xme - xit;
    float yy = yme - yit;
    float dis=sqrtf(xx*xx+yy*yy);
    float diss_soft = dis*dis + EPS * EPS;
    float F = m2 / diss_soft * rsqrtf(diss_soft);
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
for(int tile=0;tile<gridDim.x;++tile)
{
int idx = tile * blockDim.x + threadIdx.x;
if (idx<n)
    {
    float xx=posx[idx];
    float yy=posy[idx];
    float xxdot=velx[idx];
    float yydot=vely[idx];
    pos_and_vel[threadIdx.x]=make_float4(xx,yy,xxdot,yydot);
    masss[threadIdx.x] = mass[idx];
    } else 
    {
    pos_and_vel[threadIdx.x]=make_float4(0.0f,0.0f,0.0f,0.0f);
    masss[threadIdx.x] = 0.0f;
    }
    __syncthreads();

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