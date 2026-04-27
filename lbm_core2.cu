__device__ const int biasx[9] = {0,1,0,-1,0,1,-1,-1,1};
__device__ const int biasy[9] = {0,0,1,0,-1,1,1,-1,-1};
__device__ const int opp[9] = {0,3,4,1,2,7,8,5,6};
__device__ const float w[9] = {0.444444444f,0.1111111111f,0.1111111111f,0.1111111111f,0.1111111111f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f};

extern "C" 
__global__ void fused_lbmkernel(
    bool* __restrict__ mask,
    float* __restrict__ f_now,
    float* __restrict__ f_out,
    float* __restrict__ ux,
    float* __restrict__ uy,
    const int totwidth, 
    const int totheight,
    const float tau_inv,      // 内部区域的 tau_inv 
    const float u_in,         // 入口速度 
    const float tau_inv_bnd   // 入口边界的 tau_inv
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
    

    if (pixel_idy <= 0 || pixel_idy >= totheight - 1) return;

    if (pixel_idx >= totwidth - 1) return; 

    int pid = pixel_idy * totwidth + pixel_idx;
    int totpixels = totwidth * totheight;

    if (pixel_idx == 0) {
        float f[9];
        #pragma unroll
        for (int i=0; i<9; ++i){
            int pullfromx = 0 - biasx[i];
            int pullfromy = pixel_idy - biasy[i];
            
            if (pullfromx < 0) {
                f[i] = 0.0f; 
            } else {
                int sid = pullfromy * totwidth + pullfromx;
                f[i] = mask[sid] ? f_now[opp[i] * totpixels + pid] : f_now[i * totpixels + sid];
            }
        }

        // 平滑过渡
        float dist_to_wall = fminf((float)pixel_idy, (float)(totheight - 1 - pixel_idy));
        float smooth_factor = 1.0f;
        if (dist_to_wall < 50.0f) {
            smooth_factor = 0.5f * (1.0f - cosf(3.14159f * dist_to_wall / 50.0f));
        }
        float local_u = u_in * smooth_factor;

        // Zou-He 计算
        float rho = (f[0] + f[2] + f[4] + 2.0f * (f[3] + f[6] + f[7])) / (1.0f - local_u);
        f[1] = f[3] + (2.0f / 3.0f) * rho * local_u;
        f[5] = f[7] - 0.5f * (f[2] - f[4]) + (1.0f / 6.0f) * rho * local_u;
        f[8] = f[6] + 0.5f * (f[2] - f[4]) + (1.0f / 6.0f) * rho * local_u;
        
        ux[pid] = local_u;
        uy[pid] = 0.0f;

        // 碰撞步
        #pragma unroll
        for(int i=0; i<9; ++i){
            float e_dot_u = biasx[i] * local_u; 
            float usq = local_u * local_u;
            float feq = w[i] * rho * (1.0f + 3.0f * e_dot_u + 4.5f * e_dot_u * e_dot_u - 1.5f * usq);
            float post_collision = f[i] - tau_inv_bnd * (f[i] - feq);
            f_out[i * totpixels + pid] = fmaxf(0.0f, post_collision);
        }
    } 

    else {
        if (mask[pid]) return;

        float rho_loc = 0.0f;
        float ux_loc = 0.0f;
        float uy_loc = 0.0f;
        float f[9];

        #pragma unroll
        for (int i=0; i<9; ++i){
            int pullfromx = pixel_idx - biasx[i];
            int pullfromy = pixel_idy - biasy[i];
            int sid = pullfromy * totwidth + pullfromx;
            f[i] = mask[sid] ? f_now[opp[i] * totpixels + pid] : f_now[i * totpixels + sid];
            rho_loc += f[i];
            ux_loc += f[i]*biasx[i];
            uy_loc += f[i]*biasy[i];
        }

        ux_loc /= rho_loc;
        uy_loc /= rho_loc;
        ux_loc = fminf(0.4f, fmaxf(-0.4f, ux_loc));
        uy_loc = fminf(0.4f, fmaxf(-0.4f, uy_loc));
        ux[pid] = ux_loc;
        uy[pid] = uy_loc;

        #pragma unroll
        for(int i=0; i<9; ++i){
            float e_dot_u = biasx[i] * ux_loc + biasy[i] * uy_loc;
            float usq = ux_loc * ux_loc + uy_loc * uy_loc;
            

            float feq = w[i] * rho_loc * (1.0f + 3.0f * e_dot_u + 4.5f * e_dot_u * e_dot_u - 1.5f * usq);
            float ff = f[i] - tau_inv * (f[i] - feq);
            float out_val = fminf(10.0f, fmaxf(0.0f, ff));
            
            f_out[i * totpixels + pid] = out_val;

            // 【精华】：顺手处理 right_out 逻辑，零开销！
            // 如果计算的是倒数第二列，将相同的值写一份给最后一列
            if (pixel_idx == totwidth - 2) {
                f_out[i * totpixels + pid + 1] = out_val;
            }
        }
    }
}




extern "C"
__global__ void visualizekernel(
const float* __restrict__ ux,
const float* __restrict__ uy,
unsigned char* __restrict__ image,
const bool* __restrict__ mask,
const int totwidth, const int totheight,
const float vort_scale
){

// int totpixels = totwidth * totheight;
int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
int pid = pixel_idx + pixel_idy * totwidth;

if( pixel_idx <= 0 || pixel_idx >= totwidth-1  || pixel_idy <= 0 || pixel_idy >= totheight-1) return;
image[pid*4+3]=255;
if(mask[pid]) {
    image[pid*4]=0;
    image[pid*4+1]=0;
    image[pid*4+2]=0;
    return;
}
int pid_right = pixel_idy * totwidth + (pixel_idx + 1);
int pid_left  = pixel_idy * totwidth + (pixel_idx - 1);
int pid_top   = (pixel_idy + 1) * totwidth + pixel_idx;
int pid_bot   = (pixel_idy - 1) * totwidth + pixel_idx;

    float vort = ((uy[pid_right] - uy[pid_left]) - (ux[pid_top] - ux[pid_bot]))*vort_scale;
// 如果你喜欢你同学的赛博朋克风格，可以用下面这三行替换上面三行：
    // float r = fminf(fmaxf(fabsf(vort), 0.0f), 1.0f);
    // float g = r;
    // float b = 0.5f;
// float speed = sqrtf(ux[pid]*ux[pid] + uy[pid]*uy[pid]);
// float r = fminf(speed * 5.0f, 1.0f); // 放大 5 倍观察
// float g = r;
// float b = 0.5f; // 这就是你看到的蓝色背景来源


    float r = fminf(fmaxf(1.0f + vort, 0.0f), 1.0f);
    float g = fminf(fmaxf(1.0f - fabsf(vort), 0.0f), 1.0f);
    float b = fminf(fmaxf(1.0f - vort, 0.0f), 1.0f);
    image[pid*4 + 0] = (unsigned char)(r * 255);
    image[pid*4 + 1] = (unsigned char)(g * 255);
    image[pid*4 + 2] = (unsigned char)(b * 255);
}