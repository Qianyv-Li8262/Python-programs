__device__ const int biasx[9] = {0,1,0,-1,0,1,-1,-1,1};
__device__ const int biasy[9] = {0,0,1,0,-1,1,1,-1,-1};
__device__ const int opp[9] = {0,3,4,1,2,7,8,5,6}
__device__ const float w[9] = {0.444444444f,0.1111111111f,0.1111111111f,0.1111111111f,0.1111111111f,0.0277777778f,0.0277777778f,0.0277777778f,0.0277777778f}

extern "C" 
__global__ void lbmkernel(

bool* __restrict__ mask,
float* __restrict__ f_now,
float* __restrict__ f_out,
//float* __restrict__ rho,
//float* __restrict__ ux,
//float* __restrict__ uy,
const int totwidth,const int totheight,const float tau
){
int totpixels = totwidth * totheight;
int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
int pixel_idy = blockIdx.y * blockDim.y + threadIdx.y;
int pid = pixel_idx + pixel_idy * totwidth;
if( pixel_idx >= totwidth || pixel_idy >= totheight ) return;
if(mask[pid]) return;
float rho_loc = 0.0f;
float ux_loc = 0.0f;
float uy_loc = 0.0f;
float f[9];
float f_eqn[9];

//use ghost cells which means you should cover the region with mask = 1

#pragma unroll
for (int i=0;i<9;++i){
int pullfromx = pixel_idx - biasx[i];
int pullfromy = pixel_idy - biasy[i];
int sid = pullfromy * totwidth + pullfromx;
f[i] = mask[sid] ? f_now[opp[i] * totpixels + pid] : f_now[i * totpixels + sid];
rho_loc += f[i];
ux_loc += f[i]*biasx[i];
uy_loc += f[i]*biasy[i];
}

ux_loc/=rho_loc;
uy_loc/=rho_loc;
#pragma unroll
for(int i=0;i<9;++i){
float e_dot_u = biasx[i] * ux_loc + biasy[i] * uy_loc;
float usq = ux_loc * ux_loc + uy_loc * uy_loc;
f_eqn[i] = w[i] * rho_loc * (1 + 3.0f * e_dot_u + 4.5f * e_dot_u * e_dot_u - 1.5f * usq)
f_out[i * totpixels + pid] = f_eqn[i]
}
}