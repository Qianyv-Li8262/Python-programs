__device__ __forceinline__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


extern "C"
__global__ void isingstep(
volatile uint32_t* spin_field,
float* totEnergy,
bool red_or_black,
int x_words_num,
int y_lines
){

int x_wordid = blockDim.x * blockIdx.x + threadIdx.x;
int y_lineid = blockDim.y * blockIdx.y + threadIdx.y;
int s = y_lineid * x_words_num + x_wordid+1;//加一个外部传入随机数
uint32_t state = (uint32_t)s + 1;
if (x_wordid>=x_words_num || y_lineid >= y_lines) return;
uint32_t valid_mask = (red_or_black ^ (y_lineid%2==0)) ? 0x55555555U : 0xAAAAAAAAU;
uint32_t self_word = spin_field[y_lineid * x_words_num + x_wordid];
uint32_t up_neighbor = y_lineid == 0 ? spin_field[x_words_num*(y_lines-1)+x_wordid] : spin_field[x_words_num*(y_lineid-1)+x_wordid];
uint32_t down_neighbor = y_lineid == y_lines - 1 ? spin_field[x_wordid] : spin_field[x_words_num*(y_lineid+1)+x_wordid];
uint32_t left_neighbor = x_wordid == 0 ? spin_field[(y_lineid+1) * x_words_num - 1] : spin_field[x_words_num*y_lineid+x_wordid-1];
uint32_t right_neighbor = x_wordid == x_words_num-1 ? spin_field[(y_lineid) * x_words_num] : spin_field[x_words_num*y_lineid+x_wordid+1];
left_neighbor = (self_word >> 1) | (left_neighbor << 31);
right_neighbor = (self_word << 1) | (right_neighbor >> 31);
up_neighbor = ~(self_word ^ up_neighbor);
down_neighbor = ~(self_word ^ down_neighbor);
left_neighbor = ~(self_word ^ left_neighbor);
right_neighbor = ~(self_word ^ right_neighbor);
uint32_t energy = __popc(up_neighbor)+__popc(down_neighbor)+__popc(left_neighbor)+__popc(right_neighbor);
// atomicAdd(totEnergy,(float)energy);
uint32_t and1 = up_neighbor & down_neighbor;
uint32_t and2 = left_neighbor & right_neighbor;
uint32_t xor1 = up_neighbor ^ down_neighbor;
uint32_t xor2 = left_neighbor ^ right_neighbor;
uint32_t mask_4 = and1 & and2;
uint32_t mask_3 = (and1 & xor2) | (and2 & xor1);
uint32_t mask_le_2 = ~(mask_4 | mask_3);
uint32_t randnum_4 = xorshift32(&state) & xorshift32(&state);
uint32_t randnum_8 = randnum_4 & xorshift32(&state);
uint32_t flip_or_not = (mask_4 & randnum_8) | (mask_3 & randnum_4) | mask_le_2;
flip_or_not = flip_or_not & valid_mask;
self_word = flip_or_not ^ self_word;
spin_field[y_lineid * x_words_num + x_wordid] = self_word;
}