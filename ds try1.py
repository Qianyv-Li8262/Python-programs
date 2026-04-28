import cupy as cp
import numpy as np
import os

# ================= 加载 CUDA 内核 =================
base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "nbodysim_kernel_direct_sum.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
kernel = module.get_function('nbodystep')

# ================= 模拟参数 =================
n = 2                   # 二体
dt = cp.float32(0.001)  # 时间步长
G = 1.0                 # 引力常数（已在 getaccel 中隐含为 1）

# 初始条件：等质量 1，初始距离 1，绕质心圆轨道
mass = cp.array([1.0, 1.0], dtype=cp.float32)
posx = cp.array([0.0, 1.0], dtype=cp.float32)
posy = cp.array([0.0, 0.0], dtype=cp.float32)

# 圆轨道切向速度 v = sqrt(G*(m1+m2)/r) * (距离质心的比例)
v_rel = np.sqrt(G * (1.0 + 1.0) / 1.0)   # sqrt(2) ≈ 1.4142
v1 = v_rel * 1.0 / 2.0                   # 0.7071
velx = cp.array([0.0, 0.0], dtype=cp.float32)
vely = cp.array([v1, -v1], dtype=cp.float32)

# CUDA 执行配置
block = (256,)   # 1 个 block 包含 256 个线程（n 很小也没关系）
grid = (1,)      # 1 个 grid

# ================= 时间步进 =================
steps = 1000
print_freq = 100

print("Two‑body circular orbit simulation (G=1, m1=m2=1, dt=0.001)")
print("=" * 50)
for step in range(steps + 1):
    if step % print_freq == 0:
        p = cp.asnumpy(posx), cp.asnumpy(posy)
        v = cp.asnumpy(velx), cp.asnumpy(vely)
        print(f"Step {step:4d}:")
        print(f"  Pos body0: ({p[0][0]:.6f}, {p[1][0]:.6f})   Pos body1: ({p[0][1]:.6f}, {p[1][1]:.6f})")
        print(f"  Vel body0: ({v[0][0]:.6f}, {v[1][0]:.6f})   Vel body1: ({v[0][1]:.6f}, {v[1][1]:.6f})")

    if step < steps:
        kernel(grid, block, (posx, posy, velx, vely, mass, n, dt))
        cp.cuda.Device().synchronize()

print("=" * 50)
print("Simulation finished.")