import cupy as cp
import numpy as np
import os
import zero_copy_window,time

# ================= 加载 CUDA 内核 =================
base_path = os.path.dirname(os.path.abspath(__file__))
kernel_path = os.path.join(base_path, "nbodysim_kernel_direct_sum.cu")
with open(kernel_path, "r", encoding="utf-8") as f:
    cuda_source = f.read()

module = cp.RawModule(code=cuda_source, options=('-use_fast_math',))
kernel = module.get_function('nbodystep')
visualize=module.get_function('render_bodies')

# ================= 模拟参数 =================
n = 3                   # 二体
dt = cp.float32(0.001)  # 时间步长
G = 1.0                 # 引力常数（已在 getaccel 中隐含为 1）

# 初始条件：等质量 1，初始距离 1，绕质心圆轨道
mass = cp.array([1.0, 1.0,0.1], dtype=cp.float32)
posx = cp.array([0.0, 1.0,3.0], dtype=cp.float32)
posy = cp.array([0.0, 0.0,0.1], dtype=cp.float32)

# 圆轨道切向速度 v = sqrt(G*(m1+m2)/r) * (距离质心的比例)
v_rel = np.sqrt(G * (1.0 + 1.0) / 1.0)   # sqrt(2) ≈ 1.4142
v1 = v_rel * 1.0 / 2.0                   # 0.7071
velx = cp.array([0.0, 0.0,0.0], dtype=cp.float32)
vely = cp.array([v1, -v1,0.0], dtype=cp.float32)

# CUDA 执行配置
block = (256,)   # 1 个 block 包含 256 个线程（n 很小也没关系）
grid = (1,)      # 1 个 grid

# ================= 时间步进 =================
steps = 10000
print_freq = 100
window=zero_copy_window.ZeroCopyWindow(1000,1000,'try')
print("Two‑body circular orbit simulation (G=1, m1=m2=1, dt=0.001)")
print("=" * 50)

for step in range(steps + 1):

    if step < steps:
        kernel(grid, block, (posx, posy, velx, vely, mass, n, dt))
        cp.cuda.Device().synchronize()
    

    if step%print_freq==0:
        canvas=window.map_pbo()
        visualize((63,63),(16,16),(canvas,posx,posy,n,cp.int32(1000),cp.int32(1000),cp.float32(-0.5),cp.float32(3.5),cp.float32(-2),cp.float32(2),cp.float32(0.1)))
        window.unmap_and_draw()
        time.sleep(0.5)
    if window.should_close():
        window.destroy()
        break

print("=" * 50)
print("Simulation finished.")