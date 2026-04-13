import taichi as ti
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 推荐使用 GPU，如果没有则保持 ti.cpu
ti.init(arch=ti.cuda)

D = 0.01
N = 1200
M = 600
TYP_VEL = 9

# --- 场变量定义 ---
Vel = ti.Vector.field(2, ti.int32, shape=(TYP_VEL))
W = ti.field(ti.f32, shape=(TYP_VEL))
block = ti.field(ti.i8, shape=(N, M))

f = ti.Vector.field(TYP_VEL, dtype=ti.f32, shape=(N, M))
f_eq = ti.Vector.field(TYP_VEL, dtype=ti.f32, shape=(N, M))
rho = ti.field(dtype=ti.f32, shape=(N, M))
u = ti.Vector.field(2, dtype=ti.f32, shape=(N, M))

# 【新增】图像渲染场：存储RGB颜色
img = ti.Vector.field(3, dtype=ti.f32, shape=(N, M))

# 记录受力的全局累加器
lift = ti.field(dtype=ti.f32, shape=())
drag = ti.field(dtype=ti.f32, shape=())

opp = ti.Vector([8, 7, 6, 5, 4, 3, 2, 1, 0])

# 流体参数
spec_vel = 0.1
tau = 0.8
omega = 1.0 / tau
rho_in = 1.0

# --- 初始化方向与权重 ---
@ti.kernel
def init_lbm_constants():
    for i, j in ti.ndrange(3, 3):
        idx = i * 3 + j
        Vel[idx][0] = i - 1
        Vel[idx][1] = j - 1
        if i == 1 and j == 1:
            W[idx] = 4.0 / 9.0
        elif i == 1 or j == 1:
            W[idx] = 1.0 / 9.0
        else:
            W[idx] = 1.0 / 36.0

@ti.func
def gen_eq():
    for i, j in f:
        for k in ti.static(range(TYP_VEL)):
            cu = Vel[k].dot(u[i, j])
            uu = u[i, j].dot(u[i, j])
            f_eq[i, j][k] = W[k] * rho[i, j] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * uu)

@ti.kernel
def init_flow():
    for i, j in f:
        rho[i, j] = 1.0
        u[i, j] = [0.0, 0.0]
    gen_eq()
    for i, j in f:
        f[i, j] = f_eq[i, j]

@ti.kernel
def step() -> bool:
    lift[None] = 0.0
    drag[None] = 0.0

    # 1. 宏观参数
    rho.fill(0)
    u.fill(0)
    for i, j in f:
        if block[i, j] == 0:
            for k in ti.static(range(TYP_VEL)):
                rho[i, j] += f[i, j][k]
                u[i, j] += f[i, j][k] * Vel[k]
            u[i, j] /= rho[i, j]
            u[i, j][0] = ti.math.clamp(u[i, j][0], -0.4, 0.4)
            u[i, j][1] = ti.math.clamp(u[i, j][1], -0.4, 0.4)

    # 2. 碰撞
    gen_eq()
    for i, j in f:
        if block[i, j] == 0:
            f[i, j] += (f_eq[i, j] - f[i, j]) * omega
            f[i, j] = ti.math.clamp(f[i, j], 0, 10)

    # 3. 迁移 & 动量交换受力
    for i, j in f:
        if block[i, j] == 0:
            for k in ti.static(range(TYP_VEL)):
                ip = i - Vel[k][0]
                jp = j - Vel[k][1]
                if 0 <= ip < N and 0 <= jp < M:
                    if block[ip, jp] == 1:
                        f_eq[i, j][k] = f[i, j][opp[k]]
                        momentum_y = 2.0 * f[i, j][opp[k]] * Vel[k][1]
                        momentum_x = 2.0 * f[i, j][opp[k]] * Vel[k][0]
                        lift[None] += momentum_y
                        drag[None] += momentum_x
                    else:
                        f_eq[i, j][k] = f[ip, jp][k]
                else:
                    f_eq[i, j][k] = 0.0

    # 4. 边界条件
    for i in range(N):
        if block[i, 0] == 0:
            for k in ti.static(range(TYP_VEL)):
                if Vel[k][1] > 0:
                    f_eq[i, 0][k] = max(f_eq[i, 1][opp[k]], 1e-12)
        if block[i, M-1] == 0:
            for k in ti.static(range(TYP_VEL)):
                if Vel[k][1] < 0:
                    f_eq[i, M-1][k] = max(f_eq[i, M-2][opp[k]], 1e-12)

    ux_in = spec_vel
    for j in range(M):
        if block[0, j] == 0:
            f_eq[0, j][7] = f_eq[0, j][1] + 2/3 * rho_in * ux_in
            f_eq[0, j][8] = f_eq[0, j][0] + 1/6 * rho_in * ux_in - 0.5 * (f_eq[0, j][5] - f_eq[0, j][3])
            f_eq[0, j][6] = f_eq[0, j][2] + 1/6 * rho_in * ux_in + 0.5 * (f_eq[0, j][5] - f_eq[0, j][3])
            for k in ti.static([6, 7, 8]):
                f_eq[0, j][k] = max(f_eq[0, j][k], 1e-12)
        if block[N-1, j] == 0:
            for k in ti.static(range(TYP_VEL)):
                if Vel[k][0] < 0:
                    f_eq[N-1, j][k] = max(f_eq[N-2, j][k], 1e-12)

    for i, j in f:
        if block[i, j] == 0:
            f[i, j] = f_eq[i, j]

    return True

# 【新增】画面渲染 Kernel
@ti.kernel
def update_img():
    # 遍历除去四周单层边缘的区域，计算涡量
    for i, j in ti.ndrange((1, N-1), (1, M-1)):
        # 涡量公式 (dv/dx - du/dy)
        vort = (u[i+1, j][1] - u[i-1, j][1]) - (u[i, j+1][0] - u[i, j-1][0])
        
        if block[i, j] == 1:
            img[i, j] = [0.0, 0.0, 0.3]  # 障碍物设为深蓝色
        else:
            # 流体颜色映射：涡量越大，R和G通道值越高（发白/发黄）
            img[i, j][0] = ti.math.clamp(20 * ti.abs(vort), 0.0, 1.0)
            img[i, j][1] = ti.math.clamp(20 * ti.abs(vort), 0.0, 1.0)
            img[i, j][2] = ti.math.clamp(rho[i, j] / 2.0, 0.0, 1.0) # B通道基于密度

# --- 图像处理 ---
def get_reference_chord(img_path):
    mat = cv2.imread(img_path)
    if mat is None:
        raise ValueError(f"找不到图片 {img_path}!")
    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray < 128).astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    return w

def load_and_rotate_wing(img_path, angle):
    mat = cv2.imread(img_path)
    h, w = mat.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, -angle, 1.0)
    mat_rot = cv2.warpAffine(mat, M_rot, (w, h), borderValue=(255, 255, 255))
    
    block_np = np.zeros((N, M), dtype=np.int8)
    gray = cv2.cvtColor(mat_rot, cv2.COLOR_BGR2GRAY)
    
    for i in range(M):
        for j in range(N):
            if gray[i, j] < 128:
                block_np[j, M - 1 - i] = 1
                
    block.from_numpy(block_np)

# --- 主程序 ---
def main():
    init_lbm_constants()
    
    img_path = 'wing.bmp'
    L_ref = get_reference_chord(img_path)
    print(f"检测到机翼参考弦长 (L_ref) = {L_ref} 像素")

    angles = [-5, 0, 5, 10, 15, 20,25,30,40,50,60,70,80,90]
    CL_results = []
    CD_results = []

    TOTAL_STEPS = 10000
    AVG_STEPS = 1000

    # 【新增】初始化 GUI 窗口
    gui = ti.GUI('LBM Airfoil Auto-Test', (N, M), fast_gui=True)

    for alpha in angles:
        print(f"\n>>> 开始测试攻角: {alpha}° ...")
        
        load_and_rotate_wing(img_path, alpha)
        init_flow()
        
        lift_sum = 0.0
        drag_sum = 0.0
        
        for step_idx in range(TOTAL_STEPS):
            # 允许用户在测试中途直接关闭窗口退出程序
            if not gui.running:
                print("模拟被用户强制终止。")
                return

            suc = step()
            if not suc:
                print("发散！(Diverged)")
                break
                
            if step_idx >= TOTAL_STEPS - AVG_STEPS:
                lift_sum += lift[None]
                drag_sum += drag[None]
                
            if step_idx > 0 and step_idx % 1000 == 0:
                print(f"    已计算 {step_idx}/{TOTAL_STEPS} 步...")

            # 【新增】每 20 步更新并渲染一次图像
            if step_idx % 20 == 0:
                update_img()             # 计算像素颜色
                gui.set_image(img)       # 将颜色传入 GUI
                gui.show()               # 刷新屏幕

        lift_avg = lift_sum / AVG_STEPS
        drag_avg = drag_sum / AVG_STEPS
        
        q_inf = 0.5 * rho_in * (spec_vel ** 2) * L_ref
        
        CL = lift_avg / q_inf
        CD = drag_avg / q_inf
        
        CL_results.append(CL)
        CD_results.append(CD)
        
        print(f"攻角 {alpha}° 测试完成! CL = {CL:.4f}, CD = {CD:.4f}")

    # 测试结束，关闭窗口
    gui.close()

    # --- 绘制图表 ---
    plt.figure(figsize=(8, 5))
    plt.plot(angles, CL_results, marker='o', linestyle='-', color='b', label='Lift Coefficient ($C_L$)')
    plt.title('Airfoil Lift Coefficient vs Angle of Attack')
    plt.xlabel('Angle of Attack $\\alpha$ (degrees)')
    plt.ylabel('Lift Coefficient $C_L$')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Lift_Polar.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()