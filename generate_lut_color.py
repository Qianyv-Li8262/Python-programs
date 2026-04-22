import numpy as np

def temperature_to_color(temp):
    t = temp / 100.0
    
    if t <= 66.0:
        r = 1
    else:
        r = 1.292936186 * np.pow(t - 60.0, -0.1332047592)
        r = np.clip(r,0,1)

    if t <= 66.0: 
        g = 0.39008157876 * np.log(t) - 0.631841444
        g = np.clip(g,0,1)
    else:
        g = 1.129890861 * np.pow(t - 60.0, -0.0755148492)
        g = np.clip(g,0,1)
    
    if t >= 66.0:
        b = 1.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 0.543206789 * np.log(t - 10.0) - 1.196254089
        b = np.clip(b,0,1)
    

    return r, g, b
u = np.empty((1,2000,3),dtype=np.float32)
for i in range(2000):
    t = i*10+1010
    r,g,b=temperature_to_color(t)
    u[0,i,:]=(r,g,b)
np.save('color_lut.npy', u)
print("LUT 生成完成，形状:", u.shape)