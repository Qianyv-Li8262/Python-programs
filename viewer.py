import glfw
from OpenGL.GL import *
# from cuda import cudart
from cupy.cuda import runtime as cudart
import cupy as cp

class ZeroCopyWindow:
    def __init__(self, width, height, title="Ultimate Zero-Copy Tracer"):
        self.w = width
        self.h = height
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0) # 关掉垂直同步，放飞FPS

        # 创建显示纹理
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)

        # 创建 PBO（像素缓冲区：这是CUDA和OpenGL接头的地方）
        self.pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # 将 PBO 注册给 CUDA
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        err, self.cuda_pbo_res = cudart.cudaGraphicsGLRegisterBuffer(self.pbo, flags)

        self.pressed_key = -1
        def char_cb(win, codepoint): self.pressed_key = codepoint
        def key_cb(win, key, scancode, action, mods):
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS: self.pressed_key = 27
        glfw.set_char_callback(self.window, char_cb)
        glfw.set_key_callback(self.window, key_cb)

    def map_pbo_to_cupy(self):
        """核心黑科技：向 OpenGL 借出显存，伪装成 CuPy 数组"""
        cudart.cudaGraphicsMapResources(1, self.cuda_pbo_res, None)
        err, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(self.cuda_pbo_res)
        
        # 用这个纯显存指针，创建一个 uint8 的 CuPy 数组！
        mem = cp.cuda.UnownedMemory(ptr, size, self)
        mptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((self.w * self.h * 3,), dtype=cp.uint8, memptr=mptr)

    def unmap_pbo(self):
        """用完立刻归还"""
        cudart.cudaGraphicsUnmapResources(1, self.cuda_pbo_res, None)

    def draw(self):
        """直接把接头地点的画面刷上屏幕"""
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0,  1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def waitKey(self):
        k = self.pressed_key
        self.pressed_key = -1
        return k

    def should_close(self): return glfw.window_should_close(self.window)