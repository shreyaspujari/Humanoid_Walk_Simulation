import mujoco
import glfw
import time

def load_and_render(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    window = glfw.create_window(800, 600, model_path, None, None)
    glfw.make_context_current(window)

    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    scene = mujoco.MjvScene(model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = model.stat.center
    cam.distance = 5.0
    cam.azimuth = 45.0
    cam.elevation = -15.0

    for _ in range(300):
        mujoco.mj_step(model, data)
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)

    glfw.terminate()

# 👇 Preview each terrain
for terrain in ["humanoid_flat.xml", "humanoid_ice.xml", "humanoid_sand.xml", "humanoid_hill.xml"]:
    print(f"Previewing {terrain}")
    load_and_render(f"models/{terrain}")
