# File: environments/multi_terrain_env.py

import gymnasium as gym
import numpy as np
import mujoco
import os
import random
import glfw

class MultiTerrainEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_dir="models", render_mode=None):
        super().__init__()
        self.model_dir = model_dir
        self.render_mode = render_mode
        self.terrain_options = [
            "humanoid_flat.xml",
            "humanoid_ice.xml",
            "humanoid_sand.xml",
            "humanoid_hill.xml"
        ]
        self.model = None
        self.data = None
        self._load_new_model()

        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def _load_new_model(self):
        terrain = random.choice(self.terrain_options)
        model_path = os.path.join(self.model_dir, terrain)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

    def reset(self, seed=None, options=None):
        self._load_new_model()
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_termination()
        truncated = False

        return obs, reward, done, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def _compute_reward(self):
        forward_velocity = self.data.qvel[0]
        upright = 1.0 if self.data.qpos[2] > 0.8 else 0.0
        alive_bonus = 0.2
        control_penalty = 0.005 * np.square(self.data.ctrl).sum()
        return forward_velocity + upright + alive_bonus - control_penalty

    def _check_termination(self):
        return self.data.qpos[2] < 0.5

    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, "window"):
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")
            self.window = glfw.create_window(800, 600, "MuJoCo Viewer", None, None)
            glfw.make_context_current(self.window)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = self.model.stat.center
            self.cam.distance = 4.0
            self.cam.azimuth = 90.0
            self.cam.elevation = -20.0

        mujoco.mjv_updateScene(
            self.model, self.data,
            mujoco.MjvOption(), None,
            self.cam, mujoco.mjtCatBit.mjCAT_ALL,
            self.scene
        )

        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        pass
