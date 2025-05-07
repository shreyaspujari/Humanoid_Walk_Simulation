# File: train_multi_terrain.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from environments.multi_terrain_env import MultiTerrainEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ✅ Initialize multi-terrain environment
env = MultiTerrainEnv(model_dir="models", render_mode=None)

# ✅ Set up PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./ppo_logs_multi"
)

# ✅ Optional: checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints_multi",
    name_prefix="ppo_multi_terrain"
)

# ✅ Start training
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback
)

# ✅ Save final model
model.save("ppo_multi_terrain_final")

# ✅ Clean up
env.close()
