from stable_baselines3 import PPO
from environments.multi_terrain_env import MultiTerrainEnv
import time

# ✅ Load the trained flat walking model
model = PPO.load("ppo_humanoid_final")

# ✅ Force test on flat terrain with updated humanoid.xml
env = MultiTerrainEnv(model_dir="models")
env.terrain_options = ["humanoid_flat.xml"]

obs, _ = env.reset()

# ✅ Simulate for a few seconds to test new dynamics
for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.01)
    if done or truncated:
        obs, _ = env.reset()

env.close()
