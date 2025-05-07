from stable_baselines3 import PPO
from environments.multi_terrain_env import MultiTerrainEnv
import time

# ✅ Load your trained model
model = PPO.load("ppo_multi_terrain_final")

# ✅ Choose the terrain to test — just change the filename here
terrain_to_test = "humanoid_flat.xml"  # Options: flat, sand, hill, ice

# ✅ Override env to test one terrain only
class SingleTerrainEnv(MultiTerrainEnv):
    def __init__(self, model_dir="models"):
        super().__init__(model_dir=model_dir, render_mode="human")
        self.terrain_options = [terrain_to_test]

env = SingleTerrainEnv()

obs, _ = env.reset()

for _ in range(2000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.01)
    if done or truncated:
        obs, _ = env.reset()

env.close()
