import gymnasium as gym
import mujoco

# Create the environment
env = gym.make("Humanoid-v4")
model = env.unwrapped.model

# Save the model XML to a file
save_path = "models/humanoid.xml"
mujoco.mj_saveLastXML(save_path, model)

print(f"✅ Saved humanoid.xml to: {save_path}")
env.close()
