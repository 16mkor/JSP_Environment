from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, A2C
from GridWorld import GridWorldEnv

env = DummyVecEnv([lambda: GridWorldEnv()])
# env = GridWorldEnv()

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)

obs = env.reset()

for i in range(2_000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
