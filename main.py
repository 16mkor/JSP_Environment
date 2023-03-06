from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from JSP_env.envs.production_env import ProductionEnv

# env = DummyVecEnv([lambda: ProductionEnv()])
env = ProductionEnv()
check_env(env)  # passed :)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)

obs = env.reset()

for i in range(2_000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
