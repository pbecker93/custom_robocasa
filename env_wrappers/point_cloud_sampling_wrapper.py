import gym

from custom_robocasa.utils.point_cloud.sampling.base_pc_sampler import BasePointCloudSampler
from custom_robocasa.utils.point_cloud.sampling.uniform_pc_sampler import UniformPointCloudSampler


class PointCloudSamplingWrapper(gym.Wrapper):
    def __init__(self, env, pc_sampler: BasePointCloudSampler, num_points: int):
        super().__init__(env)
        self.pc_sampler = pc_sampler
        self.num_points = num_points

        self.key = "uniform_sampled_point_cloud" if isinstance(self.pc_sampler, UniformPointCloudSampler) else "sampled_point_cloud"
        
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict
    
    def _check_success(self):
        return self.env._check_success()