import gym
from custom_robocasa.utils.point_cloud.pc_generator import PointCloudGenerator

class PointCloudWrapper(gym.Wrapper):
    def __init__(self, env, global_frame: bool, get_segmented_pc: bool = False, get_normal_pc: bool = True):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(env.sim, self.cam_names, env.camera_widths[0], env.camera_heights[0], global_frame)
        
        self.get_segmented_pc = get_segmented_pc
        self.get_normal_pc = get_normal_pc

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict
    
    def get_point_cloud(self, obs_dict):
        assert self.get_normal_pc, "get_normal_pc is set to False"

        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}
        
        return self.pc_generator.get_point_cloud(imgs, depths)
    
    def get_segmented_point_cloud(self, obs_dict):
        assert self.get_segmented_pc, "get_segmented_pc is set to False"

        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}
        segmentations = {cam: obs_dict[f"{cam}_segmentation_mask"] for cam in self.cam_names}
        
        return self.pc_generator.get_segmented_point_cloud(imgs, depths, segmentations)
    
    def _check_success(self):
        return self.env._check_success()

