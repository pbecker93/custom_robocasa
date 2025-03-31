import gym
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix

class CameraInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        obs_dict.update(self.get_camera_info(obs_dict))

        return obs_dict

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        obs_dict.update(self.get_camera_info(obs_dict))

        return obs_dict

    def get_camera_info(self, obs_dict):
        heights = {cam: obs_dict[f"{cam}_image"].shape[0] for cam in self.cam_names}
        widths = {cam: obs_dict[f"{cam}_image"].shape[1] for cam in self.cam_names}

        intrinsic_matrices = {f"{cam}_intrinsics": get_camera_intrinsic_matrix(self.env.sim, cam, camera_height=heights[cam], camera_width=widths[cam]) for cam in self.cam_names}
        extrinsic_matrices = {f"{cam}_extrinsics": get_camera_extrinsic_matrix(self.env.sim, cam) for cam in self.cam_names}

        return intrinsic_matrices | extrinsic_matrices

    def _check_success(self):
        return self.env._check_success()

