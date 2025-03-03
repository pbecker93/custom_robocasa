from custom_robocasa.utils.point_cloud.sampling.base_pc_sampler import (
    BasePointCloudSampler,
)
import gym
import numpy as np

from custom_robocasa.utils.point_cloud.sampling.uniform_pc_sampler import UniformPointCloudSampler

ENV_OBJ_NAMES = {
    "PnPCounterToCab": "obj",
    "PnPCabToCounter": "obj",
    "PnPCounterToSink": "obj",
    "PnPSinkToCounter": "obj",
    "PnPCounterToMicrowave": "obj",
    "PnPMicrowaveToCounter": "obj",
    "PnPCounterToStove": "obj",
    "PnPStoveToCounter": "obj",
    "TurnOnStove": "Stove",
    "TurnOffStove": "Stove",
    "CoffeeSetupMug": "obj",
    "CoffeeServeMug": "obj",
    "CoffeePressButton": "CoffeeMachine",
    "TurnOnMicrowave": "Microwave",
    "TurnOffMicrowave": "Microwave",
    "TurnOnSinkFaucet": "Sink",
    "TurnOffSinkFaucet": "Sink",
    "TurnSinkSpout": "Sink",
}


class SegmentedPointCloudSamplingWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        env_name: str,
        obj_sampler: BasePointCloudSampler,
        rest_sampler: BasePointCloudSampler,
        num_points: int,
        obj_max_num_points: int,
    ):
        super().__init__(env)

        assert env_name in ENV_OBJ_NAMES, f"env_name {env_name} not supported"

        self.obj_name = ENV_OBJ_NAMES[env_name]
        self.obj_sampler = obj_sampler
        self.rest_sampler = rest_sampler
        self.num_points = num_points
        self.obj_max_num_points = obj_max_num_points

        self.in_key = "segmented_point_cloud"
        self.key = (
            "segmented_uniform_sampled_point_cloud"
            if isinstance(self.obj_sampler, UniformPointCloudSampler)
            else "segmented_sampled_point_cloud"
        )

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        sampled_pc = self.sample_point_cloud(obs_dict[self.in_key])
        obs_dict[self.key] = sampled_pc

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        sampled_pc = self.sample_point_cloud(obs_dict[self.in_key])
        obs_dict[self.key] = sampled_pc

        return obs_dict

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        sampled_pc = self.sample_point_cloud(obs_dict[self.in_key])
        obs_dict[self.key] = sampled_pc

        return obs_dict

    def sample_point_cloud(
        self, segmented_point_cloud: dict[str, np.ndarray]
    ) -> np.ndarray:
        try:
            obj_point_cloud = segmented_point_cloud[self.obj_name]
        except KeyError:
            obj_point_cloud = segmented_point_cloud["Stovetop"]

        sampled_obj_pc = self.obj_sampler.sample(
            obj_point_cloud, self.obj_max_num_points
        )

        rest_point_cloud = np.concatenate(
            [
                pc
                for class_name, pc in segmented_point_cloud.items()
                if class_name != self.obj_name
            ]
        )

        sampled_rest_pc = self.rest_sampler.sample(
            rest_point_cloud, self.num_points - sampled_obj_pc.shape[0]
        )

        return np.concatenate([sampled_obj_pc, sampled_rest_pc])

    def _check_success(self):
        return self.env._check_success()
