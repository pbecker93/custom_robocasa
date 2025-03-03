import gym
import numpy as np
from robosuite.utils.camera_utils import get_camera_segmentation

ENV_CONFIG = {
    "PnPCounterToCab": {
        "classes": ["Counter", "SingleCabinet", "HingeCabinet"],
        "include_obj": True
    },
    "PnPCabToCounter": {
        "classes": ["Counter", "SingleCabinet", "HingeCabinet"],
        "include_obj": True
    },
    "PnPCounterToSink": {
        "classes": ["Counter", "Sink"],
        "include_obj": True
    },
    "PnPSinkToCounter": {
        "classes": ["Counter", "Sink"],
        "include_obj": True
    },
    "PnPCounterToMicrowave": {
        "classes": ["Counter", "Microwave"],
        "include_obj": True
    },
    "PnPMicrowaveToCounter": {
        "classes": ["Counter", "Microwave"],
        "include_obj": True
    },
    "PnPCounterToStove": {
        "classes": ["Counter", "Stove", "Stovetop"],
        "include_obj": True
    },
    "PnPStoveToCounter": {
        "classes": ["Counter", "Stove", "Stovetop"],
        "include_obj": True
    },
    "TurnOnStove": {
        "classes": ["Stove", "Stovetop"],
        "include_obj": False
    },
    "TurnOffStove": {
        "classes": ["Stove", "Stovetop"],
        "include_obj": False
    },
    "CoffeeSetupMug": {
        "classes": ["CoffeeMachine", "Counter"],
        "include_obj": True
    },
    "CoffeeServeMug": {
        "classes": ["CoffeeMachine", "Counter"],
        "include_obj": True
    },
    "CoffeePressButton": {
        "classes": ["CoffeeMachine"],
        "include_obj": False
    },
    "TurnOnMicrowave": {
        "classes": ["Microwave"],
        "include_obj": False
    },
    "TurnOffMicrowave": {
        "classes": ["Microwave"],
        "include_obj": False
    },
    "TurnOnSinkFaucet": {
        "classes": ["Sink"],
        "include_obj": False
    },
    "TurnOffSinkFaucet": {
        "classes": ["Sink"],
        "include_obj": False
    },
    "TurnSinkSpout": {
        "classes": ["Sink"],
        "include_obj": False
    },
}


class SegmentationWrapper(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.env = env

        assert env_name in ENV_CONFIG, f"Environment {env_name} not supported"

        self.classes = ENV_CONFIG[env_name]["classes"] + ["PandaOmron", "PandaGripper"]
        self.include_obj = ENV_CONFIG[env_name]["include_obj"]

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        segmentation_masks = self.get_segmentation_mask(obs_dict)

        for cam, mask in segmentation_masks.items():
            obs_dict[f"{cam}_segmentation_mask"] = mask

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        segmentation_masks = self.get_segmentation_mask(obs_dict)

        for cam, mask in segmentation_masks.items():
            obs_dict[f"{cam}_segmentation_mask"] = mask

        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        segmentation_masks = self.get_segmentation_mask(obs_dict)

        for cam, mask in segmentation_masks.items():
            obs_dict[f"{cam}_segmentation_mask"] = mask

        return obs_dict

    def get_segmentation_mask(self, obs_dict):
        masks = {}
        class_to_geom_ids = self.get_class_to_geom_ids()
        
        for cam in self.env.camera_names:
            masks[cam] = {}
            
            img_height = obs_dict[f"{cam}_image"].shape[0]
            img_width = obs_dict[f"{cam}_image"].shape[1]
            
            segmentation = get_camera_segmentation(self.env.sim, cam, img_height, img_width)[:, :, 1]
            
            for class_name, geom_ids in class_to_geom_ids.items():
                masks[cam][class_name] = np.zeros(segmentation.shape, dtype=bool)
                for geom_id in geom_ids:
                    masks[cam][class_name][segmentation == geom_id] = True
                    
                masks[cam][class_name] = masks[cam][class_name].squeeze()
                    
        return masks

    def get_class_to_geom_ids(self):
        geom_ids = {}

        if self.include_obj:
            obj_body_id = self.env.obj_body_id["obj"]
            geom_ids["obj"] = [geom_id for geom_id in range(self.env.sim.model.ngeom) if self.env.sim.model.geom_bodyid[geom_id] == obj_body_id]

        for class_name in self.classes:
            if class_name in self.env.model.classes_to_ids:
                geom_ids[class_name] = self.env.model.classes_to_ids[class_name]["geom"]

        return geom_ids

    def _check_success(self):
        return self.env._check_success()