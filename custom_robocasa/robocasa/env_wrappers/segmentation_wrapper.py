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
        "classes": ["Counter", "Stove", "Stovetop", "container"],
        "include_obj": True
    },
    "PnPStoveToCounter": {
        "classes": ["Counter", "Stove", "Stovetop", "container"],
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
    "CloseSingleDoor": {
        "classes": ["door_obj"],
        "include_obj": False
    },
    "OpenDoubleDoor": {
        "classes": ["door_obj"],
        "include_obj": False
    },
    "OpenDrawer": {
        "classes": ["drawer_obj"],
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

        self.segmentation_active = True

        self.class_to_geom_ids = None


    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        if self.segmentation_active:
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

            # import matplotlib.pyplot as plt
            # plt.imshow(segmentation)
            # plt.show()

            for class_name, geom_ids in class_to_geom_ids.items():
                geom_ids_arr = np.array(geom_ids)
                # Broadcasting comparison over the last axis, then taking any() along that axis.
                # masks[cam][class_name] = (segmentation[..., None] == geom_ids_arr).any(axis=-1)

                masks[cam][class_name] = np.isin(segmentation, geom_ids_arr)
                # masks[cam][class_name] = masks[cam][class_name].squeeze()

        return masks

    def get_class_to_geom_ids(self):
        geom_ids = {}

        blacklist = ["PandaGripper", "PandaOmron", "distr_counter", "distr_sink", "distr_cab", "distr_counter_1", "distr_counter_2"]

        if self.include_obj:
            obj_body_id = self.env.obj_body_id["obj"]
            geom_ids["obj"] = [geom_id for geom_id in range(self.env.sim.model.ngeom) if self.env.sim.model.geom_bodyid[geom_id] == obj_body_id and geom_id not in blacklist]

        for obj_body_key, obj_body_id in self.env.obj_body_recursive_ids.items():
            if obj_body_key != "obj" and obj_body_key not in blacklist:
                geom_ids[obj_body_key] = self.env.obj_body_recursive_ids[obj_body_key]

        # for class_name in self.env.model.classes_to_ids.keys():
        #     if class_name in blacklist:
        #         continue
        #     geom_ids[class_name] = self.env.model.classes_to_ids[class_name]["geom"]

        for class_name in self.classes:
            if class_name in self.env.model.classes_to_ids and class_name not in blacklist:
                geom_ids[class_name] = self.env.model.classes_to_ids[class_name]["geom"]

        # if "door_obj" in self.env.obj_body_id:
        #     print(f"Adding door_obj segmentation for {self.env.obj_body_id['door_obj']}")
        #     print(self.env.obj_body_recursive_ids["door_obj"])
        #     obj_body_id = self.env.obj_body_id["door_obj"]
        #     geom_ids["door_obj"] = [geom_id for geom_id in range(self.env.sim.model.ngeom) if self.env.sim.model.geom_bodyid[geom_id] == obj_body_id]
        #     print(f"Found {len(geom_ids['door_obj'])} door_obj geom_ids")
        #     print(geom_ids["door_obj"])

        if hasattr(self.env.env, "door_fxtr"):
            geom_names = self.env.env.door_fxtr.visual_geoms + self.env.env.door_fxtr.contact_geoms
            geom_ids["door_obj"] = [self.env.sim.model.geom_names.index(geom_name) for geom_name in geom_names if geom_name in self.env.sim.model.geom_names]

            a_results = []
            a_stack = [self.env.sim.model.geom_bodyid[geom_id] for geom_id in geom_ids["door_obj"]]
            a_stack = list(set(a_stack))  # Ensure unique body IDs
            door_geom_ids = []
            while len(a_stack) > 0:
                a = a_stack.pop()
                a_geom_ids = [geom_id for geom_id in range(self.env.sim.model.ngeom) if self.env.sim.model.geom_bodyid[geom_id] == a]
                door_geom_ids.extend(a_geom_ids)

                a_stack.extend([geom_id for geom_id in range(self.env.sim.model.nbody) if self.env.sim.model.body_parentid[geom_id] == a])
                a_stack = list(set(a_stack))

            door_geom_ids = list(set(door_geom_ids))
            geom_ids["door_obj"] = door_geom_ids

        if hasattr(self.env.env, "drawer"):
            geom_names = self.env.env.drawer.visual_geoms + self.env.env.drawer.contact_geoms
            geom_ids["drawer_obj"] = [self.env.sim.model.geom_names.index(geom_name) for geom_name in geom_names if geom_name in self.env.sim.model.geom_names]

            a_results = []
            a_stack = [self.env.sim.model.geom_bodyid[geom_id] for geom_id in geom_ids["drawer_obj"]]
            drawer_geom_ids = []
            while len(a_stack) > 0:
                a = a_stack.pop()
                a_geom_ids = [geom_id for geom_id in range(self.env.sim.model.ngeom) if self.env.sim.model.geom_bodyid[geom_id] == a]
                drawer_geom_ids.extend(a_geom_ids)

                a_stack.extend([geom_id for geom_id in range(self.env.sim.model.nbody) if self.env.sim.model.body_parentid[geom_id] == a])

            drawer_geom_ids = list(set(drawer_geom_ids))
            geom_ids["drawer_obj"] = drawer_geom_ids

        return geom_ids

    def activate_segmentation(self):
        self.segmentation_active = True

    def deactivate_segmentation(self):
        self.segmentation_active = False

    def _check_success(self):
        return self.env._check_success()