"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import gc
import os
import json
from typing import OrderedDict
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback

from robocasa.env_wrappers.camera_info_wrapper import CameraInfoWrapper
from robocasa.utils.env_utils import create_env
import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
from tqdm import tqdm

from robocasa.env_wrappers.robosuite_wrapper import RobosuiteWrapper
from robocasa.env_wrappers.segmentation_wrapper import SegmentationWrapper
from robocasa.utils.transform_utils import (
    axisangle2quat_numpy,
    mat2quat_numpy,
    quat2axisangle_numpy,
    quat2mat_numpy,
)

# from robomimic.utils.log_utils import log_warning

def get_leaf_memory_usage(d, path=()):
    """
    Recursively traverse a nested dictionary and yield (path, memory in bytes)
    for each leaf that is a list of NumPy arrays.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from get_leaf_memory_usage(v, path + (k,))
    elif isinstance(d, list) and all(isinstance(arr, np.ndarray) for arr in d):
        total_bytes = sum(arr.nbytes for arr in d)
        yield (path, total_bytes)
    elif isinstance(d, list) and all(isinstance(arr, dict) for arr in d):
        for i, arr in enumerate(d):
            yield from get_leaf_memory_usage(arr, path + (i,))
    elif isinstance(d, np.ndarray):
        total_bytes = d.nbytes
        yield (path, total_bytes)


def find_largest_leaf(d, top_n=5):
    """
    Finds the paths to the top N largest leaves and prints their memory allocation.
    """
    memory_usages = list(get_leaf_memory_usage(d))
    memory_usages.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {top_n} largest leaves by memory usage:\n")
    for path, mem in memory_usages[:top_n]:
        print(f"Path: {' -> '.join(map(str, path))} | Memory: {mem / (1024 ** 2):.2f} MB")

def prepare_hdf5_file(args, output_path, init_sample, total_length = 0, use_in_memory=False):
    """
    Prepares the HDF5 file for writing by creating the necessary groups and datasets.
    """
    if use_in_memory:
        # everything lives in RAM; nothing hits disk until we explicitly grab it
        f_out = h5py.File(output_path, "w", driver="core", backing_store=False)
    else:
        f_out = h5py.File(output_path, "w")
    f_out.create_group("obs")
    f_out.create_group("action_dict")

    compression = "gzip" if not args.no_compress else None

    f_out.create_dataset("rewards", (total_length,), maxshape=(total_length,), chunks=True, compression=compression)
    f_out.create_dataset("dones", (total_length,), maxshape=(total_length,), chunks=True, compression=compression)

    f_out.create_dataset("actions", (total_length,*init_sample["actions"].shape), maxshape=(total_length,*init_sample["actions"].shape), chunks=(1, *init_sample["actions"].shape), compression=compression)
    f_out.create_dataset("states", maxshape=(total_length,*init_sample["states"].shape), chunks=(1, *init_sample["states"].shape), shape=(total_length, *init_sample["states"].shape), compression=compression)


    for k, v in init_sample["obs"].items():
        if isinstance(v, np.ndarray):
            f_out.create_dataset(f"obs/{k}", maxshape=(total_length, *v.shape), chunks=(1, *v.shape), compression=compression, shape=(total_length, *v.shape))
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                dtype = sub_v.dtype
                f_out.create_dataset(f"obs/{k}/{sub_k}", maxshape=(total_length, *sub_v.shape), chunks=(1, *sub_v.shape), compression=compression, shape=(total_length, *sub_v.shape), dtype=dtype)

    return f_out

def prepare_dict_for_hdf5_file(args, init_sample):
    hdf5_dict = {}
    hdf5_dict["rewards"] = []
    hdf5_dict["dones"] = []

    hdf5_dict["actions"] = []
    hdf5_dict["states"] = []

    hdf5_dict["obs"] = {}
    for k, v in init_sample["obs"].items():
        if isinstance(v, np.ndarray):
            hdf5_dict["obs"][k] = []
        elif isinstance(v, dict):
            hdf5_dict["obs"][k] = {}
            for sub_k, sub_v in v.items():
                hdf5_dict["obs"][k][sub_k] = []

    return hdf5_dict

def append_to_hdf5_dict(hdf5_dict, data, index):
    """
    Appends data to the HDF5 dictionary.
    """
    # Rewards
    hdf5_dict["rewards"].append(data["rewards"])
    hdf5_dict["dones"].append(data["dones"])
    hdf5_dict["states"].append(data["states"])
    # Actions
    hdf5_dict["actions"].append(data["actions"])

    # Observations
    for k, v in data["obs"].items():
        if isinstance(v, np.ndarray):
            hdf5_dict["obs"][k].append(v)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                hdf5_dict["obs"][k][sub_k].append(sub_v)

def append_to_hdf5_file(f_out, data, index):
    """
    Appends data to the HDF5 file.
    """
    # Rewards
    f_out["rewards"][index] = data["rewards"]
    f_out["dones"][index] = data["dones"]
    f_out["states"][index] = data["states"]
    # Actions
    f_out["actions"][index] = data["actions"]

    # Observations
    f_obs = f_out["obs"]
    for k, v in data["obs"].items():
        if isinstance(v, np.ndarray):
            if k not in f_obs:
                raise ValueError(f"Key {k} not found in f_obs")
            else:
                f_obs[k][index] = v
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if sub_k not in f_obs[k]:
                    raise ValueError(f"Key {v}/{sub_k} not found in f_obs")
                else:
                    f_obs[f"{k}/{sub_k}"][index] = sub_v

def finish_hdf5_file_from_dict(hdf5_dict, output_path, args):
    """
    Prepares the HDF5 file for writing by creating the necessary groups and datasets.
    """
    f_out = h5py.File(output_path, "w")
    f_out.create_group("obs")
    f_out.create_group("action_dict")

    compression = "gzip" if not args.no_compress else None

    rewards = np.asarray(hdf5_dict["rewards"])
    f_out.create_dataset("rewards", data=rewards, chunks=True, compression=compression)

    dones = np.asarray(hdf5_dict["dones"])
    f_out.create_dataset("dones", data=dones, chunks=True, compression=compression)

    actions = np.asarray(hdf5_dict["actions"])
    f_out.create_dataset("actions", data=actions, chunks=True, compression=compression)

    states = np.asarray(hdf5_dict["states"])
    f_out.create_dataset("states", data=states, chunks=True, compression=compression)

    for k, v in hdf5_dict["obs"].items():
        if isinstance(v, list):
            np_value = np.asarray(v)
            f_out.create_dataset(f"obs/{k}", data=np_value, chunks=True, compression=compression)
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                np_value = np.asarray(sub_v)
                f_out.create_dataset(f"obs/{k}/{sub_k}", data=np_value, chunks=True, compression=compression)

    f_out.close()

    DatasetUtils.sync_extract_action_dict(dataset=output_path)
    print(f"Writing has finished ${output_path}")
    return #f_out


def finish_hdf5_file(f_out, traj, output_path, action_dict):
    """
    Finalizes the HDF5 file by closing it.
    """
    for k in action_dict:
        data = np.array(action_dict[k][()])
        f_out.create_dataset(f"action_dict/{k}", data=data)
        del data
        gc.collect()

    f_out.attrs["model_file"] = str(traj["initial_state_dict"]["model"])
    f_out.attrs["ep_meta"] = str(traj["initial_state_dict"]["ep_meta"])
    f_out.attrs["num_samples"] = traj["actions"].shape[0]

    # write everything into the in‑memory file
    if f_out.driver == "core":
        # pull the HDF5 binary image out of RAM
        img = f_out.id.get_file_image()
        # write it down once, here at the end
        with open(output_path, "wb") as disk_f:
            disk_f.write(img)

    f_out.close()

    DatasetUtils.sync_extract_action_dict(dataset=output_path)
    print(f"Writing has finished ${output_path}")

def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    action_dict,
    done_mode,
    args,
    output_path,
    ep,
    add_datagen_info=False,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    initial_state["model"] = initial_state["model"].replace(
        '<site name="gripper0_right_grip_site_cylinder" pos="0 0 0" size="0.005 10" group="1" type="cylinder" rgba="0 1 0 0.3"/>',
        '<site name="gripper0_right_grip_site_cylinder" pos="0 0 0" size="0.005 0.001" group="1" type="cylinder" rgba="0 1 0 0.3"/>')
    initial_state["model"] = initial_state["model"].replace(
        '<site name="gripper0_right_ee_z" pos="0 0 0.1" size="0.005 0.1" group="1" type="cylinder" rgba="0 0 1 0"/>',
        '<site name="gripper0_right_ee_z" pos="0 0 0.1" size="0.005 0.001" group="1" type="cylinder" rgba="0 0 1 0"/>')
    initial_state["model"] = initial_state["model"].replace(
        '<site name="gripper0_right_ee_y" pos="0 0.1 0" quat="0.707105 0.707108 0 0" size="0.005 0.1" group="1" type="cylinder" rgba="0 1 0 0"/>',
        '<site name="gripper0_right_ee_y" pos="0 0.1 0" quat="0.707105 0.707108 0 0" size="0.005 0.001" group="1" type="cylinder" rgba="0 1 0 0"/>')
    initial_state["model"] = initial_state["model"].replace(
        '<site name="gripper0_right_ee_x" pos="0.1 0 0" quat="0.707105 0 0.707108 0" size="0.005 0.1" group="1" type="cylinder" rgba="1 0 0 0"/>',
        '<site name="gripper0_right_ee_x" pos="0.1 0 0" quat="0.707105 0 0.707108 0" size="0.005 0.001" group="1" type="cylinder" rgba="1 0 0 0"/>')
    env.reset_to(initial_state)

    # get updated ep meta in case it's been modified
    ep_meta = env.env.get_ep_meta()
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    if args.global_actions:
        base_rot = env.sim.data.get_site_xmat(f"mobilebase{env.robots[0].idn}_center")

        global_actions = actions.copy()

        global_actions[:, :3] = np.einsum("ij, nj -> ni", base_rot, actions[:, :3])

        rot_mats = quat2mat_numpy(axisangle2quat_numpy(actions[:, 3:6]))
        global_rot_mats = np.einsum(
            "ij, njk, kl -> nil", base_rot, rot_mats, base_rot.T
        )
        global_actions[:, 3:6] = quat2axisangle_numpy(mat2quat_numpy(global_rot_mats))

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        # actions_abs=[],
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )
    if args.global_actions:
        traj["global_actions"] = np.array(global_actions)

    # hdf5 file to write to
    hdf5_file = None
    hdf5_dict = None

    traj_len = 20 #states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in tqdm(range(traj_len)):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        obs_keys_to_remove = []
        for obs_key in obs:
            if args.dont_store_image and "image" in obs_key:
                obs_keys_to_remove.append(obs_key)

            if args.dont_store_depth and "depth" in obs_key:
                obs_keys_to_remove.append(obs_key)

            # if args.segmentation and "mask" in obs_key:
            #     obs_keys_to_remove.append(obs_key)

        for obs_key in obs_keys_to_remove:
            del obs[obs_key]

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env._check_success()
        done = int(done)

        # get the absolute action
        local_action = np.zeros_like(actions[t])
        action_abs = env.convert_rel_to_abs_action(local_action).copy() #actions[t])
        action_abs2 = env.convert_rel_to_abs_action(actions[t]).copy()
        rel_action = env.convert_abs_to_rel_action(action_abs2)
        rel_action2 = actions[t]
        #
        # print("action_abs", action_abs)
        # print("action_abs2", action_abs2)
        # print("rel_action", rel_action)
        # print("rel_action2", rel_action2)
        # print("diff", action_abs - action_abs2)

        # print("action_abs", action_dict["abs_pos"][t])
        # print("robot0_base_to_eef_pos", obs["robot0_base_to_eef_pos"])
        # print("diff", action_dict["abs_pos"][t] - obs["robot0_base_to_eef_pos"])

        # collect transition
        # traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)

        sample = {
            "obs": obs,
            "rewards": r,
            "dones": done,
            "datagen_info": datagen_info,
            "states": states[t],
            "actions": actions[t],
        }

        if hdf5_dict is None:
            # create hdf5 file if it doesn't exist
            # hdf5_file = prepare_hdf5_file(args, output_path, sample, total_length=traj_len, use_in_memory=args.use_in_memory)
            hdf5_dict = prepare_dict_for_hdf5_file(args, sample)

        # append_to_hdf5_file(hdf5_file, sample, t)
        append_to_hdf5_dict(hdf5_dict, sample, t)

        # find_largest_leaf(traj)
    # finish_hdf5_file(hdf5_file, traj, output_path, action_dict)
    f_out = finish_hdf5_file_from_dict(hdf5_dict, output_path, args)
    return traj, hdf5_file


"""Write observations to hdf5 file"""

def retrieve_new_index(work_queue):
    try:
        tmp = work_queue.get(False)
        return tmp
    except queue.Empty:
        return -1


def extract_multiple_trajectories(
    process_num, args, work_queue, output_folder=None
):
    # create environment to use for data processing

    if args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = create_env_with_wrappers(env_meta["env_name"], args)

    start_time = time.time()

    # print("==== Using environment with the following metadata ====")
    # print(json.dumps(env.serialize(), indent=4))
    # print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    ind = retrieve_new_index(work_queue)
    while (not work_queue.empty()) and (ind != -1):
        # print("Running {} index".format(ind))
        ep = demos[ind]

        out_file = os.path.join(output_folder, f"processed_{ep}.hdf5")

        if os.path.exists(out_file):
            ind = retrieve_new_index(work_queue)
            continue

        # if "processed_demo_10.hdf5" not in out_file:
        #     ind = retrieve_new_index(work_queue)
        #     continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
            "ep_meta", None
        )

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        action_dict= f["data/{}/action_dict".format(ep)]

        traj = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            action_dict=action_dict,
            done_mode=args.done_mode,
            add_datagen_info=args.add_datagen_info,
            output_path=out_file,
            args=args,
            ep=ep
        )

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

        ep_grp = f["data/{}".format(ep)]

        states = ep_grp["states"][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = ep_grp.attrs["model_file"]
        initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

        # store transitions
        # write_traj_to_file(args, out_file, [ep, traj, process_num])

        ind = retrieve_new_index(work_queue)

    f.close()
    print("Process {} finished".format(process_num))


def create_env_with_wrappers(env_name, args):
    base_env = create_env(
        env_name=env_name,
        camera_names=args.camera_names,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )

    if args.segmentation:
        env = CameraInfoWrapper(SegmentationWrapper(RobosuiteWrapper(base_env), env_name=env_name))
    else:
        env = RobosuiteWrapper(base_env)

    return env


def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_folder = args.output_folder
    if output_folder is None:
        output_folder = os.path.dirname(args.dataset)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("input file: {}".format(args.dataset))
    print("output folder: {}".format(output_folder))

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    num_processes = args.num_procs

    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        # if "17" not in demos[index] and "18" not in demos[index]:
        #     continue
        work_queue.put(index)

    processes = []
    # for i in range(num_processes):
    #     process = multiprocessing.Process(
    #         target=extract_multiple_trajectories,
    #         args=(
    #             i,
    #             args,
    #             work_queue,
    #             output_folder
    #         ),
    #     )
    #     processes.append(process)

    extract_multiple_trajectories(0, args, work_queue, output_folder)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_folder",
        type=str,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )

    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    parser.add_argument(
        "--pc_size",
        type=int,
        default=1024,
        help="number of points in point cloud",
    )

    parser.add_argument(
        "--pc_obj_max_size",
        type=int,
        default=512,
        help="max number of points of object in the point cloud",
    )

    parser.add_argument(
        "--pc_in_global_frame",
        action="store_true",
        help="whether to return point cloud in global frame",
    )

    parser.add_argument(
        "--keep_full_pc",
        action="store_true",
        help="whether to keep full point cloud in observation",
    )

    parser.add_argument(
        "--segmentation",
        action="store_true",
        help="whether to include segmented point cloud",
    )

    parser.add_argument(
        "--global_actions",
        action="store_true",
        help="whether to include global actions",
    )

    parser.add_argument(
        "--dont_store_image",
        action="store_true",
        help="whether to store image observations",
    )

    parser.add_argument(
        "--dont_store_depth",
        action="store_true",
        help="whether to store depth observations",
    )

    parser.add_argument(
        "--use_in_memory",
        action="store_true",
        help="whether to use in-memory hdf5 file",
    )

    args = parser.parse_args()
    dataset_states_to_obs_multiprocessing(args)
