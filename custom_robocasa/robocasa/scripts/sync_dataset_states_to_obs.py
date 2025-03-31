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


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    args,
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

    max_segmented_pc_size = 0

    traj_len = states.shape[0]
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
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    # list to numpy array
    for k in traj:
        # if k == "initial_state_dict":
        #     continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                if isinstance(traj[k][kp][0], dict):
                    traj[k][kp] = TensorUtils.list_of_flat_dict_to_dict_of_list(
                        traj[k][kp]
                    )
                    for kpp in traj[k][kp]:
                        traj[k][kp][kpp] = np.array(traj[k][kp][kpp])
                else:
                    traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


""" The process that writes over the generated files to memory """


def write_traj_to_file(args, output_path, item):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    start_time = time.time()

    ep = item[0]
    traj = item[1]
    process_num = item[2]

    f_out.create_dataset("actions", data=np.array(traj["actions"]))
    if args.global_actions:
        f_out.create_dataset("global_actions", data=np.array(traj["global_actions"]))
    f_out.create_dataset("states", data=np.array(traj["states"]))
    f_out.create_dataset("rewards", data=np.array(traj["rewards"]))
    f_out.create_dataset("dones", data=np.array(traj["dones"]))

    for k in traj["obs"]:
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
        print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing {k}", flush=True)

        if isinstance(traj["obs"][k], OrderedDict):
            for kp in traj["obs"][k]:
                data = np.array(traj["obs"][k][kp])
                if args.no_compress:
                    f_out.create_dataset(f"obs/{k}/{kp}", data=data)
                else:
                    f_out.create_dataset(f"obs/{k}/{kp}", data=data, compression="gzip", chunks=True)
                del data
                gc.collect()
            continue

        data = np.array(traj["obs"][k])
        if args.no_compress:
            f_out.create_dataset(f"obs/{k}", data=data)
        else:
            f_out.create_dataset(f"obs/{k}", data=data, compression="gzip", chunks=True)
        del data
        gc.collect()

        if args.include_next_obs:
            data = np.array(traj["next_obs"][k])
            if args.no_compress:
                f_out.create_dataset(f"next_obs/{k}", data=data)
            else:
                f_out.create_dataset(f"next_obs/{k}", data=data, compression="gzip")
            del data
            gc.collect()

    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing datagen_info", flush=True)
    if "datagen_info" in traj:
        for k in traj["datagen_info"]:
            data = np.array(traj["datagen_info"][k])
            f_out.create_dataset(f"datagen_info/{k}", data=data)
            del data
            gc.collect()

    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing action_dict", flush=True)
    if "data/{}/action_dict".format(ep) in f:
        action_dict = f["data/{}/action_dict".format(ep)]
        for k in action_dict:
            data = np.array(action_dict[k][()])
            f_out.create_dataset(f"action_dict/{k}", data=data)
            del data
            gc.collect()

    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing initial_state_dict/model", flush=True)
    f_out.attrs["model_file"] = str(traj["initial_state_dict"]["model"])
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing initial_state_dict/ep_meta", flush=True)
    f_out.attrs["ep_meta"] = str(traj["initial_state_dict"]["ep_meta"])
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before writing num_samples", flush=True)
    f_out.attrs["num_samples"] = traj["actions"].shape[0]

    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    total_swap, used_swap, free_swap = map(int, os.popen("free -m | awk '/Swap/ {print $2, $3, $4}'").read().split())
    print(f"{used_memory}/{total_memory} | {used_swap}/{total_swap} before closing", flush=True)
    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    print("Writing has finished")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return


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

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
            "ep_meta", None
        )

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]

        traj = extract_trajectory(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            done_mode=args.done_mode,
            add_datagen_info=args.add_datagen_info,
            args=args,
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

        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"Memory usage: {used_memory}MB used out of {total_memory}MB total before putting {ep} at process {process_num}", flush=True)

        # mul_queue.put([ep, traj, process_num])
        out_file = os.path.join(output_folder, f"processed_{ep}.hdf5")
        write_traj_to_file(args, out_file, [ep, traj, process_num])

        total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        print(f"Memory usage: {used_memory}MB used out of {total_memory}MB total after putting {ep} at process {process_num}", flush=True)

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
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

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
        work_queue.put(index)

    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                args,
                work_queue,
            ),
        )
        processes.append(process)

    # extract_multiple_trajectories(0, args, work_queue, output_folder=os.path.dirname(args.dataset))

    # process1 = multiprocessing.Process(
    #     target=write_traj_to_file,
    #     args=(
    #         args,
    #         output_path,
    #         num_demos,
    #         mul_queue,
    #     ),
    # )
    # processes.append(process1)

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
        "--output_name",
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

    args = parser.parse_args()
    dataset_states_to_obs_multiprocessing(args)
