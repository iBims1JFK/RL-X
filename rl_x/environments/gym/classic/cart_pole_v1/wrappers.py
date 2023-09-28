import numpy as np
import gymnasium as gym

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
    

    def reset(self):
        return self.env.reset()


    def get_episode_infos(self, info):
        episode_infos = []
        for single_info in info:
            maybe_episode_info = single_info.get("episode")
            if maybe_episode_info is not None:
                episode_infos.append(maybe_episode_info)
        return episode_infos


    def get_step_infos(self, info):
        step_infos = []
        keys_to_remove = ["episode", "final_observation", "TimeLimit.truncated"]
        info_keys = [info_key for info_key in list(info[0].keys()) if info_key not in keys_to_remove]
        for single_info in info:
            step_info = {}
            for key in info_keys:
                step_info[key] = single_info[key]
            step_infos.append(step_info)
        return step_infos


    def get_final_observation(self, info, id):
        return info[id]["final_observation"]

    
    def get_action_space_type(self):
        return ActionSpaceType.DISCRETE


    def get_single_action_space_shape(self):
        return self.action_space.shape


    def get_observation_space_type(self):
        return ObservationSpaceType.FLAT_VALUES
