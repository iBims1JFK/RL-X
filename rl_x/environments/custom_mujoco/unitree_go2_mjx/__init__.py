from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.custom_mujoco.unitree_go2_mjx.create_env import create_env
from rl_x.environments.custom_mujoco.unitree_go2_mjx.default_config import get_config
from rl_x.environments.custom_mujoco.unitree_go2_mjx.general_properties import GeneralProperties


GENERAL_LOCOMOTION_ENV = extract_environment_name_from_file(__file__)
register_environment(GENERAL_LOCOMOTION_ENV, get_config, create_env, GeneralProperties)
