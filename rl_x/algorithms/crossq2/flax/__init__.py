from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.crossq2.flax.crossq2 import CrossQ2
from rl_x.algorithms.crossq2.flax.default_config import get_config
from rl_x.algorithms.crossq2.flax.general_properties import GeneralProperties


CROSSQ2_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSQ2_FLAX, get_config, CrossQ2, GeneralProperties)
