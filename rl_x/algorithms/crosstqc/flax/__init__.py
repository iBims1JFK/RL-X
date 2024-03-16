from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.crosstqc.flax.crosstqc import CrossTQC
from rl_x.algorithms.crosstqc.flax.default_config import get_config
from rl_x.algorithms.crosstqc.flax.general_properties import GeneralProperties


CROSSTQC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSTQC_FLAX, get_config, CrossTQC, GeneralProperties)
