from .direct_rl_vla_env import DirectRLVlaEnv


class DirectRLOpenVLAEnv(DirectRLVlaEnv):
    default_vla_model = "openvla"
