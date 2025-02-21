# minimal-atc-rl/envs/__init__.py
import gymnasium
from gymnasium.envs.registration import register

register(
    id='AtcEnv-v0',
    entry_point='envs.atc.atc_gym:AtcGym',
)