

"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from envs import offline_ant_envs
from envs import fetch_envs

os.environ['SDL_VIDEODRIVER'] = 'dummy'


def load(env_name):
    """Loads the train and eval environments, as well as the obs_dim."""

    kwargs = {}
    if env_name == 'fetch_reach':
        CLASS = fetch_envs.FetchReachEnv
        max_episode_steps = 50
    elif env_name == 'fetch_push':
        CLASS = fetch_envs.FetchPushEnv
        max_episode_steps = 50
    elif env_name == 'fetch_reach_image':
        CLASS = fetch_envs.FetchReachImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_push_image':
        CLASS = fetch_envs.FetchPushImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_slide':
        CLASS = fetch_envs.FetchSlideEnv
        max_episode_steps = 50
    elif env_name == 'fetch_slide_image':
        CLASS = fetch_envs.FetchSlideImageEnv
        max_episode_steps = 50
    elif env_name == 'fetch_pick_and_place':
        CLASS = fetch_envs.FetchPickAndPlaceEnv
        max_episode_steps = 50
    elif env_name == 'fetch_pick_and_place_image':
        CLASS = fetch_envs.FetchPickAndPlaceImageEnv
        max_episode_steps = 50
    elif env_name.startswith('offline_ant'):
        def CLASS(): return offline_ant_envs.make_offline_d4rl(env_name)
        if 'umaze' in env_name:
            max_episode_steps = 700
        else:
            max_episode_steps = 1000
    else:
        raise NotImplementedError('Unsupported environment: %s' % env_name)

    gym_env = CLASS(**kwargs)
    obs_dim = gym_env.observation_space.shape[0] // 2
    return gym_env, obs_dim, max_episode_steps
