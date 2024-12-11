import math
import random
import copy

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from gymnasium.wrappers import TimeLimit
from omegaconf import DictConfig
import multiprocessing as mp
from tqdm import tqdm

import loco_mujoco  # noqa

from robobase.utils import add_demo_to_replay_buffer
from robobase.envs.env import EnvFactory, DemoEnv
from robobase.envs.wrappers import (
    OnehotTime,
    FrameStack,
    RescaleFromTanh,
    ActionSequence,
)
from robobase.envs.utils.locomujoco_utils import (
    TASK_DESCRIPTION,
)

import logging


def _task_name_to_description(task_name: str) -> str:
    return TASK_DESCRIPTION.get(task_name, None)


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
        episode_return += rew

    return episode_return


def split_into_trajectories(
    observations, actions, rewards, masks, dones_float, next_observations
):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append(
            (
                observations[i],
                actions[i],
                rewards[i],
                masks[i],
                dones_float[i],
                next_observations[i],
            )
        )
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def get_traj_dataset(env, sorting=True):
    dataset = env.create_dataset()
    trajs = split_into_trajectories(
        dataset["states"],
        dataset["actions"],
        dataset["rewards"],
        dataset["last"],
        dataset["absorbing"],
        dataset["next_states"],
    )
    if sorting:
        trajs.sort(key=compute_returns)

    # Convert traj to RoboBase demo
    converted_trajs = [[]]
    for traj in trajs:
        # The first transition only contains (obs, info),
        # corresponding to the ouput of env.reset()
        converted_trajs[-1].append([traj[0][0], {"demo": 1}])

        # For the subsequent transitions. we convert
        # (obs, actions, rew, masks, dones_float, next_obs)
        # to (next_obs, rew, term, trunc, next_info) required by robobase.DemoEnv.
        for ts in traj:
            # truncation is always False as the time limit is handled by
            # the `TimeLimit` wrapper.
            converted_trajs[-1].append(
                [ts[5], ts[2], ts[4], False, {"demo_action": ts[1], "demo": 1}]
            )

        # If traj length equals to max_episode_len, then termination=False and
        # truncated=True
        # NOTE: For d4rl, the collected trajectory has 1 less step then
        # max_episode_steps.
        if len(traj) == env.spec.max_episode_steps - 1:
            converted_trajs[-1][-1][2] = False

        converted_trajs.append([])
    converted_trajs.pop()  # Remove the last empty traj

    # NOTE: this raw_dataset is not sorted
    return converted_trajs


class LocoMujoco(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        task_name,
        from_pixels: bool = False,
        action_repeat: int = 1,
        visual_observation_shape: tuple[int, int] = (84, 84),
        render_mode: str = "rgb_array",
        use_rlhf: bool = False,
        query_keys: list[str] = ["right"],
        reward_mode: str = "dense",
        reward_term_type: str = "all",
        initial_terms: list[float] = [],
    ):
        self._task_name = task_name
        self._from_pixels = from_pixels
        self._action_repeat = action_repeat
        self._viewer = None
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert reward_mode in ["initial", "dense"]
        self._i = 0
        self._render_mode = render_mode
        self._reward_mode = reward_mode
        self._reward_term_type = reward_term_type
        self._initial_terms = initial_terms
        self._pixels_key = "pixels"
        self._query_keys = query_keys
        assert all(
            key == self._pixels_key for key in query_keys
        ), f"Only {self._pixels_key} view is supported"

        self._locomujoco_env = None

        # Set up rendering parameters
        height, width = visual_observation_shape
        logging.info(
            f"Creating LocoMujoco environment with task name: {self._task_name}"
        )

        self._locomujoco_env = gym.make(
            "LocoMujoco",
            env_name=self._task_name,
            width=width,
            height=height,
            render_mode=render_mode,
        )

        # Set up observation space
        _obs_space = {}
        if from_pixels:
            image_shape = [3, height, width]  # (channels, height, width)
            # For pixel observations, use RGB image space
            rgb_space = spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8)
            _obs_space["rgb"] = rgb_space
        else:
            _obs_space["low_dim_state"] = spaces.Box(
                low=self._locomujoco_env.observation_space.low,
                high=self._locomujoco_env.observation_space.high,
                dtype=np.float32,
            )

        self._use_rlhf = use_rlhf
        if use_rlhf:
            for key in self._query_keys:
                _obs_space[f"query_video_{key}"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                )
        self.observation_space = spaces.Dict(_obs_space)
        self.action_space = self._locomujoco_env.action_space
        self.reward_space = gym.spaces.Dict(
            {
                "target_velocity": gym.spaces.Box(
                    low=1e-10, high=1e1, shape=(), dtype=np.float32
                ),
            }
        )

        if len(self._initial_terms) == 0:
            self._initial_terms = [key for key in self.reward_space.keys()]
        else:
            self._initial_terms = [f"Reward/{key}" for key in self._initial_terms]

        if self._reward_term_type == "all":
            self._reward_terms = [key for key in self.reward_space.keys()]
        elif self._reward_term_type == "initial":
            self._reward_terms = self._initial_terms
        else:
            raise ValueError(
                f"reward_term_type must be one of ['all', 'initial'], got {self._reward_term_type}"
            )

        self.reward_space = spaces.Dict(
            {
                k: gym.spaces.Box(
                    low=self.reward_space[k].low,
                    high=self.reward_space[k].high,
                    shape=self.reward_space[k].shape,
                )
                for k in self._reward_terms
            }
        )

        self.initial_reward_scale = {
            k: self.reward_space[k].high for k in self._initial_terms
        }

    def _get_obs(self, observation):
        ret_obs = {}
        if self._from_pixels:
            ret_obs["rgb"] = self._locomujoco_env.render().transpose([2, 0, 1]).copy()
        else:
            ret_obs["low_dim_state"] = observation.astype(np.float32)

        if self._use_rlhf:
            ret_obs[f"query_video_{self._query_keys[0]}"] = self.render().copy()
        return ret_obs

    def step(self, action):
        reward = 0
        info = {"task_reward": 0.0, **{f"Reward/{k}": 0.0 for k in self._reward_terms}}
        for _ in range(self._action_repeat):
            (
                next_obs,
                task_reward,
                terminated,
                truncated,
                _info,
            ) = self._locomujoco_env.step(action)
            # target velocity is the reward
            _info["target_velocity"] = task_reward
            info["task_reward"] += task_reward
            if self._reward_mode == "initial":
                _reward = np.sum(
                    [
                        self.initial_reward_scale[key] * _info[key]
                        for key in self._initial_terms
                    ]
                )
            else:
                _reward = task_reward

            reward += _reward
            for key in self._reward_terms:
                info[f"Reward/{key}"] += _info[key]
            if terminated or truncated:
                break
        # See https://github.com/google-deepmind/dm_control/blob/f2f0e2333d8bd82c0b6ba83628fe44c2bcc94ef5/dm_control/rl/control.py#L115C18-L115C29
        truncated = truncated and not terminated
        assert not np.any(
            terminated and truncated
        ), "Can't be both terminal and truncated."
        self._i += 1
        return self._get_obs(next_obs), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        agym_obs, info = self._locomujoco_env.reset(seed=seed)
        info.update({key: 0.0 for key in self.reward_space.keys()})
        info.update({"task_reward": 0.0})
        return self._get_obs(agym_obs), info

    def render(self) -> None:
        image = self._locomujoco_env.render()
        # Rotate image 90 degrees clockwise
        image = np.rot90(image, k=-1)
        return image


class LocoMujocoEnvFactory(EnvFactory):
    def _wrap_env(self, env, cfg):
        env = RescaleFromTanh(env)
        if cfg.env.episode_length != 1000:
            # Used in unit tests.
            env = TimeLimit(env, cfg.env.episode_length)
        if cfg.use_onehot_time_and_no_bootstrap:
            env = OnehotTime(
                env, cfg.env.episode_length // cfg.action_repeat
            )  # Time limits are handles by DMC
        env = ActionSequence(env, cfg.action_sequence)
        env = FrameStack(env, cfg.frame_stack)
        return env

    def _create_env(self, cfg: DictConfig) -> LocoMujoco:
        return LocoMujoco(
            task_name=cfg.env.task_name,
            from_pixels=cfg.pixels,
            action_repeat=cfg.action_repeat,
            visual_observation_shape=cfg.visual_observation_shape,
            render_mode="rgb_array",
            use_rlhf=cfg.rlhf.use_rlhf,
            query_keys=cfg.env.query_keys,
            reward_mode=cfg.env.reward_mode,
            reward_term_type=cfg.env.reward_term_type,
            initial_terms=cfg.env.initial_terms,
        )

    def make_train_env(self, cfg: DictConfig) -> gym.vector.VectorEnv:
        vec_env_class = gym.vector.AsyncVectorEnv
        kwargs = dict(context=None)
        # vec_env_class = gym.vector.SyncVectorEnv
        # kwargs = dict()
        return vec_env_class(
            [
                lambda: self._wrap_env(
                    self._create_env(cfg),
                    cfg,
                )
                for _ in range(cfg.num_train_envs)
            ],
            **kwargs,
        )

    def make_eval_env(self, cfg: DictConfig) -> gym.Env:
        env, (self._action_space, self._observation_space) = self._wrap_env(
            self._create_env(cfg),
            cfg,
            return_raw_spaces=True,
        )
        return env

    def get_task_description(self, cfg: DictConfig) -> str:
        return _task_name_to_description(cfg.env.task_name)

    def _get_demo_fn(self, cfg: DictConfig, num_demos: int) -> None:
        env = self._create_env(cfg)
        demos = get_traj_dataset(env)
        env.close()
        logging.info("Finished loading demos.")
        return demos

    def collect_or_fetch_demos(self, cfg: DictConfig, num_demos: int):
        manager = mp.Manager()
        mp_list = manager.list()
        p = mp.Process(
            target=self._get_demo_fn,
            args=(
                cfg,
                num_demos,
                mp_list,
            ),
        )
        p.start()
        p.join()

        # Only extract num_demos from the full dataset
        all_demos = list(mp_list)
        if not math.isfinite(num_demos):
            num_demos = len(all_demos)

        if cfg.env.random_traj:
            self._raw_demos = random.sample(all_demos, num_demos)
        else:
            self._raw_demos = all_demos[:num_demos]

    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        self._demos = self._raw_demos

    def load_demos_into_replay(self, cfg: DictConfig, buffer):
        """See base class for documentation."""
        assert hasattr(self, "_demos"), (
            "There's no _demo attribute inside the factory, "
            "Check `collect_or_fetch_demos` is called before calling this method."
        )
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._demos), self._action_space, self._observation_space
            ),
            cfg,
        )
        for _ in range(len(self._demos)):
            add_demo_to_replay_buffer(demo_env, buffer)
