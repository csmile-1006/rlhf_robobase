"""Wrapper for allowing action sequences."""

from copy import deepcopy
from typing import Any, Dict, Union

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from robobase import utils


class ActionSequence(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Wrapper for allowing action sequences."""

    def __init__(self, env: gym.Env, sequence_length: int):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self._sequence_length = sequence_length
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.is_demo_env = getattr(env, "is_demo_env", False)
        if self.is_vector_env:
            raise NotImplementedError(
                "It is not possible to use this wrapper with a VecEnv."
            )
        low, high = env.action_space.low, env.action_space.high
        self.action_space = Box(
            np.expand_dims(low, 0).repeat(sequence_length, 0),
            np.expand_dims(high, 0).repeat(sequence_length, 0),
            dtype=self.action_space.dtype,
        )

    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)
        for i, sub_action in enumerate(action):
            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break
        assert action_idx_reached <= self._sequence_length
        info["action_sequence_mask"] = (
            np.arange(self._sequence_length) < action_idx_reached
        ).astype(int)
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        return observation, total_reward, termination, truncation, info

    def step(self, action):
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action to be of shape {self.action_space.shape}, "
                f"but got action of shape {action.shape}."
            )
        return self._step_sequence(action)


class RecedingHorizonControl(ActionSequence):
    """Receding horizon control with temporal ensembling of ACT.

    This wrapper allows agent predict an action sequence of length N,
    but performs receding horizon control of only K <= N steps of actions.
    We also support temporal ensembling (from ALOHA https://arxiv.org/abs/2304.13705),
    which caches the previous actions and outputs a weighted average of them.
    """

    def __init__(
        self,
        env: gym.Env,
        sequence_length: int,
        time_limit: int,
        execution_length: int,
        temporal_ensemble: bool = True,
        gain: float = 0.01,
        stddev_schedule: Union[float, str] = "linear(1.0, 0.0, 1000000)",
        num_explore_steps: int = 1000000,
        eval_mode: bool = False,
    ):
        """Init.

        Args:
            env: The gym env to wrap.
            sequence_length: Action sequence length.
            time_limit: The time limit of the env for creating buffers.
            execution_length: The execution length of the receding horizion control.
            temporal_ensemble: Whether to use temporal ensembling. Defaults to True.
            gain: Temporal ensembling gain. Defaults to 0.01.
        """
        super().__init__(env, sequence_length)
        self._time_limit = time_limit
        self._execution_length = execution_length
        self._temporal_ensemble = temporal_ensemble
        self._gain = gain

        self._total_step = 0
        self._eval_mode = eval_mode
        self._stddev_schedule = stddev_schedule
        self._num_explore_steps = num_explore_steps

        self._init_action_history()

    def _init_action_history(self):
        """Initialize the action history buffer.

        We store the history actions within a buffer of shape [T, T + L, A],
        where T is the time limit, L is the sequence length, and A is the action size.

        For example, self._action_history[t, t:t + L] stores the predicted action
        sequence of size A and length L at time step t.
        """
        self._action_history = np.zeros(
            [
                self._time_limit,
                self._time_limit + self._sequence_length,
                self.action_space.shape[-1],
            ],
            dtype=self.action_space.dtype,
        )
        self._cur_step = 0

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._init_action_history()
        return super().reset(seed=seed, options=options)

    @property
    def cur_step(self):
        return self._cur_step

    def _step_sequence(self, action):
        total_reward = np.array(0.0)
        action_idx_reached = 0
        if self.is_demo_env:
            demo_actions = np.array(action)

        self._action_history[
            self._cur_step, self._cur_step : self._cur_step + self._sequence_length
        ] = action

        new_action = deepcopy(action)
        for i, sub_action in enumerate(action):
            if self._temporal_ensemble and self._sequence_length > 1:
                # Select all predicted actions for self._cur_step. This will cover the
                # actions from [cur_step - sequence_length + 1, cur_step)
                # Note that not all actions in this range will be valid as we might have
                # execution_length > 1, which skips some of the intermediate steps.
                cur_actions = self._action_history[:, self._cur_step]
                indices = np.all(cur_actions != 0, axis=1)
                cur_actions = cur_actions[indices]

                # earlier predicted actions will have smaller weights.
                exp_weights = np.exp(-self._gain * np.arange(len(cur_actions)))
                exp_weights = (exp_weights / exp_weights.sum())[:, None]
                sub_action = (cur_actions * exp_weights).sum(axis=0)
                if not self._eval_mode:
                    # add noise to the action in training mode.
                    if self._total_step < self._num_explore_steps:
                        sub_action = np.random.uniform(
                            -1.0, 1.0, size=sub_action.shape
                        ).astype(sub_action.dtype)
                    else:
                        stddev = utils.schedule(self._stddev_schedule, self._total_step)
                        sub_action = np.clip(
                            sub_action
                            + np.random.normal(0, stddev, size=sub_action.shape).astype(
                                sub_action.dtype
                            ),
                            -1.0,
                            1.0,
                        )
                new_action[i] = sub_action

            observation, reward, termination, truncation, info = self.env.step(
                sub_action
            )
            self._cur_step += 1
            self._total_step += 1
            if self.is_demo_env:
                demo_actions[i] = info.pop("demo_action")
            total_reward += reward
            action_idx_reached += 1
            if termination or truncation:
                break

            if not self.is_demo_env:
                if action_idx_reached == self._execution_length:
                    break

        assert action_idx_reached <= self._sequence_length
        # TODO not sure this is correct in the case of receding horizon control
        #      Currently, for every action_sequence, all actions that are not applied
        #      will be masked out!!
        info["action_sequence_mask"] = (
            np.arange(self._sequence_length) < action_idx_reached
        ).astype(int)
        if self._temporal_ensemble and self._sequence_length > 1:
            info["temporal_ensemble_action"] = new_action
        if self.is_demo_env:
            info["demo_action"] = np.array(demo_actions)
        return (
            observation,
            total_reward,
            termination,
            truncation,
            info,
        )
