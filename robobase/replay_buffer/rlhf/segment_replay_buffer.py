"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""

from __future__ import annotations
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Type
from multiprocessing import Value
from collections import defaultdict
import logging
from typing_extensions import override

import torch
from gymnasium import spaces
from natsort import natsort
import numpy as np

from robobase.replay_buffer.replay_buffer import (
    ReplayBuffer,
    ReplayElement,
)

from robobase.replay_buffer.uniform_replay_buffer import (
    save_episode,
    load_episode,
    ACTION,
    INDICES,
    IS_FIRST,
)

LABEL = "label"


def episode_len(episode):
    # subtract -1 because the last final transition
    return next(iter(episode.values())).shape[1] - 1


class SegmentReplayBuffer(ReplayBuffer):
    """A simple Replay Buffer for training reward model from preferences.

    Stores transitions, state, action, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations or sequences of action, storing
    the states is inefficient. This class writes observations and constructs the
    stacked states at sample time.

    NOTE: Layout of the uniform replay with last empty elements (=)
    - Observations: o_0, o_1, o_2, ..., o_{N-1}, O_N
    - Actions:      a_0, a_1, a_2, ..., a_{N-1},  =
    - Terminations:  F ,  F ,  F , ...,  T ,      =
    - Truncations:   F ,  F ,  F , ..., T/F,      =
    - Infos:        i_0, i_1, i_2, ..., i_{N-1},  =
    Agent gives a_t given o_t, and Env gives r_{t+1} and i_{t+1} given a_t
    Because model-free RL only needs last observations, we can have empty elements.
    We have dedicated add_final method that ignores (=) values by using np.empty

    Alternatively, user could store episodes in sequential layout.
    NOTE: Layout of the sequential replay with first default ('') elements
    - Observations: o_0, o_1, o_2, ..., o_{N-1}, O_N
    - (Prev)Actions:'0', a_0, a_1, ..., a_{N-2}, a_{N-1}
    - Terminations: 'F',  F ,  F , ...,  F,       T
    - Truncations:  'F',  F ,  F , ...,  F,       T
    Agent gives a_t given o_t, and Env gives r_{t+1} and i_{t+1} given a_t
    Because model-based RL often depends on prev actions for learning world models,
    each transition has prev_action as an element unlike UniformReplay

    Across RooBase, observation is in the form of dictionary. Replay buffer stores
    observation key by key. I.e, instead of transition["obs"] = obs,
    transition[obs_key] = observation[obs_key] for obs_key in obs.

    For each transition, replay buffer stores extra elements specified by
    extra_replay_elements arguments. Typically, these are important information
    contained in the info given by environment. But they can also be from other sources,
    such as the flag "demo" marking whether the transition is part of a successful
    episode.

    Attributes:
      _add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
    """

    def __init__(
        self,
        batch_size: int = 32,
        replay_capacity: int = int(1e6),
        action_shape: tuple = (),
        action_dtype: Type[np.dtype] = np.float32,
        observation_elements: spaces.Dict = None,
        extra_replay_elements: spaces.Dict = None,
        save_dir: str = None,
        purge_replay_on_shutdown: bool = True,
        preprocessing_fn: list[Callable[[list[spaces.Dict]], list[spaces.Dict]]] = None,
        preprocess_every_sample: bool = False,
        num_workers: int = 0,
        fetch_every: int = 100,
        sequential: bool = False,
        transition_seq_len: int = 50,
        num_labels: int = 1,
    ):
        """Initializes OutOfGraphReplayBuffer.

        Args:
          batch_size (int):
          replay_capacity (int): number of transitions to keep in memory.
          action_shape (tuple of ints): the shape for the action vector.
            Empty tuple means the action is a scalar.
          action_dtype (np.dtype): type of elements in the action.
          observation_elements (dict): a dict of spaces defining the type and size of
            the observation contents that will be stored and returned.
          extra_replay_elements (dict): a dict of spaces defining the type and size of
            the extra transition information that will be stored and returned.
          preprocessing_fn (list of callables): list of processing functions which
            process observations before adding to replay buffer
          preprocess_every_sample (bool): if False preprocessing will be performed on
            setting, else if True preprocessing will be performed on getting.
          num_workers (int): The number of workers used to sample from this replay
            buffer.
          fetch_every (int): The number of samples returned from replay buffer before
            new episodes are fetch from disk.
          sequential (bool): whether the replay buffer should store episodes in
            sequential format.
          transition_seq_len (int): the length of the transition sequence to sample
            from sequential replay buffer. Only applicable if sequential is true.
          num_labels (int): The number of human labels to store in the replay buffer.

        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """
        if observation_elements is None:
            observation_elements = {}
        if extra_replay_elements is None:
            extra_replay_elements = {}

        # Check that all observations have a time component
        time_dims = []
        new_observation_elements = {}
        for name, space in observation_elements.items():
            if len(space.shape) <= 1:
                raise ValueError(
                    f"Expected observation space {name} to have >= 1 dimensions."
                    "Observations spaces should have a temporal dimension."
                )
            time_dims.append(space.shape[0])
            # Now, temporal aspect must be same with the transition_seq_len.
            new_observation_elements[name] = spaces.Box(
                np.stack([space.low[0] for _ in range(transition_seq_len)]),
                np.stack([space.high[0] for _ in range(transition_seq_len)]),
                shape=(transition_seq_len,) + space.shape[1:],
                dtype=space.dtype,
            )
        observation_elements = new_observation_elements

        # Now remove temporal aspect from action, as we won't be storing them.
        # action_seq_len = action_shape[0]
        action_seq_len = transition_seq_len
        new_action_shape = (action_seq_len,) + action_shape[1:]

        if not np.all(time_dims[0] == np.array(time_dims)):
            raise ValueError("Expected all observation spaces to have same temporal dimension.")
        frame_stack = time_dims[0]

        if sequential and replay_capacity < 1 + transition_seq_len:
            raise ValueError("There is not enough capacity to cover nstep and transition_seq_len.")

        if sequential or action_seq_len != transition_seq_len:
            raise ValueError(
                "Segment replay buffer must have non-sequential replay and action_seq_len == transition_seq_len."
            )

        self._tmpdir = None
        if save_dir is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            save_dir = self._tmpdir.name
        self._replay_dir = Path(save_dir)
        self._purge_replay_on_shutdown = purge_replay_on_shutdown
        logging.info("\t saving to disk: %s", self._replay_dir)
        os.makedirs(save_dir, exist_ok=True)

        self._action_shape = new_action_shape
        self._action_dtype = action_dtype
        self._frame_stacks = frame_stack
        self._action_seq_len = action_seq_len
        self._transition_seq_len = transition_seq_len
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._sequential = sequential
        self._num_labels = num_labels

        self.observation_elements = observation_elements
        self.extra_replay_elements = extra_replay_elements

        self._storage_signature, self._obs_signature = self.get_storage_signature()
        self._add_count = Value("i", 0)
        self._replay_capacity = replay_capacity

        self._preprocessing_fn = preprocessing_fn
        self._preprocess_every_sample = preprocess_every_sample

        # =======
        self._episode_files = []  # list of episode file path
        self._episodes = {}  # Key: eps_file_path, value: episode
        # Key: global_idx. Global_idx refers to the index in the entire replay buffer.
        # Value: (episode_file_path, transition_idx) where transition_idx
        # refers to the index of transition in the episode
        self._global_idxs_to_episode_and_transition_idx = defaultdict(list)
        self._current_episode = defaultdict(list)
        self._num_episodes = 0
        self._num_transitions = 0

        # data loader
        self._size = 0
        self._max_size_per_worker = replay_capacity // max(1, num_workers)
        self._num_workers = num_workers
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = self._fetch_every
        save_snapshot = False
        self._save_snapshot = save_snapshot

        logging.info(
            "Creating a %s replay memory with the following parameters:",
            self.__class__.__name__,
        )
        logging.info("\t frame_stack: %d", self._frame_stacks)
        logging.info("\t replay_capacity: %d", self._replay_capacity)
        logging.info("\t batch_size: %d", self._batch_size)
        self._is_first = True

    @property
    def frame_stack(self):
        return self._frame_stacks

    @property
    def action_seq(self):
        return self._action_seq_len

    @property
    def invalid_range(self):
        return np.array(self._invalid_range)

    @invalid_range.setter
    def invalid_range(self, value: np.array):
        self._invalid_range = value.tolist()

    @property
    def replay_capacity(self):
        return self._replay_capacity

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sequential(self):
        return self._sequential

    def get_storage_signature(
        self,
    ) -> tuple[dict[str, ReplayElement], dict[str, ReplayElement]]:
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          dict of ReplayElements defining the type of the contents stored.
        """
        storage_elements, obs_elements = {}, {}
        storage_elements.update({LABEL: ReplayElement(LABEL, (self._num_labels,), np.int8)})
        for i in range(2):
            storage_elements.update(
                {
                    f"seg{i}_{ACTION}": ReplayElement(f"seg{i}_{ACTION}", self._action_shape, self._action_dtype),
                }
            )

            for obs_name, space in self.observation_elements.items():
                obs_elements[f"seg{i}_{obs_name}"] = ReplayElement(f"seg{i}_{obs_name}", space.shape, space.dtype)
            storage_elements.update(obs_elements)

            for element_name, space in self.extra_replay_elements.items():
                storage_elements[f"seg{i}_{element_name}"] = ReplayElement(
                    f"seg{i}_{element_name}", space.shape, space.dtype
                )

        return storage_elements, obs_elements

    def add_segment(self, segment_0, segment_1, label, **kwargs):
        """Adds a pair of segments to the replay memory."""
        transition = {}
        for idx, seg in enumerate([segment_0, segment_1]):
            target_keys = [key for key in self._storage_signature.keys() if key.startswith(f"seg{idx}_")]
            for key in target_keys:
                transition[key] = seg[key.split(f"seg{idx}_")[1]]

        transition[LABEL] = label
        # # transition[ACTION] = seg[ACTION]
        # # Info and observation are stored key by key
        # transition.update(kwargs)
        # # transition.update(seg)
        # observation = {}
        # for obs_key in seg.keys():
        # # for obs_key in self._obs_signature.keys():
        #     observation[f"seg{idx}_{obs_key}"] = seg[obs_key]
        # transition.update(observation)

        # Check transition shape is correct
        self._check_add_types(transition, self._storage_signature)

        # Add transition
        self._add(transition)
        self._add_count.value += self._transition_seq_len

        # Current episode has terminated, store episode and reset current episode
        episode = dict()
        for k, v in self._current_episode.items():
            episode[k] = np.array(v, self._storage_signature[k].type)
        self._current_episode = defaultdict(list)
        self._store_episode(episode)

    @override
    def add(
        self,
        observation: dict,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncated: bool,
        **kwargs,
    ):
        raise NotImplementedError

    @override
    def add_final(self, final_observation: dict):
        raise NotImplementedError

    def _store_episode(self, episode):
        # if self._sequential:
        #     # If sequential, convert the episode layout
        #     episode = self.convert_episode_layout(episode)

        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        global_idx = self.add_count - eps_len
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}_{global_idx}.npz"
        save_episode(episode, self._replay_dir / eps_fn)

        if self._is_first:
            # A special case for first insert. So that the user can have arbitrary
            # num_workers, we replicate the first episode across all workers.
            # This means that all workers have some data to start.
            self._is_first = False
            for worker_id in range(1, self._num_workers):
                eps_fn = f"{worker_id}{ts}_{eps_idx+worker_id}_{eps_len}_{global_idx}.npz"
                save_episode(episode, self._replay_dir / eps_fn)

    def _add(self, transition: dict):
        """Internal add method to add to the storage arrays.

        Args:
          transition: All the elements in a transition.
        """
        if self._preprocessing_fn is not None and not self._preprocess_every_sample:
            for fn in self._preprocessing_fn:
                transition = fn([transition])[0]

        for name, data in transition.items():
            self._current_episode[name].append(data)

    def _check_add_types(self, transition: dict, signature: dict):
        """Checks if args passed to the add method match those of the storage.

        Args:
          transition: The current transition to add to replay buffer

        Raises:
          ValueError: If transition have wrong shape or dtype.
        """

        if (len(transition)) != len(signature):
            expected = str(natsort.natsorted(list(signature.keys())))
            actual = str(natsort.natsorted(list(transition.keys())))
            error_list = "\nlist of expected:\n{}\nlist of actual:\n{}".format(expected, actual)
            raise ValueError(
                "Add expects {} elements, received {}.".format(len(signature), len(transition)) + error_list
            )

        for name, store_element in signature.items():
            arg_element = transition[store_element.name]
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO: This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError("arg {} has shape {}, expected {}".format(name, arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self._add_count.value == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self._add_count.value >= self._replay_capacity

    def __len__(self):
        return np.minimum(self._add_count.value, self._replay_capacity)

    @property
    def add_count(self):
        return self._add_count.value

    @add_count.setter
    def add_count(self, count: int):
        self._add_count.value = count

    def shutdown(self):
        if self._purge_replay_on_shutdown:
            logging.info("Clearing disk replay buffer.")
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
            for f in self._replay_dir.glob(".npz"):
                f.unlink(missing_ok=True)

    ### Below are the Dataset functions ###

    def _sample_episode(self):
        eps_fn = np.random.choice(self._episode_files)
        _, _, global_index = [int(x) for x in eps_fn.stem.split("_")[1:]]
        return self._episodes[eps_fn], global_index

    def _load_episode_into_worker(self, eps_fn: Path, global_idx: int):
        # Load episode into memory
        try:
            episode = load_episode(eps_fn)
        except Exception:
            return False

        # Remove earliest episode if buffer is full.
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size_per_worker:
            early_eps_files = self._episode_files.pop(0)
            early_eps = self._episodes.pop(early_eps_files)
            self._size -= episode_len(early_eps)
            keys = list(self._global_idxs_to_episode_and_transition_idx.keys())
            for k in keys[: episode_len(early_eps)]:
                del self._global_idxs_to_episode_and_transition_idx[k]
            early_eps_files.unlink(missing_ok=True)

        self._episode_files.append(eps_fn)
        self._episode_files.sort()  # NOTE: eps_fn starts with created timestamp.
        # so after sort, earliest episode appears first.
        self._episodes[eps_fn] = episode
        global_idxs = np.arange(global_idx, global_idx + eps_len)
        global_idxs_wrapped = (global_idxs % self.replay_capacity).tolist()
        self._global_idxs_to_episode_and_transition_idx.update(
            {global_i: [eps_fn, ep_transition_i] for ep_transition_i, global_i in enumerate(global_idxs_wrapped)}
        )
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0

        try:
            worker_id = torch.utils.data.get_worker_info().id
        except Exception:
            worker_id = 0

        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len, global_idx = [int(x) for x in eps_fn.stem.split("_")[1:]]

            # Each worker should only contain its relevant indices.
            if self._num_workers > 0 and eps_idx % self._num_workers != worker_id:
                continue

            # If episode is already loaded, skip
            if eps_fn in self._episodes.keys():
                break

            # Check max_size per worker
            if fetched_size + eps_len > self._max_size_per_worker:
                break
            fetched_size += eps_len

            if not self._load_episode_into_worker(eps_fn, global_idx):
                break

    def _flatten_episodes(self, episodes: list[dict]):
        for ep in episodes:
            is_first = np.zeros(episode_len(ep) + 1, np.int8)
            is_first[0] = 1
            ep[IS_FIRST] = is_first
        flattened = dict(episodes[0])
        for ep in episodes[1:]:
            for k, v in ep.items():
                flattened[k] = np.concatenate([flattened[k], v], 0)
        return flattened

    def _sample_non_sequential(self, global_index=None):
        # Sample transition index
        if global_index is None:
            # NOTE: here global index is the index of the start of episode.
            episode, global_index = self._sample_episode()
            # each episode has only one segment
            idx = 0
            # min_idx, max_idx = 0, np.maximum(episode_len(episode), 1)
            # idx = np.random.randint(min_idx, max_idx)

            # global index of the transition = index of episode_start + transition_idx
            global_index += idx

        else:
            if global_index not in self._global_idxs_to_episode_and_transition_idx:
                # This worker does not have this sample
                return None

            (
                episode_fn,
                transition_idx,
            ) = self._global_idxs_to_episode_and_transition_idx[global_index]
            episode = self._episodes[episode_fn]
            idx = transition_idx

        # Construct replay sample from sampled transition index
        # ep_len = episode_len(episode)
        # next_idx = idx + 1
        # If next_idx > eps_len, next_idx will point to final_obs
        replay_sample = {}

        # Handle frame stacking, calculate observation indices.
        # obs_start_idx = (idx - self._frame_stacks) + 1
        # next_obs_start_idx = (next_idx - self._frame_stacks) + 1
        # Obs_idxs contains indices of all frames, considering frame stacking.
        # - Turn all negative idxs to 0
        # obs_idxs = list(map(lambda x: np.clip(x, 0, ep_len), range(obs_start_idx, idx + 1)))
        # next_obs_idxs = list(
        #     map(
        #         lambda x: np.clip(x, 0, ep_len),
        #         range(next_obs_start_idx, next_idx + 1),
        #     )
        # )

        # Add observation frames into sample
        for name in self._obs_signature.keys():
            replay_sample[name] = episode[name][idx]
            # Retrieve tp1 observations
            # replay_sample[name + "_tp1"] = episode[name][next_obs_idxs]

        # Handle action sequences
        # action_start_idx = idx
        # action_end_idx = min(idx + self._action_seq_len, ep_len)
        # # - action_idxs contains indices of all action, considering action sequences.
        # action_idxs = list(range(action_start_idx, action_end_idx))
        # action_seq = episode[ACTION][action_idxs]
        # # - Pad zeros to the end if action_sequences exceeds eps_len
        # if len(action_seq) < self._action_seq_len:
        #     num_action_to_pad = self._action_seq_len - len(action_seq)
        #     action_seq = np.concatenate(
        #         [
        #             action_seq,
        #             np.zeros((num_action_to_pad, *action_seq.shape[1:]), dtype=np.float32),
        #         ],
        #         axis=0,
        #     )
        # replay_sample[ACTION] = action_seq

        # Add the rest
        replay_sample.update(
            {
                # TERMINAL: episode[TERMINAL][next_idx - 1],
                # TRUNCATED: episode[TRUNCATED][next_idx - 1],
                INDICES: global_index,
            }
        )
        # Add remaining (extra) items
        for name in self._storage_signature.keys():
            if name not in replay_sample:
                replay_sample[name] = episode[name][idx]

        return replay_sample

    def sample_single(self, global_index: int = None) -> dict:
        """Sample a single transition from replay buffer.

        Args:
            global_index (int, optional): If provided, will return the transition at
                                          the global_idx.
                                          If not, will sample randomly.

        Returns:
            dict: replay sample.
        """
        # index here is the "global" index of a flattened sample
        self._try_fetch()

        self._samples_since_last_fetch += 1

        return self._sample_non_sequential(global_index)

    @override
    def sample(self, batch_size: int | None = None, indices: list[int] = None) -> np.ndarray:
        batch_size = self._batch_size if batch_size is None else batch_size
        if indices is not None and len(indices) != batch_size:
            raise ValueError(f"indices was of size {len(indices)}, but batch size was {batch_size}")
        if indices is None:
            indices = [None] * batch_size

        samples = [self.sample_single(indices[i]) for i in range(batch_size)]
        batch = {}
        for k in samples[0].keys():
            batch[k] = np.stack([sample[k] for sample in samples])
        return batch

    def __iter__(self):
        while True:
            yield self.sample_single()
