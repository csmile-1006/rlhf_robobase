import torch

from robobase.replay_buffer.uniform_replay_buffer import INDICES


def get_query_fn(query_type):
    if query_type == "random":
        return collect_random_query
    elif query_type == "timestep":
        return collect_timestep_query
    else:
        raise ValueError(
            "Invalid query type. Please choose between 'random' or 'timestep'."
        )


def collect_random_query(batches):
    return batches


def collect_timestep_query(batches):
    sorted_indices = torch.argsort(batches[INDICES], descending=False)
    sorted_batch = {}
    for key in batches:
        sorted_batch[key] = batches[key][sorted_indices]
    return sorted_batch
