import torch

from robobase.replay_buffer.uniform_replay_buffer import INDICES


def get_query_fn(query_type):
    if query_type == "entropy":
        return collect_entropy_query
    elif query_type == "disagreement":
        return collect_disagreement_query
    elif query_type == "random":
        return collect_random_query
    elif query_type == "timestep":
        return collect_timestep_query
    else:
        raise ValueError(
            "Invalid query type. Please choose between 'entropy' or 'disagreement' or 'random'."
        )


def collect_entropy_query(batches):
    raise NotImplementedError("Entropy query is not implemented yet.")


def collect_disagreement_query(batches):
    raise NotImplementedError("Entropy query is not implemented yet.")


def collect_random_query(batches):
    return batches


def collect_timestep_query(batches):
    sorted_indices = torch.argsort(batches[INDICES], descending=False)
    sorted_batch = {}
    for key in batches:
        sorted_batch[key] = batches[key][sorted_indices]
    return sorted_batch
