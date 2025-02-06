from pathlib import Path  # noqa
from pprint import pprint  # noqa
from datetime import datetime
import pickle

import hydra
import matplotlib.pyplot as plt


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.onpolicy_workspace import OnPolicyWorkspace

    # root_dir = Path.cwd()

    cfg.num_train_envs = 0
    cfg.rlhf.use_rlhf = False
    cfg.env.query_keys = cfg.env.query_keys.split(",")
    work_dir = f"/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/eval/{cfg.env.env_name}_{cfg.env.task_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}"  # noqa

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    workspace = OnPolicyWorkspace(cfg, work_dir=work_dir)
    # snapshot = root_dir / "snapshot.pt"
    snapshot = Path(
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250128160423/snapshots/latest_snapshot.pt" # LLM # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250127092312/snapshots/latest_snapshot.pt"  # Human Reward # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.1_sc-False_sc-temp1.0_sc-n5_seed1_20250129180805/snapshots/latest_snapshot.pt"  # RLHF Hybrid # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.1_sc-False_sc-temp1.0_sc-n5_seed1_20250129180805/snapshots/latest_snapshot.pt"  # RLHF Hybrid # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.5_sc-False_sc-temp1.0_sc-n5_seed1_20250204211418/snapshots/latest_snapshot.pt" # noqa
        "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.25_sc-False_sc-temp1.0_sc-n5_seed1_20250205214203/snapshots/2304000_snapshot.pt"  # noqa
    )
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot(path_to_snapshot_to_load=snapshot, override_cfg=True)
    workspace.agent.logging = False
    workspace._setup_training_functions()

    mode = "replay"
    if mode == "replay":
        base_path = Path(
            "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/eval/agym_FeedingBaxter-v0/20250206162548"
        )
        randomness_values = pickle.load(
            (base_path / "randomness_values.pkl").open("rb")
        )
        actions = pickle.load(
            (base_path / "eval_episode_actions" / "total_eval_actions.pkl").open("rb")
        )

        metrics = workspace.replay(randomness_values, actions)
        del metrics["eval_rollout"]
    else:
        metrics = workspace.eval()
        del metrics["eval_rollout"]

        randomness_values = workspace.eval_env.randomness_values
        pickle.dump(
            randomness_values, (workspace.work_dir / "randomness_values.pkl").open("wb")
        )

        total_actions = metrics.pop("eval_actions")
        eval_episode_actions_dir = workspace.work_dir / "eval_episode_actions"
        eval_episode_actions_dir.mkdir(parents=True, exist_ok=True)
        total_eval_actions_file = eval_episode_actions_dir / "total_eval_actions.pkl"
        pickle.dump(total_actions, total_eval_actions_file.open("wb"))
        for idx, ep_actions in enumerate(total_actions):
            with open(eval_episode_actions_dir / f"eval_actions_{idx}.txt", "w") as f:
                for action in ep_actions:
                    f.write(str(action) + "\n")

            for act_dim in range(ep_actions.shape[-1]):
                plt.plot(ep_actions[..., act_dim].squeeze())
                plt.title(f"Action dim {act_dim}")
                plt.savefig(
                    eval_episode_actions_dir
                    / f"eval_actions_{idx}_actdim{act_dim}.png",
                    dpi=300,
                )
                plt.close()

    pprint(metrics)


if __name__ == "__main__":
    main()
