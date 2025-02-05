from pathlib import Path  # noqa
from pprint import pprint  # noqa

import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.onpolicy_workspace import OnPolicyWorkspace

    # root_dir = Path.cwd()

    cfg.num_train_envs = 0
    cfg.rlhf.use_rlhf = False

    workspace = OnPolicyWorkspace(cfg)

    # snapshot = root_dir / "snapshot.pt"
    snapshot = Path(
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250128160423/snapshots/latest_snapshot.pt" # LLM # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250127092312/snapshots/latest_snapshot.pt"  # Human Reward # noqa
        "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.1_sc-False_sc-temp1.0_sc-n5_seed1_20250129180805/snapshots/latest_snapshot.pt"  # RLHF Hybrid # noqa
    )
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot(path_to_snapshot_to_load=snapshot, override_cfg=True)
    workspace.agent.logging = False
    workspace._setup_training_functions()
    metrics = workspace.eval()

    del metrics["eval_rollout"]

    pprint(metrics)


if __name__ == "__main__":
    main()
