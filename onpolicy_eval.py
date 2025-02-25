from pathlib import Path
from pprint import pprint  # noqa
from datetime import datetime
import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.onpolicy_workspace import OnPolicyWorkspace

    # root_dir = Path.cwd()

    work_dir = f"./agym_exp/eval/{cfg.env.env_name}_{cfg.env.task_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}"  # noqa

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg.num_train_envs = 0
    cfg.rlhf.use_rlhf = False

    workspace = OnPolicyWorkspace(cfg, work_dir=work_dir)
    # snapshot = root_dir / "snapshot.pt"
    snapshot = Path(
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250128160423/snapshots/latest_snapshot.pt" # LLM # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_FeedingBaxter-v0/default_seed1_20250127092312/snapshots/latest_snapshot.pt"  # Human Reward # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.1_sc-False_sc-temp1.0_sc-n5_seed1_20250129180805/snapshots/latest_snapshot.pt"  # RLHF Hybrid # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda0.25_sc-False_sc-temp1.0_sc-n5_seed1_20250206221835/snapshots/latest_snapshot.pt" # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda1.0_sc-False_sc-temp1.0_sc-n5_seed1_20250210194740/snapshots/7872000_snapshot.pt" # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda1.0_sc-False_sc-temp1.0_sc-n5_seed1_20250210194740/snapshots/latest_snapshot.pt" # noqa
        # "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo_rlhf_hybrid/agym_FeedingBaxter-v0/rlhf_hybrid_gemini_numf500_numq50_update30_pretrain30_comp-major_column_reward-reset-True_agent-reset-False_lambda1.0_sc-False_sc-temp1.0_sc-n5_seed1_20250210194740/snapshots/latest_snapshot.pt" # noqa
        "/home/changyeon/ICML2025/workspace/rlhf_robobase/agym_exp/state_ppo/agym_drinking_sawyer/gt_reward_seed1_/snapshots/latest_snapshot.pt"
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
