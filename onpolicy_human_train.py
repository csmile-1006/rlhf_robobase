import pickle

import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.onpolicy_workspace import OnPolicyWorkspace

    workspace = OnPolicyWorkspace(cfg)
    resume_frames = cfg.num_resume_frames

    try:
        snapshot = workspace.work_dir / "snapshots" / f"{resume_frames}_snapshot.pt"
        assert snapshot.exists(), f"Snapshot {snapshot} does not exist"
        if snapshot.exists():
            print(f"resuming: {snapshot}")
            workspace.load_snapshot(snapshot, override_cfg=True)
        reward_model_snapshot = (
            workspace.work_dir
            / "reward_model_snapshots"
            / f"{resume_frames}_snapshot.pt"
        )
        assert (
            reward_model_snapshot.exists()
        ), f"Reward model snapshot {reward_model_snapshot} does not exist"
        if reward_model_snapshot.exists():
            print(f"resuming: {reward_model_snapshot}")
            workspace.load_reward_model_snapshot(reward_model_snapshot)
            workspace.reward_model.set_activated(True)
        human_feedback_files = pickle.load(
            open(
                workspace.work_dir
                / "human_feedback"
                / f"human_feedback_{cfg.num_human_iterations}.pkl",
                "rb",
            )
        )
        assert len(human_feedback_files) > 0, "Human feedback files are empty"
    except Exception as e:
        print(f"Error loading snapshot: {e}")
        raise e

    workspace.train()


if __name__ == "__main__":
    main()
