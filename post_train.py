from pathlib import Path

import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.workspace import Workspace

    snapshot = (
        Path(cfg.snapshot_dir)
        / f"{cfg.env.env_name}_{cfg.env.task_name}"
        / "latest_snapshot.pt"
    )
    assert snapshot.exists(), f"Snapshot {snapshot} does not exist"
    print(f"resuming: {snapshot}")
    workspace = Workspace(cfg)
    workspace.load_snapshot(path_to_snapshot_to_load=snapshot, override_cfg=True)
    workspace.train()


if __name__ == "__main__":
    main()
