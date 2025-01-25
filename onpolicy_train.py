from pathlib import Path

import hydra


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.onpolicy_workspace import OnPolicyWorkspace

    root_dir = Path.cwd()

    workspace = OnPolicyWorkspace(cfg)

    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
