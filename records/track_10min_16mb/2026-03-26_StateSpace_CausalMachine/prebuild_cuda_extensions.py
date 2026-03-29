#!/usr/bin/env python3
import importlib.util
import os
import sys
from pathlib import Path


def load_train_module():
    train_path = Path(__file__).with_name("train_gpt.py").resolve()
    spec = importlib.util.spec_from_file_location("causal_machine_train_gpt", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # Prebuild must be allowed to compile missing extensions even when the
    # timed training path later requires prebuilt artifacts to already exist.
    os.environ["CAUSAL_MACHINE_REQUIRE_PREBUILT_EXTENSIONS"] = "0"
    module = load_train_module()
    print("prebuilding causal_machine_scan_cuda...")
    module.load_causal_machine_scan_cuda()
    print("causal_machine_scan_cuda ready")
    print("prebuilding causal_machine_latent_scan_cuda...")
    module.load_causal_machine_latent_scan_cuda()
    print("causal_machine_latent_scan_cuda ready")
    print("prebuilding muon_cuda...")
    module.load_muon_cuda()
    print("muon_cuda ready")


if __name__ == "__main__":
    main()
