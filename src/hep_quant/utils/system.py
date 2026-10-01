"""System/environment introspection used for reproducibility reporting."""
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict


@dataclass
class SystemInfo:
    python_version: str
    platform: str
    processor: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    git_commit: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_system_info() -> SystemInfo:
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None
        gpu_name = None

    return SystemInfo(
        python_version=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        git_commit=get_git_commit(),
    )
