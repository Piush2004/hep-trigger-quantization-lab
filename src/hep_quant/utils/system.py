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
    git_dirty: bool | None = None  # True if tracked files had uncommitted changes when the run started
    packages: dict | None = None  # versions of the libraries that determine numerical results
    environment_hash: str | None = None  # sha256 of the sorted "name==version" list above

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


def get_git_dirty() -> bool | None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


_TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "torch", "hls4ml", "qkeras", "h5py", "matplotlib", "PyYAML"]


def get_package_versions() -> dict:
    """Installed versions of the numerically relevant packages; None for packages that are not installed."""
    from importlib import metadata

    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_hash(packages: dict) -> str:
    import hashlib

    return hashlib.sha256(";".join(f"{k}=={v}" for k, v in sorted(packages.items())).encode()).hexdigest()


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
        git_dirty=get_git_dirty(),
        packages=(pk := get_package_versions()),
        environment_hash=environment_hash(pk),
    )
