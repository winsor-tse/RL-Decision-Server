r"""Diagnose whether an NVIDIA GPU, driver, and PyTorch build match.

Run from the repository root with the same Python environment used for training:

    python -m Tests.pytorch_gpu_check

Or explicitly use this project's virtual environment on Windows:

    .\RL_venv\Scripts\python.exe -m Tests.pytorch_gpu_check

How the versions fit together
-----------------------------
1. ``nvidia-smi`` reports the installed driver and the newest CUDA runtime that
   the driver can support. It does not report the locally installed CUDA
   Toolkit version.
2. A prebuilt PyTorch CUDA wheel includes its own CUDA runtime. A separate CUDA
   Toolkit is normally unnecessary unless compiling PyTorch or custom CUDA code.
3. ``torch.version.cuda`` is the runtime bundled with the installed PyTorch
   wheel. It must not be newer than the maximum supported by the driver.
4. The wheel also needs kernels for the GPU's compute capability. The actual
   CUDA matrix multiplication at the end is the authoritative end-to-end test.

This project pins PyTorch 2.9.0. Official PyTorch 2.9.0 wheels are available
for CUDA 12.6, 12.8, and 13.0. The script selects the newest compatible wheel
from those choices and prints a command; it never changes the environment.

Official references:
- https://pytorch.org/get-started/locally/
- https://pytorch.org/get-started/previous-versions/
- https://docs.pytorch.org/docs/stable/generated/torch.cuda.get_arch_list.html
- https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_TORCH_VERSION = "2.9.0"
PYTORCH_29_CUDA_WHEELS = (
    ((13, 0), "cu130"),
    ((12, 8), "cu128"),
    ((12, 6), "cu126"),
)


@dataclass(frozen=True)
class NvidiaGpu:
    index: int
    name: str
    driver_version: str
    memory_mib: int | None
    compute_capability: tuple[int, int] | None


@dataclass(frozen=True)
class NvidiaSystem:
    executable: str
    maximum_cuda: tuple[int, int] | None
    gpus: tuple[NvidiaGpu, ...]


def version_tuple(value: str | None) -> tuple[int, int] | None:
    """Extract a major/minor pair from strings such as ``13.1``."""
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def format_version(value: tuple[int, int] | None) -> str:
    return "unknown" if value is None else f"{value[0]}.{value[1]}"


def find_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable

    if os.name == "nt":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidate = system_root / "System32" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def run_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def query_nvidia_system() -> NvidiaSystem | None:
    executable = find_nvidia_smi()
    if executable is None:
        return None

    summary = run_command([executable])
    maximum_cuda_match = re.search(
        r"CUDA Version:\s*(\d+\.\d+)",
        summary.stdout + summary.stderr,
    )
    maximum_cuda = version_tuple(
        maximum_cuda_match.group(1) if maximum_cuda_match else None
    )

    query_fields = "index,name,driver_version,memory.total,compute_cap"
    query = run_command(
        [
            executable,
            f"--query-gpu={query_fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    if query.returncode == 0:
        for row in query.stdout.splitlines():
            columns = [column.strip() for column in row.split(",")]
            if len(columns) != 5:
                continue
            try:
                index = int(columns[0])
            except ValueError:
                continue
            try:
                memory_mib = int(columns[3])
            except ValueError:
                memory_mib = None
            gpus.append(
                NvidiaGpu(
                    index=index,
                    name=columns[1],
                    driver_version=columns[2],
                    memory_mib=memory_mib,
                    compute_capability=version_tuple(columns[4]),
                )
            )

    return NvidiaSystem(
        executable=executable,
        maximum_cuda=maximum_cuda,
        gpus=tuple(gpus),
    )


def minimum_cuda_for_gpu(
    compute_capability: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Return the CUDA floor relevant to current NVIDIA architectures."""
    if compute_capability is None:
        return None
    if compute_capability >= (10, 0):
        # Blackwell compute capabilities 10.0 and 12.0 start at CUDA 12.8.
        return 12, 8
    if compute_capability >= (9, 0):
        return 11, 8
    if compute_capability >= (8, 9):
        return 11, 8
    if compute_capability >= (8, 0):
        return 11, 0
    if compute_capability >= (7, 5):
        return 10, 0
    return None


def recommend_cuda_wheel(
    nvidia: NvidiaSystem | None,
    device_index: int,
) -> tuple[tuple[int, int], str] | None:
    if nvidia is None or nvidia.maximum_cuda is None:
        return None

    selected_gpu = next(
        (gpu for gpu in nvidia.gpus if gpu.index == device_index),
        None,
    )
    if selected_gpu is None or selected_gpu.compute_capability is None:
        return None
    minimum_cuda = minimum_cuda_for_gpu(
        selected_gpu.compute_capability
    )
    for cuda_version, wheel_name in PYTORCH_29_CUDA_WHEELS:
        driver_is_new_enough = cuda_version <= nvidia.maximum_cuda
        gpu_is_new_enough = minimum_cuda is None or cuda_version >= minimum_cuda
        # CUDA 13 drops toolkit support for pre-Turing architectures.
        cuda_still_supports_gpu = not (
            selected_gpu.compute_capability < (7, 5)
            and cuda_version >= (13, 0)
        )
        if driver_is_new_enough and gpu_is_new_enough and cuda_still_supports_gpu:
            return cuda_version, wheel_name
    return None


def install_command(wheel_name: str) -> str:
    arguments = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        f"torch=={PROJECT_TORCH_VERSION}",
        "--index-url",
        f"https://download.pytorch.org/whl/{wheel_name}",
    ]
    return subprocess.list2cmdline(arguments)


def print_nvidia_report(
    nvidia: NvidiaSystem | None,
    device_index: int,
) -> None:
    print("\n[NVIDIA driver]")
    if nvidia is None:
        print("FAIL: nvidia-smi was not found. Install an NVIDIA driver first.")
        return

    print(f"nvidia-smi: {nvidia.executable}")
    print(f"Driver maximum CUDA: {format_version(nvidia.maximum_cuda)}")
    if not nvidia.gpus:
        print("WARNING: nvidia-smi ran but GPU details could not be queried.")
    for gpu in nvidia.gpus:
        selected = " (selected)" if gpu.index == device_index else ""
        memory = "unknown" if gpu.memory_mib is None else f"{gpu.memory_mib} MiB"
        print(
            f"GPU {gpu.index}{selected}: {gpu.name}; driver "
            f"{gpu.driver_version}; memory {memory}; compute capability "
            f"{format_version(gpu.compute_capability)}"
        )


def print_recommendation(
    nvidia: NvidiaSystem | None,
    device_index: int,
) -> None:
    print("\n[Recommended project wheel]")
    recommendation = recommend_cuda_wheel(nvidia, device_index)
    if recommendation is None:
        print(
            "No PyTorch 2.9 CUDA wheel can be selected confidently. Update the "
            "NVIDIA driver, verify the GPU compute capability, then use the "
            "official PyTorch selector."
        )
        return

    cuda_version, wheel_name = recommendation
    print(
        f"Recommended for this driver/GPU: PyTorch {PROJECT_TORCH_VERSION} "
        f"with CUDA {format_version(cuda_version)} ({wheel_name})."
    )
    print("Review, then run this command in the intended virtual environment:")
    print(install_command(wheel_name))


def print_pytorch_report(torch: Any, nvidia: NvidiaSystem | None) -> bool:
    print("\n[Installed PyTorch]")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Bundled CUDA runtime: {torch.version.cuda or 'none (CPU build)'}")
    print(f"cuDNN version: {torch.backends.cudnn.version() or 'not available'}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    bundled_cuda = version_tuple(torch.version.cuda)
    if bundled_cuda is None:
        print("FAIL: This is a CPU-only PyTorch build.")
        return False

    if nvidia and nvidia.maximum_cuda:
        if bundled_cuda <= nvidia.maximum_cuda:
            print(
                "PASS: The NVIDIA driver supports the PyTorch CUDA runtime "
                f"({format_version(bundled_cuda)} <= "
                f"{format_version(nvidia.maximum_cuda)})."
            )
        else:
            print(
                "FAIL: The PyTorch CUDA runtime is newer than the driver "
                "supports. Install a newer NVIDIA driver or an older PyTorch "
                "CUDA wheel."
            )
            return False

    if not torch.cuda.is_available():
        print(
            "FAIL: PyTorch contains CUDA, but CUDA initialization failed. "
            "Check the driver and restart the shell after driver updates."
        )
        return False
    return True


def run_cuda_test(torch: Any, device_index: int, matrix_size: int) -> bool:
    print("\n[PyTorch CUDA device]")
    device_count = torch.cuda.device_count()
    print(f"Visible CUDA devices: {device_count}")
    if device_index < 0 or device_index >= device_count:
        print(f"FAIL: CUDA device index {device_index} is not visible.")
        return False

    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    architecture = f"sm_{capability[0]}{capability[1]}"
    compiled_architectures = torch.cuda.get_arch_list()
    print(f"Device: {properties.name}")
    print(f"Compute capability: {capability[0]}.{capability[1]} ({architecture})")
    print(f"PyTorch compiled architectures: {compiled_architectures}")
    if architecture in compiled_architectures:
        print(f"PASS: The wheel explicitly includes {architecture} kernels.")
    else:
        print(
            f"WARNING: {architecture} is not listed explicitly. PTX forward "
            "compatibility may still work; the operation below is decisive."
        )

    print(f"Running {matrix_size}x{matrix_size} CUDA matrix multiplication...")
    try:
        left = torch.randn((matrix_size, matrix_size), device=device)
        right = torch.randn((matrix_size, matrix_size), device=device)
        result = left @ right
        torch.cuda.synchronize(device)
        checksum = float(result.abs().mean().item())
    except Exception as error:
        print(f"FAIL: CUDA operation raised {type(error).__name__}: {error}")
        return False

    print(f"PASS: CUDA operation completed; mean absolute value={checksum:.6f}")
    print(f"Allocated CUDA memory: {torch.cuda.memory_allocated(device) / 2**20:.1f} MiB")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to diagnose (default: 0).",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=1024,
        help="Matrix width used for the real CUDA workload (default: 1024).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix_size <= 0:
        raise ValueError("matrix-size must be greater than zero")

    print("NVIDIA / PyTorch compatibility check")
    print("=" * 40)
    nvidia = query_nvidia_system()
    print_nvidia_report(nvidia, args.device_index)

    try:
        import torch
    except ImportError as error:
        print(f"\n[Installed PyTorch]\nFAIL: PyTorch is not installed: {error}")
        print_recommendation(nvidia, args.device_index)
        return 1

    pytorch_ready = print_pytorch_report(torch, nvidia)
    cuda_test_passed = False
    if pytorch_ready:
        cuda_test_passed = run_cuda_test(
            torch,
            args.device_index,
            args.matrix_size,
        )

    print_recommendation(nvidia, args.device_index)
    print("\n[Result]")
    if pytorch_ready and cuda_test_passed:
        print("PASS: NVIDIA, the PyTorch CUDA build, and a real GPU operation match.")
        return 0

    print("FAIL: GPU acceleration is not ready; use the recommendation above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
