"""Print the Python, PyTorch, and GPU details of the current environment."""

import platform
import sys

import torch


def main() -> None:
    """Print Python version, torch version, CUDA availability, GPU name, and VRAM."""
    print(f"Python:          {platform.python_version()} ({sys.executable})")
    print(f"torch:           {torch.__version__}")
    print(f"torch CUDA:      {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:  {cuda_available}")
    if not cuda_available:
        print("GPU:             none detected")
        return
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:             {props.name}")
    print(f"VRAM:            {props.total_memory / 1024**3:.2f} GiB")
    print(f"Compute cap.:    {props.major}.{props.minor}")
    # Tiny fp16 matmul to confirm kernels actually run on this GPU.
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    _ = (x @ x).sum().item()
    print("fp16 matmul:     OK")


if __name__ == "__main__":
    main()
