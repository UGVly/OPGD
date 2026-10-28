"""Short multi-process CUDA GEMM smoke/stress test for the OPGD host."""

from __future__ import annotations

import os
import time

import torch


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    matrix_size = int(os.environ.get("OPGD_GPU_STRESS_MATRIX_SIZE", "16384"))
    duration_seconds = float(os.environ.get("OPGD_GPU_STRESS_SECONDS", "20"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260824 + local_rank)

    left = torch.randn(
        (matrix_size, matrix_size), device=device, dtype=torch.bfloat16
    )
    right = torch.randn(
        (matrix_size, matrix_size), device=device, dtype=torch.bfloat16
    )
    torch.cuda.synchronize(device)

    iterations = 0
    started_at = time.monotonic()
    product = None
    while time.monotonic() - started_at < duration_seconds:
        product = torch.mm(left, right)
        # Keep the requested duration tied to completed GPU work rather than
        # the amount of asynchronous work queued by the Python process.
        torch.cuda.synchronize(device)
        iterations += 1
    elapsed_seconds = time.monotonic() - started_at
    checksum = float(product[0, 0].float().cpu()) if product is not None else 0.0
    allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(
        f"gpu={local_rank} iterations={iterations} elapsed={elapsed_seconds:.2f}s "
        f"max_allocated={allocated_gib:.2f}GiB checksum={checksum}",
        flush=True,
    )


if __name__ == "__main__":
    main()
