from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset import Dataset

# Subset flags (CLI) -> directory names under ``data/benchmark-opencl/<dir>/``.
SUBSET_TO_BENCHMARK_DIR = {
    "L1": "L1",
    "L2": "L2",
    "L3": "L3",
    "level1": "L1",
    "level2": "L2",
    "level3": "L3",
    "sol-level1": "sol-level1",
    "sol-level2": "sol-level2",
}

RUN_FOLDER_PARENT_TO_BENCHMARK_DIR = {
    "level1": "L1",
    "level2": "L2",
    "level3": "L3",
    "L1": "L1",
    "L2": "L2",
    "L3": "L3",
    "sol-level1": "sol-level1",
    "sol-level2": "sol-level2",
}


def subset_flag_to_benchmark_dir(subset: str | None) -> str | None:
    if subset is None:
        return None
    if subset not in SUBSET_TO_BENCHMARK_DIR:
        raise ValueError(f"Invalid kernelbench-opencl subset: {subset}")
    return SUBSET_TO_BENCHMARK_DIR[subset]


def run_output_parent_to_benchmark_dir(parent: str) -> str:
    return RUN_FOLDER_PARENT_TO_BENCHMARK_DIR.get(parent, parent)


def default_benchmark_opencl_root() -> Path:
    """Primary benchmark tree for OpenCL kernels under ``data/benchmark-opencl``."""
    return Path(__file__).resolve().parents[1] / "data" / "benchmark-opencl"


def kernelbench_opencl_port_root() -> Path:
    """Hand-curated OpenCL layout: ``data/kernelbench-opencl/<subset>/<problem>/``."""
    return Path(__file__).resolve().parents[1] / "data" / "kernelbench-opencl"


class KernelBenchOpenCLDataset(Dataset):
    """
    Dataset over curated OpenCL kernel artifacts targeting Qualcomm Adreno GPUs.

    Default layout:

      data/benchmark-opencl/<L1|L2|L3>/<problem_name>/
        - driver.c          (host-side OpenCL driver code)
        - kernel.cl         (OpenCL C kernel source)
        - reference.py      (optional, reference implementation for validation)

    Each problem entry provides:
      - id: unique problem identifier (e.g., "L1/001_vector_add")
      - problem_name: directory name
      - problem_num: numeric prefix
      - level: subset level
      - driver_c_fp: path to host driver (.c)
      - kernel_cl_fp: path to kernel source (.cl)
      - reference_py_fp: path to reference implementation (if exists)
    """

    def __init__(
        self,
        level_str: str | None = None,
        problem_numbers: list[int] | None = None,
        start: int | None = None,
        end: int | None = None,
        root_dir: str | Path | None = None,
    ):
        root = Path(root_dir) if root_dir is not None else default_benchmark_opencl_root()
        super().__init__(root)
        if level_str is not None and level_str not in SUBSET_TO_BENCHMARK_DIR:
            raise ValueError(f"Invalid level subset: {level_str}")
        self.level_str = level_str
        self._load_dataset(problem_numbers=problem_numbers, start=start, end=end)

    def _scan_level_dir(
        self,
        level_dir: Path,
        id_prefix: str,
        problem_numbers: list[int] | None,
        start: int | None,
        end: int | None,
    ) -> None:
        if not level_dir.is_dir():
            return
        for problem_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
            try:
                num = int(problem_dir.name.split("_", 1)[0])
            except Exception:
                continue

            if problem_numbers is not None and num not in problem_numbers:
                continue
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue

            driver_c = problem_dir / "driver.c"
            kernel_cl = problem_dir / "kernel.cl"

            # Also support alternative naming
            if not kernel_cl.exists():
                kernel_cl = problem_dir / "kernel.opencl"
            if not driver_c.exists():
                driver_c = problem_dir / "main.c"

            if not kernel_cl.exists():
                continue

            reference_py = problem_dir / "reference.py"
            entry: dict[str, Any] = {
                "id": f"{id_prefix}/{problem_dir.name}",
                "problem_name": problem_dir.name,
                "problem_num": num,
                "level": id_prefix,
                "driver_c_fp": str(driver_c) if driver_c.exists() else None,
                "kernel_cl_fp": str(kernel_cl),
            }
            if reference_py.exists():
                entry["reference_py_fp"] = str(reference_py)
            self.data.append(entry)

    def _load_dataset(
        self,
        problem_numbers: list[int] | None,
        start: int | None,
        end: int | None,
    ) -> None:
        if not self.data_dir.exists():
            if self.level_str in {"sol-level1", "sol-level2"} and kernelbench_opencl_port_root().is_dir():
                pass
            else:
                extra = ""
                if self.level_str in {"sol-level1", "sol-level2"}:
                    extra = (
                        f"; for --subset {self.level_str} without data/benchmark-opencl, create directory "
                        f"{kernelbench_opencl_port_root()} (e.g. …/{self.level_str}/001_…/)"
                    )
                raise FileNotFoundError(
                    f"Dataset directory {self.data_dir} not found{extra}"
                )

        if self.level_str is None:
            for bench_level in ["L1", "L2", "L3"]:
                self._scan_level_dir(
                    self.data_dir / bench_level,
                    bench_level,
                    problem_numbers,
                    start,
                    end,
                )
        elif self.level_str in {"sol-level1", "sol-level2"}:
            sol_subset = self.level_str
            fallback_bench_level = "L1" if sol_subset == "sol-level1" else "L2"
            port_sol = kernelbench_opencl_port_root() / sol_subset
            if port_sol.is_dir() and any(p.is_dir() for p in port_sol.iterdir()):
                self._scan_level_dir(
                    port_sol, sol_subset, problem_numbers, start, end
                )
            if not self.data:
                dedicated = self.data_dir / sol_subset
                if dedicated.is_dir() and any(p.is_dir() for p in dedicated.iterdir()):
                    self._scan_level_dir(
                        dedicated, sol_subset, problem_numbers, start, end
                    )
            if not self.data:
                self._scan_level_dir(
                    self.data_dir / fallback_bench_level, sol_subset, problem_numbers, start, end
                )
        else:
            bench_level = subset_flag_to_benchmark_dir(self.level_str)
            self._scan_level_dir(
                self.data_dir / bench_level,
                bench_level,
                problem_numbers,
                start,
                end,
            )

        self.data.sort(key=lambda x: x["id"])
