"""
OpenCL kernel generation agent — translates PyTorch reference.py into
driver.c + kernel.cl for Qualcomm Adreno GPUs.

Two-step validation:
  1. Compile driver.c with a dummy (no-op) kernel → must print "failed"
     (proves the driver's verification logic actually catches bad kernels).
  2. Compile driver.c with the real kernel.cl → must print "passed".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from .feedback import FeedbackAgent, Feedback, FeedbackConfig
from .prompt.kgen_opencl import build_system_prompt, build_user_prompt
from .utils import (
    FeedbackError,
    compile_and_run_opencl,
    NamedTimer,
    extract_code_from_response,
    write_code_to_file,
)


def _extract_kernel_name_from_driver(driver_code: str) -> Optional[str]:
    """Extract the kernel name from clCreateKernel(program, "name", ...)."""
    m = re.search(r'clCreateKernel\s*\(\s*\w+\s*,\s*"(\w+)"', driver_code)
    return m.group(1) if m else None


def _extract_kernel_name_from_cl(kernel_code: str) -> Optional[str]:
    """Extract the kernel name from __kernel void name(...)."""
    m = re.search(r'__kernel\s+void\s+(\w+)\s*\(', kernel_code)
    return m.group(1) if m else None


def _count_set_kernel_args(driver_code: str) -> int:
    """Count clSetKernelArg calls in the driver."""
    code = _strip_c_like_comments_and_strings(driver_code)
    return len(re.findall(r"\bclSetKernelArg\s*\(", code))


def _count_kernel_params(kernel_code: str) -> int:
    """Count parameters in the __kernel void signature."""
    m = re.search(r'__kernel\s+void\s+\w+\s*\(([^)]*)\)', kernel_code, re.DOTALL)
    if not m:
        return 0
    # Strip comments in the signature so commas inside comments do not
    # inflate the parameter count (e.g., "// [N, M, K]").
    params = _strip_c_like_comments_and_strings(m.group(1)).strip()
    if not params:
        return 0
    return len([p for p in params.split(',') if p.strip()])


def _strip_c_like_comments_and_strings(text: str) -> str:
    """Remove C/OpenCL strings and comments for safer regex counting."""
    # Remove string and char literals first so comment markers inside strings
    # do not affect comment stripping.
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    # Remove block comments, then line comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def _generate_dummy_kernel(kernel_name: str, kernel_code: str) -> str:
    """Generate a no-op kernel with matching signature for driver validation.

    The dummy kernel does nothing — so the driver's CPU reference comparison
    should print "failed".  If it prints "passed" instead, the driver's
    verification logic is broken.
    """
    m = re.search(
        rf'(__kernel\s+void\s+{re.escape(kernel_name)}\s*\([^)]*\))',
        kernel_code,
        re.DOTALL,
    )
    if not m:
        raise FeedbackError(
            f"Could not extract kernel signature for '{kernel_name}' from the kernel code."
        )
    signature = m.group(1)
    return (
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
        f"{signature}\n{{\n    // dummy: no-op\n}}\n"
    )


# ---------------------------------------------------------------------------
# File-rule callables (passed to FeedbackConfig.file_rules)
# ---------------------------------------------------------------------------

def rule_driver_has_cl_header(code: str):
    if "CL/cl.h" not in code:
        raise FeedbackError(
            "driver.c must #include <CL/cl.h>. Add the missing header."
        )


def rule_driver_has_main(code: str):
    if re.search(r'\bint\s+main\s*\(', code) is None:
        raise FeedbackError(
            "driver.c must define int main(...). Add the missing entry point."
        )


def rule_driver_has_passed_failed(code: str):
    if '"passed' not in code or '"failed' not in code:
        raise FeedbackError(
            'driver.c must print "passed" or "failed" for verification. '
            "Add the missing verification output."
        )


def rule_driver_has_read_file(code: str):
    if "read_file" not in code:
        raise FeedbackError(
            "driver.c must use read_file() to load kernel.cl at runtime. "
            "Add the read_file helper and load the kernel source."
        )


def rule_driver_no_torch(code: str):
    if "torch" in code.lower() or "libtorch" in code.lower():
        raise FeedbackError(
            "driver.c must be plain C — no PyTorch/LibTorch. Remove all torch references."
        )


_INFRA_ERROR_PATTERNS = [
    "permission denied",
    "connection refused",
    "connection timed out",
    "no route to host",
    "network is unreachable",
    "ssh_mkdir",
    "ssh_scp",
    "ssh_exec",
    "host key verification failed",
]


def _is_infra_error(error_str: str) -> bool:
    """Detect SSH/network infrastructure errors that the LLM cannot fix."""
    low = error_str.lower()
    return any(pat in low for pat in _INFRA_ERROR_PATTERNS)


DRIVER_RULES = [
    rule_driver_has_cl_header,
    rule_driver_has_main,
    rule_driver_has_passed_failed,
    rule_driver_has_read_file,
    rule_driver_no_torch,
]


class OpenCLKgenAgent(FeedbackAgent):
    """Translates a PyTorch reference into driver.c + kernel.cl for Adreno."""

    def __init__(
        self,
        fb_config: FeedbackConfig,
        reference_code: str,
        precision: str = "fp16",
    ):
        fb_config.system_prompt = build_system_prompt(precision)
        fb_config.init_user_prompt = build_user_prompt(reference_code, precision)
        fb_config.file_rules = DRIVER_RULES
        super().__init__(fb_config)

        self.reference_code = reference_code
        self.precision = precision
        self.exec_timeout_s = 600

    # ------------------------------------------------------------------
    # File extension overrides
    # ------------------------------------------------------------------
    def get_intermediate_filepath(self, attempt_id, task_id) -> Path:
        return self.folder / f"attempt{attempt_id}_task{task_id}_driver.c"

    def _kernel_filepath(self, attempt_id, task_id) -> Path:
        return self.folder / f"attempt{attempt_id}_task{task_id}_kernel.cl"

    # ------------------------------------------------------------------
    # Extract dual code blocks from one LLM response
    # ------------------------------------------------------------------
    def get_code_from_response(
        self, response, attempt_id, task_id, logger
    ) -> tuple[str, Path]:
        """Extract driver.c (```c) and kernel.cl (```opencl) from the response.

        Returns (driver_code, driver_filepath).  The kernel is written as a
        side-effect so that get_feedback can find it.
        """
        driver_code = extract_code_from_response(response, tag="c")
        kernel_code = extract_code_from_response(response, tag="opencl")

        if driver_code is None:
            raise FeedbackError(
                "Error: driver.c code must be in a ```c code block."
            )
        if kernel_code is None:
            raise FeedbackError(
                "Error: kernel.cl code must be in a ```opencl code block."
            )

        driver_fp = self.get_intermediate_filepath(attempt_id, task_id)
        kernel_fp = self._kernel_filepath(attempt_id, task_id)

        write_code_to_file(driver_code, driver_fp, logger)
        write_code_to_file(kernel_code, kernel_fp, logger)

        return driver_code, driver_fp

    # ------------------------------------------------------------------
    # Contract validation (static, before compilation)
    # ------------------------------------------------------------------
    def _validate_contract(self, driver_code: str, kernel_code: str):
        """Check that driver and kernel agree on kernel name and arg count."""
        driver_kname = _extract_kernel_name_from_driver(driver_code)
        kernel_kname = _extract_kernel_name_from_cl(kernel_code)

        if driver_kname is None:
            raise FeedbackError(
                "Cannot find clCreateKernel(..., \"name\", ...) in driver.c. "
                "The driver must create the kernel by name."
            )
        if kernel_kname is None:
            raise FeedbackError(
                "Cannot find __kernel void name(...) in kernel.cl. "
                "The kernel file must define a __kernel function."
            )
        if driver_kname != kernel_kname:
            raise FeedbackError(
                f"Kernel name mismatch: driver uses \"{driver_kname}\" but "
                f"kernel.cl defines \"{kernel_kname}\". They must match exactly."
            )

        driver_args = _count_set_kernel_args(driver_code)
        kernel_params = _count_kernel_params(kernel_code)
        if driver_args != kernel_params:
            raise FeedbackError(
                f"Argument count mismatch: driver sets {driver_args} kernel args "
                f"but kernel.cl has {kernel_params} parameters. They must match."
            )

    # ------------------------------------------------------------------
    # Two-step feedback: dummy kernel check + real kernel check
    # ------------------------------------------------------------------
    async def get_feedback(self, response, attempt_id, task_id, logger) -> Feedback:
        timer = NamedTimer()

        # 1. Extract code
        try:
            driver_code, driver_fp = self.get_code_from_response(
                response, attempt_id, task_id, logger
            )
        except FeedbackError as e:
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(e)},
                ],
                feedback=str(e),
            )

        kernel_fp = self._kernel_filepath(attempt_id, task_id)
        kernel_code = kernel_fp.read_text()

        # 2. Static file rules on driver
        try:
            self.check_rules(driver_code)
        except FeedbackError as e:
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(e)},
                ],
                feedback=str(e),
            )

        # 3. Contract validation
        try:
            self._validate_contract(driver_code, kernel_code)
        except FeedbackError as e:
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(e)},
                ],
                feedback=str(e),
            )

        kernel_name = _extract_kernel_name_from_cl(kernel_code)

        # 4. Step 1: Dummy kernel — driver must print "failed"
        try:
            dummy_code = _generate_dummy_kernel(kernel_name, kernel_code)
        except FeedbackError as e:
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": str(e)},
                ],
                feedback=str(e),
            )

        dummy_fp = self.folder / f"attempt{attempt_id}_task{task_id}_dummy_kernel.cl"
        dummy_fp.write_text(dummy_code)

        dummy_step_passed = False
        try:
            stdout_list, stderr_list, _, _ = await compile_and_run_opencl(
                main_filepath=driver_fp,
                kernel_filepath=dummy_fp,
                gpu=self.gpu,
                timer=timer,
                logger=logger,
                timeout=self.exec_timeout_s,
                num_runs=1,
            )
            dummy_stdout = stdout_list[0] if stdout_list else ""
            dummy_stderr = stderr_list[0] if stderr_list else ""

            if "passed" in dummy_stdout.lower():
                msg = (
                    "DRIVER VERIFICATION BUG: Your driver accepted a dummy (no-op) kernel "
                    "that produces garbage output, yet printed 'passed'. This means the "
                    "verification logic is broken — it does not actually compare GPU output "
                    "against the CPU reference, or the tolerance is far too loose.\n\n"
                    "Fix the driver's verification: ensure it computes the CPU reference "
                    "correctly, reads back the GPU buffer, and compares element-by-element "
                    f"with a reasonable tolerance ({'1e-1 for fp16' if self.precision == 'fp16' else '1e-3 for fp32'}).\n\n"
                    f"Dummy kernel stdout:\n{dummy_stdout}\n"
                    f"Dummy kernel stderr:\n{dummy_stderr}"
                )
                return Feedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": msg},
                    ],
                    feedback=msg,
                )
            dummy_step_passed = True
        except FeedbackError as e:
            error_str = str(e)
            if _is_infra_error(error_str):
                raise RuntimeError(
                    f"Infrastructure error (not a code issue): {error_str}"
                ) from e
            # The driver exits non-zero when verification fails, which the GPU
            # server reports as a FeedbackError.  If stdout contains "failed",
            # the dummy-kernel check actually *succeeded* (driver correctly
            # rejected the no-op kernel).
            if "failed" in error_str.lower() and "passed" not in error_str.lower():
                dummy_step_passed = True
            elif "compile" in error_str.lower() or "compilation" in error_str.lower():
                msg = (
                    f"Compilation failed when building driver.c with a dummy kernel "
                    f"(this indicates a driver-side compile error, not a kernel issue):\n\n"
                    f"{error_str}"
                )
                return Feedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": msg},
                    ],
                    feedback=msg,
                )
            else:
                msg = (
                    f"Running driver.c with a dummy kernel crashed or timed out:\n\n"
                    f"{error_str}\n\n"
                    "The dummy kernel is intentionally a no-op, so the driver should "
                    "print 'failed' and exit — not crash. Fix the driver."
                )
                return Feedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": msg},
                    ],
                    feedback=msg,
                )

        if dummy_step_passed:
            logger.info("Step 1 passed: dummy kernel correctly detected as 'failed'")

        # 5. Step 2: Real kernel — driver must print "passed"
        try:
            stdout_list, stderr_list, _, success = await compile_and_run_opencl(
                main_filepath=driver_fp,
                kernel_filepath=kernel_fp,
                gpu=self.gpu,
                timer=timer,
                logger=logger,
                timeout=self.exec_timeout_s,
                num_runs=1,
                passed_keyword="passed",
            )
            real_stdout = stdout_list[0] if stdout_list else ""
            real_stderr = stderr_list[0] if stderr_list else ""

            if not success:
                msg = (
                    "The real kernel failed verification (driver printed 'failed').\n\n"
                    f"stdout:\n{real_stdout}\n\nstderr:\n{real_stderr}\n\n"
                    "Fix the kernel so that its output matches the CPU reference "
                    "within tolerance."
                )
                return Feedback(
                    new_messages=[
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": msg},
                    ],
                    feedback=msg,
                )
        except FeedbackError as e:
            error_str = str(e)
            if _is_infra_error(error_str):
                raise RuntimeError(
                    f"Infrastructure error (not a code issue): {error_str}"
                ) from e
            msg = (
                f"Compilation or execution of driver.c + kernel.cl failed:\n\n{e}\n\n"
                "Fix the code so it compiles and runs correctly."
            )
            return Feedback(
                new_messages=[
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": msg},
                ],
                feedback=msg,
            )

        # Both steps passed — mark as success
        success_driver_fp = self.folder / f"success_attempt{attempt_id}_task{task_id}_driver.c"
        success_kernel_fp = self.folder / f"success_attempt{attempt_id}_task{task_id}_kernel.cl"
        success_driver_fp.write_text(driver_code)
        success_kernel_fp.write_text(kernel_code)

        logger.info(
            f"Step 2 passed: real kernel verified. Saved to {success_driver_fp.name} + {success_kernel_fp.name}"
        )

        return Feedback(
            new_messages=[
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Both verification steps passed. Well done!"},
            ],
            success=True,
            filename=success_driver_fp,
            contents=driver_code,
        )

    # ------------------------------------------------------------------
    # choose_best_task: prefer the most recent successful pair
    # ------------------------------------------------------------------
    def choose_best_task(self, successful_tasks: list[Path]) -> Path:
        return successful_tasks[-1]
