# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pandas as pd
from pathlib import Path
from io import StringIO

__all__ = [
    "annotate_source",
    "format_ncu_details_as_csv",
    "format_ncu_source_as_csv",
    "UTILIZATION_METRICS",
]

UTILIZATION_METRICS = [
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
    "smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.avg.pct_of_peak_sustained_active",
]


def annotate_source(
    cuda_path: Path, source_dfs: list[pd.DataFrame], details_dfs: list[pd.DataFrame]
):
    """Modified code based on chatwithncu.
    Annotate lines of code that take long cycles.
    The profiling report with selected metrics and suggestions is at end.
    Combine the inline annotation and profiling report of several kernels.
    """
    assert cuda_path.exists(), f"File not found: {cuda_path}"
    assert len(source_dfs) == len(
        details_dfs
    ), f"Expected {len(source_dfs)} source csv(s), got {len(details_dfs)}"
    num_kernels = len(source_dfs)

    line_comment = {}
    # Annotate for multiple kernels
    summary_report = []
    # Get annotation and profiling report
    for kernel_idx in range(num_kernels):
        df_source = source_dfs[kernel_idx]
        df_details = details_dfs[kernel_idx]
        
        # Handle empty source dataframes (when source profiling is skipped)
        if df_source.empty or "CUDA" not in df_source.columns:
            # Skip source-level annotations, only generate summary from details
            df_source_cuda = pd.DataFrame()
            df_source_sass = pd.DataFrame()
        else:
            # analyze source counters file
            df_source_cuda = df_source.dropna(subset=["CUDA"])
            df_source_sass = df_source.dropna(subset=["SASS"])

        # Find hot spot
        hotspot = 0
        total_stalls = 0
        if not df_source.empty and "Warp Stall Sampling (All Samples)" in df_source.columns:
            column = df_source.get(
                "Warp Stall Sampling (All Samples)",
                df_source.get("Warp Stall Sampling (All Cycles)", None),
            )
            if column is not None:
                column = pd.to_numeric(column, errors="coerce")
                total_stalls = column.sum()
                if len(column) > 0:
                    max_insts_executed_index = column.idxmax()
                    for index in range(max_insts_executed_index, -1, -1):
                        row = df_source.loc[index]
                        if "CUDA" in row and not pd.isna(row["CUDA"]):
                            hotspot = int(row["Line No"])
                            break

        # Find divergent branches
        divergent_branches = []
        column = "Divergent Branches"
        if not df_source_cuda.empty and column in df_source_cuda.columns:
            df_source_cuda.loc[:, column] = pd.to_numeric(
                df_source_cuda[column], errors="coerce"
            )
            for index, row in df_source_cuda.iterrows():
                if not pd.isna(row[column]) and row[column] > 0:
                    divergent_branches.append(int(row["Line No"]))

        # Find gmem accesses
        gmem_accesses = []
        column = "L2 Theoretical Sectors Global"
        if not df_source_cuda.empty and column in df_source_cuda.columns:
            df_source_cuda.loc[:, column] = pd.to_numeric(
                df_source_cuda[column], errors="coerce"
            )
            for index, row in df_source_cuda.iterrows():
                if not pd.isna(row[column]) and row[column] > 0:
                    gmem_accesses.append(int(row["Line No"]))

        # Find uncoalesced memory accesses
        uncoalesced_accesses = []
        column = "L2 Theoretical Sectors Global Excessive"
        if not df_source_cuda.empty and column in df_source_cuda.columns:
            df_source_cuda.loc[:, column] = pd.to_numeric(
                df_source_cuda[column], errors="coerce"
            )
            for index, row in df_source_cuda.iterrows():
                if not pd.isna(row[column]) and row[column] > 0:
                    uncoalesced_accesses.append(int(row["Line No"]))

        # Find smem accesses
        smem_accesses = []
        column = "L1 Wavefronts Shared"
        if not df_source_cuda.empty and column in df_source_cuda.columns:
            df_source_cuda.loc[:, column] = pd.to_numeric(
                df_source_cuda[column], errors="coerce"
            )
            for index, row in df_source_cuda.iterrows():
                if not pd.isna(row[column]) and row[column] > 0:
                    smem_accesses.append(int(row["Line No"]))

        # Find smem bank conflicts
        conflicts = []
        column = "L1 Wavefronts Shared Excessive"
        if not df_source_cuda.empty and column in df_source_cuda.columns:
            df_source_cuda.loc[:, column] = pd.to_numeric(
                df_source_cuda[column], errors="coerce"
            )
            for index, row in df_source_cuda.iterrows():
                if not pd.isna(row[column]) and row[column] > 0:
                    conflicts.append(int(row["Line No"]))

        # Find the total stalls
        stall_columns = []
        if not df_source_cuda.empty:
            stall_columns = [
                col for col in df_source_cuda.columns if col.find("Not Issued") != -1
            ]
            for col in stall_columns:
                df_source_cuda.loc[:, col] = pd.to_numeric(
                    df_source_cuda[col], errors="coerce"
                )
        threshold = 5

        cuda_file_contents = cuda_path.read_text()
        # Start counting lines from 1 to skip the CSV header
        for line_number, line in enumerate(cuda_file_contents.splitlines(), 1):
            comment = ""
            if line_number == hotspot:
                comment = comment + " (HOT SPOT)"
            if not df_source_cuda.empty and "Line No" in df_source_cuda.columns:
                df_source_cuda.loc[:, "Line No"] = pd.to_numeric(
                    df_source_cuda["Line No"], errors="coerce"
                )
                mask = df_source_cuda["Line No"] == line_number
                row_exists = mask.any()
                if row_exists and stall_columns:
                    filtered_df_source = df_source_cuda[mask]
                    for col in stall_columns:
                        if col in filtered_df_source.columns and col in df_source.columns:
                            stall_value = filtered_df_source.iloc[
                                0, df_source.columns.get_loc(col)
                            ]
                            # NCU can return missing/NaN stall sampling data (or total stalls = 0)
                            # for some kernels / configs. Avoid crashing annotation.
                            if total_stalls is None or pd.isna(total_stalls) or float(total_stalls) <= 0:
                                stall_percent = 0
                            else:
                                if stall_value is None or pd.isna(stall_value):
                                    stall_percent = 0
                                else:
                                    stall_percent = int(100 * float(stall_value) / float(total_stalls))
                            if stall_percent >= threshold:
                                comment = (
                                    comment + " " + str(col) + " = " + str(stall_percent) + "%"
                                )
            if line_number in gmem_accesses:
                if line_number in uncoalesced_accesses:
                    comment = comment + " (Uncoalesced Global Memory Access)"
                else:
                    comment = comment + " (Coalesced Global Memory Access)"
            if line_number in smem_accesses:
                if line_number in conflicts:
                    comment = comment + " (Shared Memory Bank Conflicts)"
                else:
                    comment = comment + " (NO Bank Conflicts)"
            if line_number in divergent_branches:
                comment = comment + " (Divergent Branch)"
            if comment != "":
                line_comment[line_number] = " // Profile information:" + comment + "\n"

        # analyze details page
        metric_list = [
            "Elapsed Cycles",
            "Memory Throughput",
            "Compute (SM) Throughput",
            "Avg. Active Threads Per Warp",
            "Active Warps Per Scheduler",
            "Registers Per Thread",
            "Waves Per SM",
            "Dynamic Shared Memory Per Block",
            "Static Shared Memory Per Block",
            "Theoretical Occupancy",
            "Achieved Occupancy",
        ]
        utilization_name = [
            "SM efficiency",
            "DRAM utilization",
            "L2 cache utilization",
            "L1 cache and shared memory utilization",
            "Double precision utilization",
            "Single precision utilization",
            "Half precision utilization",
            "Tensor core utilization",
            "Integer utilization",
        ]
        utilization_dict = {}

        # Handle empty details dataframe
        if df_details.empty:
            # Only include "no details available" summary if we have source info to indicate the kernel exists
            # If both source and details are empty, skip the summary entirely
            if not df_source.empty or kernel_idx < len(source_dfs):
                report = f"###PROFILE SUMMARY (no details available):\n\n"
                report += "No profiling details available (source profiling was skipped).\n"
                report += "This may indicate the kernel was not executed or kernel name matching failed.\n"
                summary_report.append(report)
            # Otherwise, skip adding empty summary to avoid clutter
            continue

        kernel_name = df_details.iloc[0]["Kernel Name"].split("(")[0] if "Kernel Name" in df_details.columns and len(df_details) > 0 else "Unknown"
        report = f"###PROFILE SUMMARY for {kernel_name}:\n\n"
        for index, row in df_details.iterrows():
            if row["Rule Type"] == "OPT":
                report = report + "  ADVICE: " + row["Rule Description"]
                if not pd.isna(row["Estimated Speedup"]):
                    report = (
                        report
                        + " Fixing this may yield an estimated "
                        + str(row["Estimated Speedup Type"])
                        + " speedup of "
                        + str(row["Estimated Speedup"])
                        + "%."
                    )
                report += "\n"
            if row["Metric Name"] in metric_list:
                report = (
                    report
                    + "The "
                    + str(row["Metric Name"])
                    + " for this kernel is "
                    + str(row["Metric Value"])
                    + " "
                    + (
                        str(row["Metric Unit"])
                        if not pd.isna(row["Metric Unit"])
                        else ""
                    )
                    + ".\n"
                )
            elif row["Metric Name"] in UTILIZATION_METRICS and not pd.isna(
                row["Metric Value"]
            ):
                utilization_dict[
                    utilization_name[UTILIZATION_METRICS.index(row["Metric Name"])]
                ] = (str(row["Metric Value"]) + " " + str(row["Metric Unit"]))

        report += "\nLatency metrics:\n"
        report += (
            "The "
            + utilization_name[0]
            + " for this kernel is "
            + utilization_dict[utilization_name[0]]
            + ".\n"
        )
        report += "\nMemory metrics:\n"
        for i in range(1, 4):
            if utilization_name[i] in utilization_dict:
                report += (
                    "The "
                    + utilization_name[i]
                    + " for this kernel is "
                    + utilization_dict[utilization_name[i]]
                    + ".\n"
                )
        report += "\nCompute metrics:\n"
        for i in range(4, len(utilization_name)):
            if utilization_name[i] in utilization_dict:
                report += (
                    "The "
                    + utilization_name[i]
                    + " for this kernel is "
                    + utilization_dict[utilization_name[i]]
                    + ".\n"
                )
        summary_report.append(report)

    # Write to file
    text = ""
    with open(cuda_path, "r") as file:
        for line_number, line in enumerate(file, 1):  # Start counting lines from 1
            if line_number in line_comment:
                text += line.rstrip() + line_comment[line_number]
            else:
                text += line
    text += "\n/*\n"
    sep = ""
    for report in summary_report:
        text += sep
        sep = "\n\n\n"
        text += report
    text += "*/"

    return text


def parse_csv_from_log(
    log: str, marker_string: str, header_replacement: str = ""
) -> pd.DataFrame:
    """From chatwithncu.
    Extract the csv content from the NCU log and return it.

    This removes lines like:
    ==PROF== Connected to process 2223422 (/tmp/kernelagent/compile_env/build/main)
    passed
    ==PROF== Disconnected from process 2223422
    "File Path","/tmp/kernelagent/compile_env/cuda_model.cuh"
    """
    csv_start_index = log.find(marker_string)
    if csv_start_index == -1:
        raise ValueError(f"Marker string not found in ncu output: {log}")

    if header_replacement:
        log = log.replace(marker_string, header_replacement)

    # validate the CSV content
    csv = log[csv_start_index:]

    # Extract only the CSV content from the cuda file. Subsequent CSVs report reports on atomic methods located in the cuda library like __ldg, __shfl_down_sync, etc.
    if '"File Path"' in csv:
        csv = csv[: csv.find('"File Path"')]

    lines = csv.splitlines()
    df = pd.read_csv(StringIO(lines[0]))
    num_columns = len(df.columns)
    for i, line in enumerate(lines, 1):
        line_df = pd.read_csv(StringIO(line))
        num_entries = len(line_df.columns)
        if num_entries > num_columns:
            # num_entries may be smaller than num_columns if some columns are not applicable
            # for a particular metric.
            raise ValueError(
                f"Number of entries in line {i} ({num_entries}) is larger than the number of columns ({num_columns}). Full Log:\n{csv}"
            )
    return pd.read_csv(StringIO(csv))


def format_ncu_details_as_csv(ncu_output: str) -> pd.DataFrame:
    """From chatwithncu.
    Remove the header of ncu generated details csv content and return the cleaned string.
    """
    marker_string = '"ID","Process ID","Process Name","Host Name"'
    return parse_csv_from_log(ncu_output, marker_string)


def format_ncu_source_as_csv(ncu_output: str) -> pd.DataFrame:
    """From chatwithncu.
    Remove the header of ncu generated source csv content and return the cleaned string.
    """
    marker_string = '"Line No","Source","Address","Source"'
    header_replacement = '"Line No","CUDA","Address","SASS"'
    return parse_csv_from_log(ncu_output, marker_string, header_replacement)
