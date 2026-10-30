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
#!/usr/bin/env python3
"""
Standalone GPU server starter for shared usage across multiple kernelblaster processes.
"""
import sys
import argparse
import signal
from pathlib import Path

# Add the project root to Python path (go up one level from scripts/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernelblaster.servers.management import start_standalone_gpu_server


def main():
    parser = argparse.ArgumentParser(description="Start a standalone GPU server")
    parser.add_argument("--port", type=int, default=2002, help="Port to run GPU server on")
    parser.add_argument("--log-file", type=str, default="gpu_server.log", help="Log file path")
    parser.add_argument("--info-file", type=str, default="gpu_server_info.txt", help="File to write server info")
    
    args = parser.parse_args()
    
    gpu_process = None
    
    def cleanup_handler(signum, frame):
        print(f"\nReceived signal {signum}, cleaning up...")
        if gpu_process:
            gpu_process.terminate()
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    try:
        print(f"Starting GPU server on port {args.port}...")
        gpu_process, gpu_url = start_standalone_gpu_server(
            port=args.port,
            log_file_path=Path(args.log_file) if args.log_file else None
        )
        
        print(f"GPU server started successfully at: {gpu_url}")
        print(f"GPU server PID: {gpu_process.pid}")
        
        # Write server info to file
        with open(args.info_file, 'w') as f:
            f.write(f"{gpu_url}\n{gpu_process.pid}\n")
        
        print(f"Server info written to: {args.info_file}")
        print("GPU server is running. Press Ctrl+C to stop.")
        
        # Wait for the process to complete
        gpu_process.wait()
        
    except Exception as e:
        print(f"Error starting GPU server: {e}", file=sys.stderr)
        if gpu_process:
            gpu_process.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main() 