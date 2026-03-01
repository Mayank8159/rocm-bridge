"""
ROCm Bridge - Daemon Entry Point
================================
The headless background worker that monitors a project directory
and automatically orchestrates the 4-layer transpilation pipeline.

Features:
- File Event Debouncing (prevents CPU thrashing on multi-saves)
- Thread-safe Producer/Consumer queue
- Graceful shutdown handling
- Full pipeline integration (Discovery -> Core -> Engine -> Analyzer)

Usage:
    python main.py --project ./my_cuda_project --profile mi300x
"""

import os
import time
import queue
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

# Import our 4 Layers
from rocm_bridge.core.logging_config import setup_logging
from rocm_bridge.core.hal import HardwareAbstractionLayer
from rocm_bridge.core.state import StateManager, FileStatus
from rocm_bridge.discovery.scanner import ProjectScanner
from rocm_bridge.engine.hipify_wrapper import HipifyWrapper
from rocm_bridge.analyzer.parser import CudaParser
from rocm_bridge.analyzer.metrics import PerformanceSimulator
from rocm_bridge.engine.recommender import RecommendationEngine
from rocm_bridge.engine.patcher import ByteOffsetPatcher, Patch

logger = logging.getLogger("ROCmBridge_Daemon")


class DebouncedFileHandler(FileSystemEventHandler):
    """
    Monitors the filesystem for changes.
    Uses debouncing to prevent multiple triggers for a single file save.
    """
    
    def __init__(self, task_queue: queue.Queue, debounce_seconds: float = 0.5):
        self.task_queue = task_queue
        self.debounce_seconds = debounce_seconds
        self.last_triggered: Dict[str, float] = {}
        self._lock = threading.Lock()
        
        # Only watch target source files
        self.valid_extensions = {'.cu', '.cuh', '.cpp', '.hpp', '.h'}

    def process_event(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix.lower() not in self.valid_extensions:
            return

        current_time = time.time()
        path_str = str(file_path)

        with self._lock:
            last_time = self.last_triggered.get(path_str, 0)
            if current_time - last_time > self.debounce_seconds:
                self.last_triggered[path_str] = current_time
                logger.info(f"File change detected: {file_path.name}")
                self.task_queue.put(path_str)

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent):
            self.process_event(event)

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self.process_event(event)


class DaemonWorker:
    """
    The main consumer thread that pulls files from the queue and executes
    the 4-layer transpilation pipeline safely.
    """
    
    def __init__(self, project_path: str, task_queue: queue.Queue, profile_name: str = "mi300x"):
        self.project_path = Path(project_path).resolve()
        self.task_queue = task_queue
        self.is_running = True
        self.profile_name = profile_name
        
        # Initialize Layer 1 (Core)
        self.state_manager = StateManager(str(self.project_path))
        self.hal = HardwareAbstractionLayer(profile_name=profile_name)
        self.hardware_profile = self.hal.get_profile()
        
        # Initialize Output Directories
        self.output_dir = self.project_path / "rocm_output"
        self.patch_dir = self.output_dir / "patches"
        self.hip_dir = self.output_dir / "hipified"
        self.backup_dir = self.output_dir / "backups"
        
        for dir_path in [self.patch_dir, self.hip_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Worker loop."""
        logger.info(f"Daemon worker started. Target GPU: {self.hardware_profile.name}")
        
        while self.is_running:
            try:
                # Block until a file is available (timeout allows checking is_running flag)
                file_path = self.task_queue.get(timeout=1.0)
                try:
                    self._process_file(file_path)
                except Exception as e:
                    logger.error(f"Catastrophic failure processing {file_path}: {e}", exc_info=True)
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def _process_file(self, file_path_str: str):
        """Executes the complete 4-layer transpilation pipeline."""
        file_path = Path(file_path_str)
        
        try:
            rel_path = file_path.relative_to(self.project_path).as_posix()
        except ValueError:
            logger.warning(f"File {file_path} is outside project directory")
            return
        
        # --- LAYER 1: State Check ---
        current_hash = self.state_manager.calculate_file_hash(file_path)
        if not current_hash:
            logger.warning(f"Skipping {file_path.name} - unable to read file.")
            return
            
        if not self.state_manager.is_file_changed(rel_path, current_hash):
            logger.debug(f"Skipping {file_path.name} - no content changes.")
            return
            
        logger.info(f"Processing pipeline for: {rel_path}")
        self.state_manager.update_file_state(
            rel_path, FileStatus.ANALYZING, 0.0, current_hash, save_to_disk=False
        )

        # --- LAYER 4: Hipify (Pass 1) ---
        hipify = HipifyWrapper()
        hip_output_path = self.hip_dir / f"{file_path.stem}.hip"
        
        hip_result = hipify.transpile(str(file_path), str(hip_output_path))
        if not hip_result.success:
            logger.error(f"Hipify failed for {file_path.name}")
            self.state_manager.update_file_state(
                rel_path, FileStatus.ERROR, 0.0, current_hash,
                error_message="Hipify Translation Failed"
            )
            return

        # --- LAYER 3: AST Analyzer (Pass 2) ---
        parser = CudaParser()
        parse_result = parser.analyze(str(hip_output_path))
        
        if not parse_result.success:
            logger.error(f"AST Parsing failed for {hip_output_path.name}")
            self.state_manager.update_file_state(
                rel_path, FileStatus.ERROR, 0.0, current_hash,
                error_message=parse_result.error_message
            )
            return

        # --- LAYER 3: Metrics & Physics ---
        simulator = PerformanceSimulator(self.hardware_profile.to_dict())
        metrics = simulator.simulate_metrics({
            'issues': parse_result.issues,
            'kernels_detected': parse_result.kernels_detected
        })

        # --- LAYER 4: Recommendation & Patch Generation ---
        recommender = RecommendationEngine()
        report = recommender.generate(parse_result.issues, metrics.to_dict())
        
        patch_file_path = self.patch_dir / f"{file_path.name}.patch"
        
        # Generate patch file (simplified - full implementation would extract byte offsets)
        patcher = ByteOffsetPatcher(backup_dir=str(self.backup_dir))
        
        logger.info(
            f"Successfully processed {file_path.name}. "
            f"Score: {parse_result.score}/100, Health: {metrics.health_score:.1f}/100"
        )
        
        # --- LAYER 1: State Update ---
        self.state_manager.update_file_state(
            rel_path, 
            FileStatus.CONVERTED if parse_result.score > 80 else FileStatus.PARTIAL, 
            parse_result.score / 100.0, 
            current_hash,
            converted_path=str(hip_output_path),
            patch_path=str(patch_file_path)
        )


def main():
    parser = argparse.ArgumentParser(description="ROCm Bridge Background Daemon")
    parser.add_argument(
        "--project", type=str, required=True,
        help="Path to project directory to monitor"
    )
    parser.add_argument(
        "--profile", type=str, default="mi300x",
        help="Hardware profile to simulate (mi300x, mi250x, rx7900xtx)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args()

    project_path = Path(args.project).resolve()
    if not project_path.exists() or not project_path.is_dir():
        print(f"Error: Project directory {project_path} does not exist.")
        return

    # 1. Setup Layer 1 Environment
    log_file = project_path / "rocm_bridge.log"
    setup_logging(
        log_level=args.log_level,
        log_format="standard",
        log_file=str(log_file)
    )
    logger.info(f"Starting ROCm Bridge Daemon on {project_path}")
    
    # 2. Run Initial Discovery Scan
    logger.info("Running initial project scan...")
    scanner = ProjectScanner(str(project_path))
    scanner.scan()
    
    # 3. Initialize Thread-Safe Queue & Worker
    task_queue = queue.Queue()
    worker = DaemonWorker(str(project_path), task_queue, profile_name=args.profile)
    
    worker_thread = threading.Thread(target=worker.run)
    worker_thread.daemon = True
    worker_thread.start()

    # Queue all initially discovered CUDA files
    for entry in scanner.get_convertible_files():
        task_queue.put(entry.path)
        logger.info(f"Queued initial file: {entry.relative_path}")

    # 4. Setup FileSystem Watchdog
    event_handler = DebouncedFileHandler(task_queue)
    observer = Observer()
    observer.schedule(event_handler, str(project_path), recursive=True)
    observer.start()
    
    logger.info("Watchdog observer active. Waiting for file changes...")
    logger.info(f"Output directory: {worker.output_dir}")
    logger.info(f"Patch directory: {worker.patch_dir}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        observer.stop()
        worker.is_running = False
    
    observer.join()
    worker_thread.join()
    logger.info("Daemon gracefully shutdown.")


if __name__ == "__main__":
    main()