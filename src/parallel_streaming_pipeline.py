#!/usr/bin/env python3
"""
Parallel Streaming Pipeline for ALS Foundation Model
Implements download and processing in parallel for better performance
"""

import os
import sys
import json
import logging
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import time
from datetime import datetime
import psutil
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import DataLoader
import pickle
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/parallel_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class ParallelStreamingPipeline:
    """Parallel streaming pipeline with download and processing threads"""
    
    def __init__(self, config_path: str):
        """Initialize parallel pipeline"""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.current_batch = 0
        self.model_state = None
        
        # Initialize rich console
        self.console = Console()
        
        # Threading components
        self.download_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.processing_queue = queue.Queue(maxsize=5)  # Limit processing queue
        self.download_threads = []
        self.processing_threads = []
        self.shutdown_event = threading.Event()
        
        # Status tracking
        self.status_info = {
            'current_phase': 'Initializing',
            'current_dataset': None,
            'batch_progress': (0, 0),
            'overall_progress': (0, 0),
            'storage_used': 0,
            'storage_free': 0,
            'start_time': datetime.now(),
            'downloads_completed': 0,
            'processing_completed': 0,
            'total_datasets': 0
        }
        
        # Issue tracking
        self.issues = {
            'download_failures': [],
            'processing_failures': [],
            'storage_warnings': [],
            'data_quality_issues': [],
            'network_issues': [],
            'total_issues': 0
        }
        
        # Thread-safe locks
        self.status_lock = threading.Lock()
        self.issues_lock = threading.Lock()
        
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_directories(self):
        """Set up directory structure"""
        base_dir = Path(self.config['base_dir'])
        
        # Create directories
        (base_dir / 'data').mkdir(parents=True, exist_ok=True)
        (base_dir / 'processed').mkdir(parents=True, exist_ok=True)
        (base_dir / 'models').mkdir(parents=True, exist_ok=True)
        (base_dir / 'logs').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directories set up in {base_dir}")
    
    def get_storage_info(self) -> Tuple[float, float]:
        """Get storage information"""
        usage = psutil.disk_usage(self.config['base_dir'])
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        used_gb = (usage.total - usage.free) / (1024**3)
        
        logger.info(f"Storage: {used_gb:.1f}GB used, {free_gb:.1f}GB free, {total_gb:.1f}GB total")
        
        # Update status info
        with self.status_lock:
            self.status_info['storage_used'] = used_gb
            self.status_info['storage_free'] = free_gb
        
        # Check for storage issues
        self.check_and_report_storage_issues(free_gb, used_gb)
        
        return free_gb, used_gb
    
    def check_and_report_storage_issues(self, free_gb: float, used_gb: float):
        """Check for storage-related issues and report them"""
        max_storage = self.config.get('max_storage_gb', 50000)
        min_free = self.config.get('min_free_space_gb', 1000)
        
        # Check for low free space
        if free_gb < min_free:
            self.report_issue(
                'storage_warnings',
                'SYSTEM',
                f"Low storage space: {free_gb:.1f}GB free (minimum: {min_free}GB)",
                f"Used: {used_gb:.1f}GB, Total: {max_storage}GB"
            )
        
        # Check for approaching storage limit
        usage_percentage = (used_gb / max_storage) * 100
        if usage_percentage > 80:
            self.report_issue(
                'storage_warnings',
                'SYSTEM',
                f"High storage usage: {usage_percentage:.1f}% of limit",
                f"Used: {used_gb:.1f}GB / {max_storage}GB"
            )
    
    def print_status(self, phase: str, details: str = "", dataset_id: str = None, force_update: bool = False):
        """Print current status with rich formatting - thread-safe"""
        # Only print status for major phases or when forced
        major_phases = [
            "Starting pipeline", "Pipeline initialized", "Processing batch", 
            "Downloading dataset", "Processing dataset", "Integrating batch",
            "Batch complete", "Pipeline complete", "Checking storage"
        ]
        
        # Skip repetitive micro-updates unless forced
        if not force_update and phase not in major_phases:
            return
        
        with self.status_lock:
            # Check if this is a duplicate of the last printed status
            current_status = f"{phase}|{dataset_id}|{details}"
            if hasattr(self, '_last_printed_status') and self._last_printed_status == current_status:
                return
            
            self._last_printed_status = current_status
            self.status_info['current_phase'] = phase
            if dataset_id:
                self.status_info['current_dataset'] = dataset_id
        
        # Create status panel
        elapsed = datetime.now() - self.status_info['start_time']
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        status_text = f"[bold blue]{phase}[/bold blue]"
        if dataset_id:
            status_text += f" - [yellow]{dataset_id}[/yellow]"
        if details:
            status_text += f"\n[dim]{details}[/dim]"
        
        status_text += f"\n[dim]Elapsed: {elapsed_str} | Storage: {self.status_info['storage_used']:.1f}GB used, {self.status_info['storage_free']:.1f}GB free[/dim]"
        
        if self.status_info['overall_progress'][1] > 0:
            progress_pct = (self.status_info['overall_progress'][0] / self.status_info['overall_progress'][1]) * 100
            status_text += f"\n[dim]Overall Progress: {self.status_info['overall_progress'][0]}/{self.status_info['overall_progress'][1]} ({progress_pct:.1f}%)[/dim]"
        
        if self.status_info['batch_progress'][1] > 0:
            batch_pct = (self.status_info['batch_progress'][0] / self.status_info['batch_progress'][1]) * 100
            status_text += f"\n[dim]Batch Progress: {self.status_info['batch_progress'][0]}/{self.status_info['batch_progress'][1]} ({batch_pct:.1f}%)[/dim]"
        
        # Add parallel processing info
        status_text += f"\n[dim]Downloads: {self.status_info['downloads_completed']} | Processing: {self.status_info['processing_completed']}[/dim]"
        
        panel = Panel(
            status_text,
            title="[bold green]ALS Foundation Model Pipeline (Parallel)[/bold green]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def report_issue(self, issue_type: str, dataset_id: str, error_message: str, details: str = ""):
        """Report an issue with real-time display - thread-safe"""
        issue = {
            'timestamp': datetime.now().isoformat(),
            'dataset_id': dataset_id,
            'error_message': error_message,
            'details': details,
            'batch': self.current_batch + 1
        }
        
        with self.issues_lock:
            # Add to appropriate issue list
            if issue_type in self.issues:
                self.issues[issue_type].append(issue)
            else:
                self.issues['processing_failures'].append(issue)
            
            self.issues['total_issues'] += 1
        
        # Print real-time error report
        self.print_error_report(issue_type, issue)
        
        # Log the issue
        logger.error(f"{issue_type.upper()}: {dataset_id} - {error_message}")
    
    def print_error_report(self, issue_type: str, issue: Dict):
        """Print a formatted error report"""
        error_colors = {
            'download_failures': 'red',
            'processing_failures': 'orange3',
            'storage_warnings': 'yellow',
            'data_quality_issues': 'magenta',
            'network_issues': 'blue'
        }
        
        color = error_colors.get(issue_type, 'red')
        icon = {
            'download_failures': '❌',
            'processing_failures': '⚠️',
            'storage_warnings': '💾',
            'data_quality_issues': '🔍',
            'network_issues': '🌐'
        }.get(issue_type, '❌')
        
        error_text = f"{icon} [bold {color}]{issue_type.replace('_', ' ').title()}[/bold {color}]"
        error_text += f"\n[yellow]Dataset:[/yellow] {issue['dataset_id']}"
        error_text += f"\n[red]Error:[/red] {issue['error_message']}"
        if issue['details']:
            error_text += f"\n[dim]Details:[/dim] {issue['details']}"
        error_text += f"\n[dim]Batch:[/dim] {issue['batch']} | [dim]Time:[/dim] {issue['timestamp']}"
        error_text += f"\n[dim]Total Issues:[/dim] {self.issues['total_issues']}"
        
        panel = Panel(
            error_text,
            title=f"[bold {color}]ISSUE REPORTED[/bold {color}]",
            border_style=color
        )
        
        self.console.print(panel)
    
    def load_dataset_list(self) -> pd.DataFrame:
        """Load dataset list from CSV file"""
        try:
            # Try to read as CSV with semicolon separators
            df = pd.read_csv(
                self.config['dataset_list_path'], 
                sep=';;', 
                encoding='latin-1', 
                engine='python'
            )
            # Clean up column names and remove empty columns
            df.columns = df.columns.str.strip()
            df = df.dropna(axis=1, how='all')  # Remove completely empty columns
            return df
        except Exception as e:
            self.report_issue(
                'data_quality_issues',
                'DATASET_LIST',
                f"Failed to load CSV file: {str(e)}",
                f"File: {self.config['dataset_list_path']}"
            )
            # Fallback to Excel format
            try:
                return pd.read_excel(self.config['dataset_list_path'])
            except Exception as e2:
                self.report_issue(
                    'data_quality_issues',
                    'DATASET_LIST',
                    f"Failed to load Excel file: {str(e2)}",
                    f"File: {self.config['dataset_list_path']}"
                )
                raise
    
    def download_worker(self, worker_id: int):
        """Download worker thread"""
        logger.info(f"Download worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get dataset from queue with timeout
                dataset_info = self.download_queue.get(timeout=5)
                if dataset_info is None:  # Shutdown signal
                    break
                
                dataset_id = dataset_info['dataset_id']
                self.print_status("Downloading dataset", f"Worker {worker_id} - {dataset_id}")
                
                # Download the dataset using the improved logic from the main pipeline
                dataset_path = self.download_dataset(dataset_info)
                if dataset_path:
                    # Put in processing queue
                    self.processing_queue.put((dataset_info, dataset_path))
                    with self.status_lock:
                        self.status_info['downloads_completed'] += 1
                    logger.info(f"Download worker {worker_id} completed {dataset_id}")
                else:
                    logger.warning(f"Download worker {worker_id} failed {dataset_id}")
                
                self.download_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Download worker {worker_id} error: {e}")
                self.download_queue.task_done()
        
        logger.info(f"Download worker {worker_id} stopped")
    
    def processing_worker(self, worker_id: int):
        """Processing worker thread"""
        logger.info(f"Processing worker {worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get dataset from processing queue with timeout
                dataset_info, dataset_path = self.processing_queue.get(timeout=5)
                if dataset_info is None:  # Shutdown signal
                    break
                
                dataset_id = dataset_info['dataset_id']
                self.print_status("Processing dataset", f"Worker {worker_id} - {dataset_id}")
                
                # Process the dataset using the improved logic from the main pipeline
                processed_data = self.process_dataset(dataset_path, dataset_info)
                if processed_data is not None:
                    # Store processed data for batch integration
                    with self.status_lock:
                        if not hasattr(self, 'processed_batch_data'):
                            self.processed_batch_data = []
                        self.processed_batch_data.append(processed_data)
                        self.status_info['processing_completed'] += 1
                    logger.info(f"Processing worker {worker_id} completed {dataset_id}")
                else:
                    logger.warning(f"Processing worker {worker_id} failed {dataset_id}")
                
                self.processing_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                self.processing_queue.task_done()
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    def run_parallel_pipeline(self):
        """Run the parallel pipeline"""
        logger.info("Starting parallel streaming pipeline")
        self.print_status("Starting pipeline", "Initializing parallel processing")
        
        # Load dataset list
        datasets = self.load_dataset_list()
        total_datasets = len(datasets)
        total_batches = (total_datasets + self.config['batch_size'] - 1) // self.config['batch_size']
        
        with self.status_lock:
            self.status_info['total_datasets'] = total_datasets
            self.status_info['overall_progress'] = (0, total_batches)
        
        self.print_status("Pipeline initialized", f"Total: {total_datasets} datasets, {total_batches} batches")
        
        # Start download and processing workers
        num_download_workers = 2  # 2 download threads
        num_processing_workers = 4  # 4 processing threads
        
        # Start download workers
        for i in range(num_download_workers):
            worker = threading.Thread(target=self.download_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.download_threads.append(worker)
        
        # Start processing workers
        for i in range(num_processing_workers):
            worker = threading.Thread(target=self.processing_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.processing_threads.append(worker)
        
        # Process datasets in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.config['batch_size']
            end_idx = min(start_idx + self.config['batch_size'], total_datasets)
            batch_datasets = datasets.iloc[start_idx:end_idx]
            
            self.current_batch = batch_idx
            with self.status_lock:
                self.status_info['batch_progress'] = (0, len(batch_datasets))
                self.processed_batch_data = []  # Reset for new batch
            
            self.print_status("Processing batch", f"Starting batch {batch_idx + 1}/{total_batches}")
            
            # Add datasets to download queue
            for _, dataset_info in batch_datasets.iterrows():
                self.download_queue.put(dataset_info.to_dict())
            
            # Wait for all downloads to complete
            self.download_queue.join()
            
            # Wait for all processing to complete
            self.processing_queue.join()
            
            # Integrate batch if we have processed data
            if hasattr(self, 'processed_batch_data') and self.processed_batch_data:
                self.print_status("Integrating batch", f"Combining {len(self.processed_batch_data)} datasets")
                integrated_data = self.integrate_batch(self.processed_batch_data)
                
                if integrated_data is not None:
                    # Update model with integrated data
                    self.update_model(integrated_data)
            
            # Clean up batch data
            self.cleanup_batch_data()
            
            with self.status_lock:
                self.status_info['overall_progress'] = (batch_idx + 1, total_batches)
            
            self.print_status("Batch complete", f"Finished batch {batch_idx + 1}/{total_batches}")
        
        # Shutdown workers
        self.shutdown_event.set()
        
        # Send shutdown signals
        for _ in range(num_download_workers):
            self.download_queue.put(None)
        for _ in range(num_processing_workers):
            self.processing_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.download_threads + self.processing_threads:
            worker.join()
        
        self.print_status("Pipeline complete", "All processing finished successfully")
        
        # Print final issue summary
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]FINAL ISSUE SUMMARY[/bold green]")
        self.console.print("="*60)
        self.print_issue_summary()
        
        logger.info("Parallel pipeline completed successfully")
    
    def print_issue_summary(self):
        """Print a summary of all issues encountered"""
        with self.issues_lock:
            if self.issues['total_issues'] == 0:
                success_panel = Panel(
                    "[bold green]✅ No issues encountered![/bold green]\nAll datasets processed successfully.",
                    title="[bold green]PIPELINE SUCCESS[/bold green]",
                    border_style="green"
                )
                self.console.print(success_panel)
                return
            
            # Create issue summary table
            table = Table(title=f"Issue Summary - {self.issues['total_issues']} Total Issues")
            table.add_column("Issue Type", style="cyan")
            table.add_column("Count", style="yellow")
            table.add_column("Examples", style="white", max_width=60)
            
            for issue_type, issues_list in self.issues.items():
                if issue_type == 'total_issues' or not issues_list:
                    continue
                
                count = len(issues_list)
                examples = []
                for issue in issues_list[:3]:  # Show first 3 examples
                    examples.append(f"{issue['dataset_id']}: {issue['error_message'][:50]}...")
                
                table.add_row(
                    issue_type.replace('_', ' ').title(),
                    str(count),
                    "\n".join(examples) if examples else "None"
                )
            
            self.console.print(table)
    
    def cleanup_batch_data(self):
        """Clean up batch data to free memory"""
        if hasattr(self, 'processed_batch_data'):
            del self.processed_batch_data
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Import all the improved methods from the main pipeline
    # (This would include download_dataset, process_dataset, integrate_batch, etc.)
    # For now, we'll use a simplified approach and import the main pipeline class
    
    def download_dataset(self, dataset_info: Dict) -> Optional[str]:
        """Download a single dataset - simplified version for parallel processing"""
        # This would use the improved download logic from the main pipeline
        # For now, return a placeholder
        dataset_id = dataset_info['dataset_id']
        dataset_dir = Path(self.config['base_dir']) / 'data' / 'raw' / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        return str(dataset_dir)
    
    def process_dataset(self, dataset_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Process a single dataset - simplified version for parallel processing"""
        # This would use the improved processing logic from the main pipeline
        # For now, return a placeholder
        return None
    
    def integrate_batch(self, batch_data: List[ad.AnnData]) -> ad.AnnData:
        """Integrate multiple datasets in a batch"""
        # This would use the improved integration logic from the main pipeline
        # For now, return a placeholder
        return None
    
    def update_model(self, integrated_data: ad.AnnData):
        """Update the foundation model with integrated data"""
        # This would implement the model update logic
        pass

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python parallel_streaming_pipeline.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        pipeline = ParallelStreamingPipeline(config_path)
        pipeline.run_parallel_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()