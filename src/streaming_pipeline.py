#!/usr/bin/env python3
"""
Streaming Data Processing Pipeline for ALS Foundation Model
Processes datasets in batches to work within storage constraints
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
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DownloadTracker:
    """Track downloaded and processed datasets to enable resume functionality"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.tracker_file = self.base_dir / 'download_tracker.json'
        self.tracked_datasets = self.load_tracker()
    
    def load_tracker(self) -> Dict:
        """Load existing download tracker"""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load download tracker: {e}")
                return {}
        return {}
    
    def save_tracker(self):
        """Save download tracker to file"""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.tracked_datasets, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save download tracker: {e}")
    
    def is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset is already downloaded and processed"""
        if dataset_id not in self.tracked_datasets:
            return False
        
        dataset_info = self.tracked_datasets[dataset_id]
        
        # Check if download is complete
        if not dataset_info.get('download_complete', False):
            return False
        
        # Check if processing is complete
        if not dataset_info.get('processing_complete', False):
            return False
        
        # Check if files still exist
        dataset_path = Path(dataset_info.get('dataset_path', ''))
        if not dataset_path.exists():
            logger.info(f"Dataset {dataset_id} marked as downloaded but files missing, re-downloading")
            return False
        
        # Check if processed data exists
        processed_path = Path(dataset_info.get('processed_path', ''))
        if not processed_path.exists():
            logger.info(f"Dataset {dataset_id} marked as processed but processed file missing, re-processing")
            return False
        
        return True
    
    def mark_downloaded(self, dataset_id: str, dataset_path: str, data_type: str = ''):
        """Mark dataset as downloaded"""
        if dataset_id not in self.tracked_datasets:
            self.tracked_datasets[dataset_id] = {}
        
        self.tracked_datasets[dataset_id].update({
            'download_complete': True,
            'dataset_path': dataset_path,
            'data_type': data_type,
            'download_timestamp': datetime.now().isoformat()
        })
        self.save_tracker()
        logger.info(f"Marked {dataset_id} as downloaded")
    
    def mark_processed(self, dataset_id: str, processed_path: str, n_cells: int = 0, n_genes: int = 0):
        """Mark dataset as processed"""
        if dataset_id not in self.tracked_datasets:
            self.tracked_datasets[dataset_id] = {}
        
        self.tracked_datasets[dataset_id].update({
            'processing_complete': True,
            'processed_path': processed_path,
            'n_cells': n_cells,
            'n_genes': n_genes,
            'processing_timestamp': datetime.now().isoformat()
        })
        self.save_tracker()
        logger.info(f"Marked {dataset_id} as processed ({n_cells} cells, {n_genes} genes)")
    
    def mark_failed(self, dataset_id: str, error_type: str, error_message: str):
        """Mark dataset as failed"""
        if dataset_id not in self.tracked_datasets:
            self.tracked_datasets[dataset_id] = {}
        
        self.tracked_datasets[dataset_id].update({
            'failed': True,
            'error_type': error_type,
            'error_message': error_message,
            'failure_timestamp': datetime.now().isoformat()
        })
        self.save_tracker()
        logger.info(f"Marked {dataset_id} as failed: {error_type}")
    
    def get_statistics(self) -> Dict:
        """Get download and processing statistics"""
        total = len(self.tracked_datasets)
        downloaded = sum(1 for d in self.tracked_datasets.values() if d.get('download_complete', False))
        processed = sum(1 for d in self.tracked_datasets.values() if d.get('processing_complete', False))
        failed = sum(1 for d in self.tracked_datasets.values() if d.get('failed', False))
        
        return {
            'total_tracked': total,
            'downloaded': downloaded,
            'processed': processed,
            'failed': failed,
            'success_rate': (processed / total * 100) if total > 0 else 0
        }

class StreamingPipeline:
    """Main pipeline class for streaming data processing"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.setup_directories()
        self.current_batch = 0
        self.model_state = None
        
        # Initialize rich console for status printing
        self.console = Console()
        self.progress = None
        self.current_task = None
        
        # Status tracking
        self.status_info = {
            'current_phase': 'Initializing',
            'current_dataset': None,
            'batch_progress': (0, 0),
            'overall_progress': (0, 0),
            'storage_used': 0,
            'storage_free': 0,
            'start_time': datetime.now()
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
        
        # Download tracking
        self.download_tracker = DownloadTracker(self.config['base_dir'])
        
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_directories(self):
        """Create necessary directories"""
        base_dir = Path(self.config['base_dir'])
        self.dirs = {
            'data': base_dir / 'data',
            'raw': base_dir / 'data' / 'raw',
            'processed': base_dir / 'data' / 'processed',
            'metadata': base_dir / 'data' / 'metadata',
            'model': base_dir / 'model',
            'checkpoints': base_dir / 'model' / 'checkpoints',
            'embeddings': base_dir / 'model' / 'embeddings',
            'cache': base_dir / 'cache',
            'outputs': base_dir / 'outputs',
            'logs': base_dir / 'model' / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Directories set up in {base_dir}")
        self.print_status("Directories created", "Setup complete")
    
    def check_storage_space(self) -> Tuple[float, float]:
        """Check available storage space"""
        usage = psutil.disk_usage(self.config['base_dir'])
        total_gb = usage.total / (1024**3)
        free_gb = usage.free / (1024**3)
        used_gb = (usage.total - usage.free) / (1024**3)
        
        logger.info(f"Storage: {used_gb:.1f}GB used, {free_gb:.1f}GB free, {total_gb:.1f}GB total")
        
        # Update status info
        self.status_info['storage_used'] = used_gb
        self.status_info['storage_free'] = free_gb
        
        # Check for storage issues
        self.check_and_report_storage_issues(free_gb, used_gb)
        
        return free_gb, used_gb
    
    def print_status(self, phase: str, details: str = "", dataset_id: str = None, force_update: bool = False):
        """Print current status with rich formatting - only for major phase changes"""
        # Only print status for major phases or when forced
        major_phases = [
            "Starting pipeline", "Pipeline initialized", "Processing batch", 
            "Downloading dataset", "Processing dataset", "Integrating batch",
            "Batch complete", "Pipeline complete", "Checking storage"
        ]
        
        # Skip repetitive micro-updates unless forced
        if not force_update and phase not in major_phases:
            return
        
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
        
        panel = Panel(
            status_text,
            title="[bold green]ALS Foundation Model Pipeline[/bold green]",
            border_style="green"
        )
        
        self.console.print(panel)
    
    def print_progress_table(self, datasets: List[Dict]):
        """Print a table showing batch progress"""
        table = Table(title=f"Batch {self.current_batch + 1} - {len(datasets)} datasets")
        table.add_column("Dataset ID", style="cyan")
        table.add_column("Title", style="white", max_width=50)
        table.add_column("Repository", style="magenta")
        table.add_column("Data Type", style="yellow")
        table.add_column("Status", style="green")
        
        for dataset in datasets:
            table.add_row(
                dataset.get('dataset_id', 'N/A'),
                dataset.get('Title', 'N/A')[:50] + "..." if len(dataset.get('Title', '')) > 50 else dataset.get('Title', 'N/A'),
                dataset.get('repository', 'N/A'),
                dataset.get('data_type', 'N/A'),
                "Pending"
            )
        
        self.console.print(table)
    
    def update_progress(self, overall_current: int, overall_total: int, batch_current: int = None, batch_total: int = None):
        """Update progress information"""
        self.status_info['overall_progress'] = (overall_current, overall_total)
        if batch_current is not None and batch_total is not None:
            self.status_info['batch_progress'] = (batch_current, batch_total)
    
    def report_issue(self, issue_type: str, dataset_id: str, error_message: str, details: str = ""):
        """Report an issue with real-time display"""
        issue = {
            'timestamp': datetime.now().isoformat(),
            'dataset_id': dataset_id,
            'error_message': error_message,
            'details': details,
            'batch': self.current_batch + 1
        }
        
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
    
    def print_issue_summary(self):
        """Print a summary of all issues encountered"""
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
        
        # Print detailed issue breakdown
        for issue_type, issues_list in self.issues.items():
            if issue_type == 'total_issues' or not issues_list:
                continue
            
            if len(issues_list) > 0:
                self.console.print(f"\n[bold]{issue_type.replace('_', ' ').title()}:[/bold]")
                for issue in issues_list:
                    self.console.print(f"  • [yellow]{issue['dataset_id']}[/yellow]: {issue['error_message']}")
    
    def check_and_report_storage_issues(self, free_gb: float, used_gb: float):
        """Check for storage-related issues and report them"""
        max_storage = self.config.get('max_storage_gb', 550)
        min_free = self.config.get('min_free_space_gb', 50)
        
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
    
    def get_next_batch(self, batch_size: int = 3) -> List[Dict]:
        """Get next batch of datasets to process"""
        self.print_status("Loading dataset list", "Reading CSV file")
        datasets = self.load_dataset_list()
        
        start_idx = self.current_batch * batch_size
        end_idx = start_idx + batch_size
        
        if start_idx >= len(datasets):
            self.print_status("No more datasets", "Pipeline complete")
            return []
        
        batch = datasets.iloc[start_idx:end_idx].to_dict('records')
        
        # Calculate total batches for progress tracking
        total_batches = (len(datasets) + batch_size - 1) // batch_size
        self.update_progress(self.current_batch, total_batches)
        
        self.print_status(f"Preparing batch {self.current_batch + 1}/{total_batches}", f"{len(batch)} datasets")
        self.print_progress_table(batch)
        
        logger.info(f"Processing batch {self.current_batch + 1}: {len(batch)} datasets")
        return batch
    
    def download_dataset(self, dataset_info: Dict) -> Optional[str]:
        """Download a single dataset"""
        dataset_id = dataset_info['dataset_id']
        repository = dataset_info.get('repository', 'GEO')
        data_type = dataset_info.get('data_type', '')
        
        # Check if already downloaded and processed
        if self.download_tracker.is_downloaded(dataset_id):
            logger.info(f"Dataset {dataset_id} already downloaded and processed, skipping")
            return str(self.dirs['raw'] / dataset_id)
        
        self.print_status("Downloading dataset", f"Repository: {repository}, Type: {data_type}", dataset_id)
        
        # Create dataset directory
        dataset_dir = self.dirs['raw'] / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        try:
            # Download based on repository
            if repository.upper() == 'GEO':
                self.download_geo_dataset(dataset_id, dataset_dir, data_type)
            elif repository.upper() == 'EBI':
                self.download_ebi_dataset(dataset_id, dataset_dir, data_type)
            elif repository.upper() == 'SRA':
                self.download_sra_dataset(dataset_id, dataset_dir)
            else:
                logger.error(f"Unknown repository: {repository}")
                self.print_status("Download failed", f"Unknown repository: {repository}", dataset_id)
                self.download_tracker.mark_failed(dataset_id, 'unknown_repository', f"Unknown repository: {repository}")
                return None
            
            # Mark as downloaded
            self.download_tracker.mark_downloaded(dataset_id, str(dataset_dir), data_type)
            
            self.print_status("Download complete", f"Data saved to {dataset_dir}", dataset_id)
            return str(dataset_dir)
            
        except Exception as e:
            self.report_issue(
                'download_failures',
                dataset_id,
                f"Download failed: {str(e)}",
                f"Repository: {repository}, Type: {data_type}"
            )
            self.download_tracker.mark_failed(dataset_id, 'download_failure', str(e))
            return None
    
    def download_geo_dataset(self, dataset_id: str, output_dir: Path, data_type: str = ''):
        """Download dataset from GEO"""
        # Status already printed by download_dataset, no need to print again
        
        # Strategy 1: Always try to download tar files first (most common format)
        if self.try_download_tar_file(dataset_id, output_dir):
            return
        
        # Strategy 2: Check if processed data is available and download it
        if self.check_processed_data_available(dataset_id, data_type):
            try:
                self.download_processed_geo_data(dataset_id, output_dir, data_type)
                return
            except Exception as e:
                logger.warning(f"Processed data download failed for {dataset_id}: {e}")
        
        # Strategy 3: Try to download individual files based on data_type
        if self.try_download_individual_files(dataset_id, output_dir, data_type):
            return
        
        # Strategy 4: Try SRA data as fallback (only for datasets that likely have SRA data)
        if self.should_try_sra_fallback(dataset_id, data_type):
            if self.try_download_sra_data(dataset_id, output_dir):
                return
        
        # Strategy 5: Try alternative download methods
        if self.try_alternative_download_methods(dataset_id, output_dir):
            return
        
        # If all strategies fail, report the issue
        self.report_issue(
            'download_failures',
            dataset_id,
            "No data available from any source",
            f"Tried tar files, processed data, individual files, SRA fallback, and alternative methods. Data type: {data_type}"
        )
        raise Exception(f"No data available for {dataset_id}")
    
    def try_download_tar_file(self, dataset_id: str, output_dir: Path) -> bool:
        """Try to download tar file from GEO using FTP first, then HTTP"""
        try:
            # Try FTP URLs first (faster for large files)
            ftp_urls = [
                f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/suppl/{dataset_id}_RAW.tar",
                f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/suppl/{dataset_id}.tar"
            ]
            
            # Try HTTP URLs as fallback
            http_urls = [
                f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={dataset_id}&format=file",
                f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={dataset_id}&format=file&file={dataset_id}_RAW.tar",
                f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={dataset_id}&format=file&file={dataset_id}.tar"
            ]
            
            # Try FTP first
            for url in ftp_urls:
                if self.download_file_with_protocol(url, output_dir, dataset_id, "FTP"):
                    return True
            
            # Fallback to HTTP
            for url in http_urls:
                if self.download_file_with_protocol(url, output_dir, dataset_id, "HTTP"):
                    return True
            
            return False
            
        except Exception as e:
            self.report_issue(
                'network_issues',
                dataset_id,
                f"Tar file download failed: {str(e)}",
                "Network or server issue"
            )
            return False
    
    def download_file_with_protocol(self, url: str, output_dir: Path, dataset_id: str, protocol: str) -> bool:
        """Download file using specified protocol (FTP or HTTP) with retry logic"""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                tar_file = output_dir / f"{dataset_id}.tar"
                
                if protocol == "FTP":
                    # Use wget with FTP and retry options
                    cmd = f"wget --passive-ftp --tries=3 --timeout=300 -O {tar_file} '{url}'"
                else:
                    # Use wget with HTTP/HTTPS and retry options
                    cmd = f"wget --tries=3 --timeout=300 -O {tar_file} '{url}'"
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0 and tar_file.exists() and tar_file.stat().st_size > 1000:
                    logger.info(f"Downloaded tar file via {protocol} from {url} (attempt {attempt + 1})")
                    # Extract tar file
                    self.extract_tar_file(tar_file, output_dir)
                    return True
                else:
                    logger.debug(f"{protocol} download failed from {url} (attempt {attempt + 1}): {result.stderr}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
            except subprocess.TimeoutExpired:
                logger.debug(f"{protocol} download timeout from {url} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
            except Exception as e:
                logger.debug(f"{protocol} download error from {url} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        return False
    
    def extract_tar_file(self, tar_file: Path, output_dir: Path):
        """Extract tar file and organize contents"""
        try:
            # Extract tar file
            cmd = f"cd {output_dir} && tar -xf {tar_file.name}"
            subprocess.run(cmd, shell=True, check=True)
            
            # Remove the tar file to save space
            tar_file.unlink()
            
            logger.info(f"Extracted tar file {tar_file.name}")
            
        except Exception as e:
            self.report_issue(
                'processing_failures',
                tar_file.name,
                f"Failed to extract tar file: {str(e)}",
                f"File: {tar_file}"
            )
            raise
    
    def try_download_sra_data(self, dataset_id: str, output_dir: Path) -> bool:
        """Try to download SRA data as fallback using NCBI SRA database"""
        try:
            # First check if this dataset actually has SRA data using NCBI SRA search
            # For GEO datasets, check if there are SRA entries
            if dataset_id.startswith('GSE'):
                # Use esearch to query SRA database for this GEO dataset
                # This searches the NCBI SRA database at https://www.ncbi.nlm.nih.gov/sra
                sra_check_cmd = f"esearch -db sra -query \"{dataset_id}[GSE]\" | efetch -format docsum | grep -c 'Run acc'"
                result = subprocess.run(sra_check_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0 or int(result.stdout.strip()) == 0:
                    logger.info(f"No SRA data available for GEO dataset {dataset_id} in NCBI SRA database")
                    return False
                else:
                    logger.info(f"Found SRA data for GEO dataset {dataset_id} in NCBI SRA database")
            
            # For direct SRA IDs, validate they exist in the database
            elif dataset_id.startswith(('SRR', 'SRP', 'SRS', 'SRX')):
                sra_validate_cmd = f"esearch -db sra -query \"{dataset_id}\" | efetch -format docsum | grep -c 'Run acc'"
                result = subprocess.run(sra_validate_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode != 0 or int(result.stdout.strip()) == 0:
                    logger.info(f"SRA ID {dataset_id} not found in NCBI SRA database")
                    return False
            
            # Download SRA metadata first
            self.download_sra_metadata(dataset_id, output_dir)
            
            # Use SRA toolkit to download
            cmd = f"prefetch {dataset_id} --output-directory {output_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Check if the SRA file was actually downloaded
                sra_file = output_dir / dataset_id / f"{dataset_id}.sra"
                if sra_file.exists() and sra_file.stat().st_size > 1000:
                    # Convert to FASTQ
                    cmd = f"fastq-dump {output_dir}/{dataset_id} --outdir {output_dir} --split-files"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully downloaded and converted SRA data for {dataset_id}")
                        return True
                    else:
                        logger.warning(f"Failed to convert SRA to FASTQ for {dataset_id}: {result.stderr}")
                        return False
                else:
                    logger.warning(f"SRA file not found or too small for {dataset_id}")
                    return False
            else:
                logger.debug(f"SRA prefetch failed for {dataset_id}: {result.stderr}")
                return False
            
        except subprocess.TimeoutExpired:
            logger.warning(f"SRA download timeout for {dataset_id}")
            return False
        except Exception as e:
            logger.debug(f"SRA download error for {dataset_id}: {e}")
            return False
    
    def download_sra_metadata(self, dataset_id: str, output_dir: Path):
        """Download SRA metadata files for a dataset"""
        try:
            # Create metadata directory
            metadata_dir = output_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            # Download different types of metadata
            metadata_types = [
                ("runinfo", "runinfo"),  # Run information
                ("sample", "sample"),    # Sample information
                ("experiment", "experiment"),  # Experiment information
                ("study", "study")       # Study information
            ]
            
            for metadata_type, filename in metadata_types:
                try:
                    # Use esearch and efetch to get metadata
                    cmd = f"esearch -db sra -query \"{dataset_id}\" | efetch -format {metadata_type} > {metadata_dir}/{filename}_{dataset_id}.txt"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        # Check if file was created and has content
                        metadata_file = metadata_dir / f"{filename}_{dataset_id}.txt"
                        if metadata_file.exists() and metadata_file.stat().st_size > 100:
                            logger.info(f"Downloaded {metadata_type} metadata for {dataset_id}")
                        else:
                            # Remove empty file
                            if metadata_file.exists():
                                metadata_file.unlink()
                    else:
                        logger.debug(f"Failed to download {metadata_type} metadata for {dataset_id}")
                        
                except Exception as e:
                    logger.debug(f"Error downloading {metadata_type} metadata for {dataset_id}: {e}")
                    continue
            
            # Also try to download the full XML metadata
            try:
                xml_cmd = f"esearch -db sra -query \"{dataset_id}\" | efetch -format xml > {metadata_dir}/full_metadata_{dataset_id}.xml"
                result = subprocess.run(xml_cmd, shell=True, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    xml_file = metadata_dir / f"full_metadata_{dataset_id}.xml"
                    if xml_file.exists() and xml_file.stat().st_size > 1000:
                        logger.info(f"Downloaded full XML metadata for {dataset_id}")
                    else:
                        if xml_file.exists():
                            xml_file.unlink()
                            
            except Exception as e:
                logger.debug(f"Error downloading XML metadata for {dataset_id}: {e}")
            
            # Download sample attributes if available
            try:
                attributes_cmd = f"esearch -db sra -query \"{dataset_id}\" | efetch -format runinfo | cut -f1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30 > {metadata_dir}/runinfo_{dataset_id}.tsv"
                result = subprocess.run(attributes_cmd, shell=True, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    tsv_file = metadata_dir / f"runinfo_{dataset_id}.tsv"
                    if tsv_file.exists() and tsv_file.stat().st_size > 100:
                        logger.info(f"Downloaded runinfo TSV for {dataset_id}")
                    else:
                        if tsv_file.exists():
                            tsv_file.unlink()
                            
            except Exception as e:
                logger.debug(f"Error downloading runinfo TSV for {dataset_id}: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to download SRA metadata for {dataset_id}: {e}")
    
    def check_processed_data_available(self, dataset_id: str, data_type: str) -> bool:
        """Check if processed data is available for download"""
        # Look for processed data indicators in data_type
        processed_indicators = ['H5', 'MTX', 'TSV', 'RDS', 'seurat', 'processed']
        return any(indicator in data_type.upper() for indicator in processed_indicators)
    
    def should_try_sra_fallback(self, dataset_id: str, data_type: str) -> bool:
        """Determine if we should try SRA fallback based on dataset and data type"""
        # Only try SRA for datasets that are likely to have SRA data
        sra_indicators = ['SRA', 'FASTQ', 'BAM', 'raw']
        return any(indicator in data_type.upper() for indicator in sra_indicators)
    
    def try_download_individual_files(self, dataset_id: str, output_dir: Path, data_type: str) -> bool:
        """Try to download individual files based on data type"""
        try:
            # Try to download specific file types mentioned in data_type
            file_types = []
            if 'H5' in data_type.upper():
                file_types.extend(['*.h5', '*.h5ad'])
            if 'MTX' in data_type.upper():
                file_types.extend(['*.mtx', 'matrix.mtx*'])
            if 'TSV' in data_type.upper():
                file_types.extend(['*.tsv', '*.txt'])
            if 'RDS' in data_type.upper():
                file_types.extend(['*.rds'])
            if 'LOOM' in data_type.upper():
                file_types.extend(['*.loom'])
            
            if not file_types:
                return False
            
            # Try to download files directly from GEO
            for file_type in file_types:
                url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={dataset_id}&format=file&file={file_type}"
                try:
                    cmd = f"wget -O {output_dir}/{dataset_id}_{file_type.replace('*', 'file')} '{url}'"
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0:
                        logger.info(f"Downloaded individual file {file_type} for {dataset_id}")
                        return True
                except Exception as e:
                    logger.debug(f"Failed to download {file_type} for {dataset_id}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.debug(f"Individual file download failed for {dataset_id}: {e}")
            return False
    
    def try_alternative_download_methods(self, dataset_id: str, output_dir: Path) -> bool:
        """Try alternative download methods"""
        try:
            # Try downloading the series matrix file
            series_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={dataset_id}&format=file&file={dataset_id}_series_matrix.txt.gz"
            cmd = f"wget -O {output_dir}/{dataset_id}_series_matrix.txt.gz '{series_url}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Check if the file was downloaded and has content
                series_file = output_dir / f"{dataset_id}_series_matrix.txt.gz"
                if series_file.exists() and series_file.stat().st_size > 1000:
                    logger.info(f"Downloaded series matrix for {dataset_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Alternative download methods failed for {dataset_id}: {e}")
            return False
    
    def download_processed_geo_data(self, dataset_id: str, output_dir: Path, data_type: str):
        """Download processed data from GEO"""
        # Try to download processed matrices directly
        # This is a simplified approach - in practice, you'd need to check GEO's API
        # or use tools like GEOquery in R
        
        # For now, we'll try to download the series matrix file
        series_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={dataset_id}&targ=gsm&view=data&form=text"
        
        try:
            cmd = f"wget -O {output_dir}/{dataset_id}_series_matrix.txt.gz {series_url}"
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Downloaded series matrix for {dataset_id}")
        except Exception as e:
            self.report_issue(
                'network_issues',
                dataset_id,
                f"Processed data download failed: {str(e)}",
                "GEO server or network issue"
            )
            raise
    
    def download_ebi_dataset(self, dataset_id: str, output_dir: Path, data_type: str = ''):
        """Download dataset from EBI using FTP first, then HTTP"""
        logger.info(f"Downloading EBI dataset {dataset_id}")
        
        try:
            # Try FTP first (faster)
            ftp_url = f"ftp://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/{dataset_id[:3]}/{dataset_id}/{dataset_id}.raw.1.zip"
            http_url = f"https://www.ebi.ac.uk/arrayexpress/files/{dataset_id}/{dataset_id}.raw.1.zip"
            
            # Try FTP first
            if self.download_ebi_file_with_protocol(ftp_url, output_dir, dataset_id, "FTP"):
                return
            
            # Fallback to HTTP
            if self.download_ebi_file_with_protocol(http_url, output_dir, dataset_id, "HTTP"):
                return
            
            # If both fail, raise exception
            raise Exception("Both FTP and HTTP downloads failed")
            
        except Exception as e:
            self.report_issue(
                'network_issues',
                dataset_id,
                f"EBI download failed: {str(e)}",
                "EBI server or network issue"
            )
            raise
    
    def download_ebi_file_with_protocol(self, url: str, output_dir: Path, dataset_id: str, protocol: str) -> bool:
        """Download EBI file using specified protocol"""
        try:
            if protocol == "FTP":
                cmd = f"wget --passive-ftp -O {output_dir}/{dataset_id}.zip '{url}'"
            else:
                cmd = f"wget -O {output_dir}/{dataset_id}.zip '{url}'"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                # Extract if it's a zip file
                cmd = f"cd {output_dir} && unzip {dataset_id}.zip"
                subprocess.run(cmd, shell=True, check=True)
                logger.info(f"Downloaded EBI dataset via {protocol}")
                return True
            else:
                logger.debug(f"EBI {protocol} download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.debug(f"EBI {protocol} download error: {e}")
            return False
    
    def download_direct_file(self, url: str, output_dir: Path):
        """Download file directly from URL"""
        filename = url.split('/')[-1]
        output_path = output_dir / filename
        
        cmd = f"wget -O {output_path} {url}"
        subprocess.run(cmd, shell=True, check=True)
    
    def download_sra_dataset(self, sra_id: str, output_dir: Path):
        """Download dataset from SRA"""
        logger.info(f"Downloading SRA dataset {sra_id}")
        
        try:
            # Use SRA toolkit to download
            cmd = f"prefetch {sra_id} --output-directory {output_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to prefetch {sra_id}: {result.stderr}")
                raise Exception(f"SRA prefetch failed for {sra_id}")
            
            # Convert to FASTQ
            cmd = f"fastq-dump {output_dir}/{sra_id} --outdir {output_dir}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to convert {sra_id} to FASTQ: {result.stderr}")
                raise Exception(f"FASTQ conversion failed for {sra_id}")
            
            logger.info(f"Successfully downloaded and converted SRA dataset {sra_id}")
            
        except Exception as e:
            self.report_issue(
                'network_issues',
                sra_id,
                f"SRA dataset download failed: {str(e)}",
                "SRA toolkit or network issue"
            )
            raise
    
    def process_dataset(self, dataset_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Process a single dataset"""
        dataset_id = dataset_info['dataset_id']
        data_type = dataset_info.get('data_type', '')
        
        # Check if already processed
        if self.download_tracker.is_downloaded(dataset_id):
            logger.info(f"Dataset {dataset_id} already processed, loading from cache")
            # Load the processed data
            processed_path = self.dirs['processed'] / f"{dataset_id}_processed.h5ad"
            if processed_path.exists():
                try:
                    adata = sc.read_h5ad(processed_path)
                    logger.info(f"Loaded cached processed data for {dataset_id}: {adata.n_obs} cells, {adata.n_vars} genes")
                    return adata
                except Exception as e:
                    logger.warning(f"Failed to load cached data for {dataset_id}: {e}")
        
        self.print_status("Processing dataset", f"Data type: {data_type}", dataset_id)
        
        try:
            # Find the actual data files in the dataset directory
            self.print_status("Finding data files", "Scanning directory structure", dataset_id)
            data_files = self.find_data_files(dataset_path, data_type)
            
            if not data_files:
                self.report_issue(
                    'data_quality_issues',
                    dataset_id,
                    "No data files found after download",
                    f"Expected formats: {data_type}"
                )
                self.download_tracker.mark_failed(dataset_id, 'no_data_files', "No data files found after download")
                return None
            
            self.print_status("Loading data file", f"Found {len(data_files)} files, using {data_files[0]}", dataset_id)
            # Load data based on file type
            adata = self.load_data_file(data_files[0], dataset_info)
            
            if adata is None:
                self.report_issue(
                    'processing_failures',
                    dataset_id,
                    "Failed to load data file",
                    f"File: {data_files[0]}"
                )
                return None
            
            # Quality control
            self.print_status("Quality control", f"Starting with {adata.n_obs} cells, {adata.n_vars} genes", dataset_id)
            adata = self.quality_control(adata, dataset_info)
            
            # Normalization
            self.print_status("Normalizing data", "Applying log transform and scaling", dataset_id)
            adata = self.normalize_data(adata, dataset_info)
            
            # Feature extraction
            self.print_status("Extracting features", "Finding highly variable genes", dataset_id)
            adata = self.extract_features(adata, dataset_info)
            
            # Save processed data
            processed_path = self.dirs['processed'] / f"{dataset_id}_processed.h5ad"
            self.print_status("Saving processed data", f"Writing to {processed_path}", dataset_id)
            adata.write(processed_path)
            
            # Mark as processed in download tracker
            self.download_tracker.mark_processed(dataset_id, str(processed_path), adata.n_obs, adata.n_vars)
            
            self.print_status("Processing complete", f"Final: {adata.n_obs} cells, {adata.n_vars} genes", dataset_id)
            return adata
            
        except Exception as e:
            self.report_issue(
                'processing_failures',
                dataset_id,
                f"Processing failed: {str(e)}",
                f"Data type: {data_type}"
            )
            self.download_tracker.mark_failed(dataset_id, 'processing_failure', str(e))
            return None
    
    def find_data_files(self, dataset_path: str, data_type: str) -> List[str]:
        """Find data files in the dataset directory with improved pattern matching"""
        dataset_dir = Path(dataset_path)
        data_files = []
        
        # Define comprehensive file patterns for each data type
        file_patterns = {
            'H5': ['*.h5', '*.h5ad', '*feature_bc_matrix.h5', '*filtered_feature_bc_matrix.h5', '*raw_feature_bc_matrix.h5'],
            'H5AD': ['*.h5ad', '*.h5'],
            'MTX': ['matrix.mtx*', '*.mtx', '*matrix.mtx*'],
            'TSV': ['*.tsv', '*counts.tsv', '*expression.tsv'],
            'RDS': ['*.rds', '*seurat.rds', '*sce.rds'],
            'LOOM': ['*.loom', '*expression.loom'],
            'CSV': ['*.csv', '*counts.csv', '*expression.csv'],
            'SEURAT': ['*.rds', '*seurat.rds'],
            'BIGWIG': ['*.bw', '*.bigwig'],
            'TBI': ['*.tbi', '*.tbi.gz']
        }
        
        # Look for files based on data_type indicators
        for root, dirs, files in os.walk(dataset_dir):
            root_path = Path(root)
            
            # Check each data type mentioned in the data_type string
            for data_type_key in file_patterns.keys():
                if data_type_key.upper() in data_type.upper():
                    for pattern in file_patterns[data_type_key]:
                        data_files.extend(list(root_path.glob(pattern)))
                        # Also try case-insensitive search
                        data_files.extend(list(root_path.glob(pattern.upper())))
                        data_files.extend(list(root_path.glob(pattern.lower())))
        
        # If no specific files found, look for common formats recursively
        if not data_files:
            common_formats = [
                '*.h5ad', '*.h5', '*.loom', '*.rds', '*.csv', '*.tsv', '*.mtx',
                '*feature_bc_matrix.h5', '*filtered_feature_bc_matrix.h5', 
                '*raw_feature_bc_matrix.h5', 'matrix.mtx*', '*seurat.rds'
            ]
            for root, dirs, files in os.walk(dataset_dir):
                root_path = Path(root)
                for pattern in common_formats:
                    data_files.extend(list(root_path.glob(pattern)))
                    # Case-insensitive search
                    data_files.extend(list(root_path.glob(pattern.upper())))
                    data_files.extend(list(root_path.glob(pattern.lower())))
        
        # Remove duplicates and sort by priority
        data_files = list(set(data_files))
        data_files = self.prioritize_data_files(data_files)
        
        # Log what we found for debugging
        if data_files:
            logger.info(f"Found {len(data_files)} data files: {[f.name for f in data_files[:5]]}")
        else:
            logger.warning(f"No data files found in {dataset_path}")
            # List what files are actually there for debugging
            all_files = []
            for root, dirs, files in os.walk(dataset_dir):
                all_files.extend([Path(root) / f for f in files])
            logger.warning(f"Available files: {[f.name for f in all_files[:10]]}")
        
        return [str(f) for f in data_files]
    
    def prioritize_data_files(self, data_files: List[Path]) -> List[Path]:
        """Prioritize data files by format preference"""
        # Priority order: processed matrices > raw data
        priority_order = {
            '.h5ad': 1,    # Highest priority - processed AnnData
            '.h5': 2,      # 10x Genomics processed
            '.loom': 3,    # Loom format
            '.rds': 4,     # R SingleCellExperiment
            '.csv': 5,     # CSV matrices
            '.tsv': 6,     # TSV matrices
            '.mtx': 7,     # Matrix Market format
            '.fastq': 8,   # Raw sequencing data
            '.bam': 9      # Aligned reads
        }
        
        def get_priority(file_path: Path) -> int:
            suffix = file_path.suffix.lower()
            return priority_order.get(suffix, 10)  # Unknown formats get lowest priority
        
        return sorted(data_files, key=get_priority)
    
    def load_data_file(self, file_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Load data from a specific file"""
        file_path = Path(file_path)
        dataset_id = dataset_info['dataset_id']
        
        try:
            if file_path.suffix == '.h5ad':
                return sc.read_h5ad(file_path)
            elif file_path.suffix == '.h5':
                return sc.read_10x_h5(file_path)
            elif file_path.suffix == '.loom':
                return sc.read_loom(file_path)
            elif file_path.suffix == '.rds':
                return self.read_rds_file(file_path)
            elif file_path.suffix in ['.csv', '.tsv']:
                return self.read_matrix_file(file_path)
            elif file_path.name == 'matrix.mtx':
                return self.read_10x_mtx(file_path.parent)
            elif file_path.suffix == '.gz':
                # Handle compressed files
                return self.read_compressed_file(file_path, dataset_info)
            elif file_path.suffix in ['.fastq', '.fq']:
                # Handle raw sequencing data
                return self.process_raw_sequencing_data(str(file_path.parent), dataset_info)
            else:
                logger.warning(f"Unknown file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def read_matrix_file(self, file_path: Path) -> ad.AnnData:
        """Read matrix from CSV/TSV file"""
        try:
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.read_csv(file_path, sep='\t', index_col=0)
            
            # Transpose to get cells as rows, genes as columns
            adata = ad.AnnData(X=df.T.values, var=pd.DataFrame(index=df.T.columns), obs=pd.DataFrame(index=df.T.index))
            return adata
            
        except Exception as e:
            logger.error(f"Failed to read matrix file {file_path}: {e}")
            return None
    
    def read_10x_mtx(self, mtx_dir: Path) -> ad.AnnData:
        """Read 10x Genomics MTX format"""
        try:
            return sc.read_10x_mtx(mtx_dir, var_names='gene_symbols', cache=True)
        except Exception as e:
            logger.error(f"Failed to read 10x MTX from {mtx_dir}: {e}")
            return None
    
    def read_compressed_file(self, file_path: Path, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Read compressed files (.gz)"""
        try:
            # Check if it's a compressed MTX file
            if 'matrix.mtx' in file_path.name:
                # For single MTX files, read directly without 10x format requirements
                import gzip
                import pandas as pd
                import numpy as np
                from scipy import sparse
                
                # Read the compressed MTX file directly
                with gzip.open(file_path, 'rt') as f:
                    # Skip header lines (start with %)
                    header_lines = []
                    for line in f:
                        if line.startswith('%'):
                            header_lines.append(line.strip())
                        else:
                            # This line contains dimensions
                            rows, cols, entries = map(int, line.strip().split())
                            break
                    
                    # Read the data
                    data = []
                    row_indices = []
                    col_indices = []
                    
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            row_idx = int(parts[0]) - 1  # Convert to 0-based indexing
                            col_idx = int(parts[1]) - 1  # Convert to 0-based indexing
                            value = float(parts[2])
                            
                            row_indices.append(row_idx)
                            col_indices.append(col_idx)
                            data.append(value)
                
                # Create sparse matrix
                matrix = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(rows, cols))
                
                # Create AnnData object
                adata = ad.AnnData(X=matrix)
                
                # Try to find barcodes and features files
                dataset_dir = file_path.parent
                barcodes_file = None
                features_file = None
                
                # Look for barcodes file
                for barcode_pattern in ['barcodes.tsv', 'barcodes.tsv.gz', 'barcodes.txt', 'barcodes.txt.gz']:
                    barcode_candidates = list(dataset_dir.glob(f"*{barcode_pattern}"))
                    if barcode_candidates:
                        barcodes_file = barcode_candidates[0]
                        break
                
                # Look for features file
                for feature_pattern in ['features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz']:
                    feature_candidates = list(dataset_dir.glob(f"*{feature_pattern}"))
                    if feature_candidates:
                        features_file = feature_candidates[0]
                        break
                
                # Load barcodes if found
                if barcodes_file:
                    try:
                        if barcodes_file.suffix == '.gz':
                            with gzip.open(barcodes_file, 'rt') as f:
                                barcodes = [line.strip() for line in f]
                        else:
                            with open(barcodes_file, 'r') as f:
                                barcodes = [line.strip() for line in f]
                        adata.obs_names = barcodes
                    except Exception as e:
                        logger.warning(f"Failed to load barcodes from {barcodes_file}: {e}")
                        adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
                else:
                    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
                
                # Load features if found
                if features_file:
                    try:
                        if features_file.suffix == '.gz':
                            with gzip.open(features_file, 'rt') as f:
                                features_df = pd.read_csv(f, sep='\t', header=None)
                        else:
                            features_df = pd.read_csv(features_file, sep='\t', header=None)
                        
                        # Handle different feature file formats
                        if features_df.shape[1] >= 2:
                            adata.var_names = features_df.iloc[:, 1].astype(str)  # Usually gene symbols
                            adata.var['gene_ids'] = features_df.iloc[:, 0].astype(str)  # Usually gene IDs
                        else:
                            adata.var_names = features_df.iloc[:, 0].astype(str)
                    except Exception as e:
                        logger.warning(f"Failed to load features from {features_file}: {e}")
                        adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
                else:
                    adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
                
                logger.info(f"Successfully loaded compressed MTX file: {adata.n_obs} cells, {adata.n_vars} genes")
                return adata
            
            else:
                logger.warning(f"Unsupported compressed file format: {file_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to read compressed file {file_path}: {e}")
            return None
    
    def process_raw_sequencing_data(self, dataset_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Process raw sequencing data (FASTQ files)"""
        dataset_id = dataset_info['dataset_id']
        logger.info(f"Processing raw sequencing data for {dataset_id}")
        
        try:
            # This is a placeholder for raw data processing
            # In practice, you would need to:
            # 1. Align reads using STAR or similar
            # 2. Count reads using featureCounts or similar
            # 3. Create count matrix
            
            logger.warning(f"Raw sequencing data processing not implemented for {dataset_id}")
            logger.warning("This would require alignment and quantification pipelines")
            
            # For now, return None to skip raw data
            return None
            
        except Exception as e:
            logger.error(f"Failed to process raw sequencing data for {dataset_id}: {e}")
            return None
    
    def quality_control(self, adata: ad.AnnData, dataset_info: Dict) -> ad.AnnData:
        """Perform quality control on dataset"""
        try:
            # Calculate QC metrics
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
            
            # Filter cells
            min_genes = dataset_info.get('min_genes', 200)
            max_genes = dataset_info.get('max_genes', 5000)
            max_mt = dataset_info.get('max_mt_percent', 20)
            
            sc.pp.filter_cells(adata, min_genes=min_genes)
            
            # Safe filtering with existence checks
            if 'n_genes_by_counts' in adata.obs.columns:
                adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
            
            if 'pct_counts_mt' in adata.obs.columns:
                adata = adata[adata.obs.pct_counts_mt < max_mt, :]
            
            # Filter genes
            min_cells = dataset_info.get('min_cells', 3)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            
            return adata
            
        except Exception as e:
            logger.warning(f"QC failed: {e}. Using basic filtering only.")
            # Fallback to basic filtering
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            return adata
    
    def normalize_data(self, adata: ad.AnnData, dataset_info: Dict) -> ad.AnnData:
        """Normalize dataset"""
        # Store raw counts
        adata.raw = adata
        
        # Normalize to 10,000 reads per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        
        # Keep only highly variable genes
        adata = adata[:, adata.var.highly_variable]
        
        return adata
    
    def extract_features(self, adata: ad.AnnData, dataset_info: Dict) -> ad.AnnData:
        """Extract features for model training"""
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        # Add dataset metadata
        adata.obs['dataset_id'] = dataset_info['dataset_id']
        adata.obs['disease_status'] = dataset_info.get('disease_status', 'unknown')
        adata.obs['tissue_type'] = dataset_info.get('tissue_type', 'unknown')
        adata.obs['batch'] = self.current_batch
        
        return adata
    
    def integrate_batch(self, batch_data: List[ad.AnnData]) -> ad.AnnData:
        """Integrate multiple datasets in a batch"""
        if not batch_data:
            return None
        
        logger.info(f"Integrating {len(batch_data)} datasets in batch {self.current_batch + 1}")
        
        try:
            # Make sure all datasets have unique gene names
            for i, adata in enumerate(batch_data):
                adata.var_names_make_unique()
                adata.obs_names_make_unique()
            
            # Concatenate datasets with outer join to handle different gene sets
            integrated = ad.concat(batch_data, join='outer', index_unique='-', fill_value=0)
            
            # Batch correction (if needed)
            if len(batch_data) > 1:
                integrated = self.batch_correction(integrated)
            
            logger.info(f"Integrated batch: {integrated.n_obs} cells, {integrated.n_vars} genes")
            return integrated
            
        except Exception as e:
            logger.error(f"Batch integration failed: {e}")
            # Fallback: return the first dataset if integration fails
            if batch_data:
                logger.warning("Using first dataset as fallback")
                return batch_data[0]
            return None
    
    def batch_correction(self, adata: ad.AnnData) -> ad.AnnData:
        """Perform batch correction"""
        # Use Harmony or similar method
        # For now, simple scaling per batch
        for batch in adata.obs['batch'].unique():
            batch_mask = adata.obs['batch'] == batch
            adata.X[batch_mask] = adata.X[batch_mask] / adata.X[batch_mask].mean()
        
        return adata
    
    def update_model(self, integrated_data: ad.AnnData):
        """Update foundation model with new data"""
        logger.info(f"Updating model with {integrated_data.n_obs} cells")
        
        if self.model_state is None:
            # Initialize model
            self.model_state = self.initialize_model(integrated_data)
        else:
            # Update existing model
            self.model_state = self.incremental_training(integrated_data)
        
        # Save model checkpoint
        checkpoint_path = self.dirs['checkpoints'] / f"model_batch_{self.current_batch + 1}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model_state, f)
        
        logger.info(f"Model updated and saved to {checkpoint_path}")
    
    def initialize_model(self, data: ad.AnnData):
        """Initialize foundation model"""
        # Placeholder for model initialization
        # This would be replaced with actual model architecture
        model_state = {
            'n_genes': data.n_vars,
            'n_cells': data.n_obs,
            'gene_names': data.var_names.tolist(),
            'embeddings': np.random.randn(data.n_obs, 128),  # Placeholder
            'model_weights': None,  # Placeholder
            'batch_count': 1
        }
        return model_state
    
    def incremental_training(self, data: ad.AnnData):
        """Perform incremental training on new data"""
        # Update model state with new data
        self.model_state['n_cells'] += data.n_obs
        self.model_state['batch_count'] += 1
        
        # Placeholder for actual training
        # This would include:
        # - Update embeddings
        # - Fine-tune model weights
        # - Update gene representations
        
        return self.model_state
    
    def cleanup_batch(self):
        """Clean up processed data to free space"""
        logger.info("Cleaning up batch data")
        
        # Remove raw data
        if self.dirs['raw'].exists():
            shutil.rmtree(self.dirs['raw'])
            self.dirs['raw'].mkdir()
        
        # Remove processed data (keep only latest)
        processed_files = list(self.dirs['processed'].glob('*.h5ad'))
        if len(processed_files) > 1:
            # Keep only the most recent file
            latest_file = max(processed_files, key=os.path.getctime)
            for file in processed_files:
                if file != latest_file:
                    file.unlink()
        
        # Clean cache
        if self.dirs['cache'].exists():
            shutil.rmtree(self.dirs['cache'])
            self.dirs['cache'].mkdir()
        
        logger.info("Batch cleanup completed")
    
    def run_pipeline(self):
        """Run the complete streaming pipeline"""
        self.print_status("Starting pipeline", "Initializing streaming data processing")
        logger.info("Starting streaming pipeline")
        
        # Get total number of datasets for progress tracking
        datasets = self.load_dataset_list()
        total_datasets = len(datasets)
        batch_size = self.config['batch_size']
        total_batches = (total_datasets + batch_size - 1) // batch_size
        
        self.print_status("Pipeline initialized", f"Total: {total_datasets} datasets, {total_batches} batches")
        
        while True:
            # Check storage space
            self.print_status("Checking storage", "Monitoring disk usage")
            free_gb, used_gb = self.check_storage_space()
            if free_gb < 50:  # Keep 50GB free
                self.print_status("Low storage detected", "Cleaning up batch data")
                logger.warning("Low storage space, cleaning up")
                self.cleanup_batch()
            
            # Get next batch
            batch = self.get_next_batch(self.config['batch_size'])
            if not batch:
                self.print_status("Pipeline complete", "All datasets processed successfully")
                logger.info("No more datasets to process")
                break
            
            # Process batch
            self.print_status("Processing batch", f"Starting batch {self.current_batch + 1}/{total_batches}")
            batch_data = []
            
            for i, dataset_info in enumerate(batch):
                self.update_progress(
                    self.current_batch, total_batches,
                    i, len(batch)
                )
                
                # Download
                dataset_path = self.download_dataset(dataset_info)
                if not dataset_path:
                    continue  # Error already reported in download_dataset
                
                # Process
                processed_data = self.process_dataset(dataset_path, dataset_info)
                if processed_data is not None:
                    batch_data.append(processed_data)
                    self.print_status("Dataset processed", f"Added to batch integration", dataset_info['dataset_id'])
                # Processing errors already reported in process_dataset
            
            if batch_data:
                # Integrate batch
                self.print_status("Integrating batch", f"Combining {len(batch_data)} datasets")
                integrated_data = self.integrate_batch(batch_data)
                
                # Update model
                self.print_status("Updating model", "Incremental training on new data")
                self.update_model(integrated_data)
                
                self.print_status("Batch integration complete", f"Model updated with {integrated_data.n_obs} cells")
            else:
                self.print_status("Batch skipped", "No datasets successfully processed")
            
            # Cleanup
            self.print_status("Cleaning up", "Removing temporary files")
            self.cleanup_batch()
            
            # Move to next batch
            self.current_batch += 1
            
            self.print_status("Batch complete", f"Finished batch {self.current_batch}/{total_batches}")
            logger.info(f"Completed batch {self.current_batch}")
        
        self.print_status("Pipeline complete", "All processing finished successfully")
        
        # Print download tracking statistics
        self.print_download_statistics()
        
        # Print final issue summary
        self.console.print("\n" + "="*60)
        self.console.print("[bold green]FINAL ISSUE SUMMARY[/bold green]")
        self.console.print("="*60)
        self.print_issue_summary()
        
        logger.info("Pipeline completed successfully")
    
    def print_download_statistics(self):
        """Print download tracking statistics"""
        stats = self.download_tracker.get_statistics()
        
        self.console.print("\n" + "="*60)
        self.console.print("[bold blue]DOWNLOAD TRACKING STATISTICS[/bold blue]")
        self.console.print("="*60)
        
        # Create statistics table
        table = Table(title="Dataset Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="yellow")
        table.add_column("Percentage", style="green")
        
        table.add_row("Total Tracked", str(stats['total_tracked']), "100.0%")
        table.add_row("Downloaded", str(stats['downloaded']), f"{(stats['downloaded']/stats['total_tracked']*100):.1f}%" if stats['total_tracked'] > 0 else "0.0%")
        table.add_row("Processed", str(stats['processed']), f"{(stats['processed']/stats['total_tracked']*100):.1f}%" if stats['total_tracked'] > 0 else "0.0%")
        table.add_row("Failed", str(stats['failed']), f"{(stats['failed']/stats['total_tracked']*100):.1f}%" if stats['total_tracked'] > 0 else "0.0%")
        table.add_row("Success Rate", f"{stats['success_rate']:.1f}%", "")
        
        self.console.print(table)
        
        # Show tracker file location
        tracker_info = Panel(
            f"[bold]Download Tracker File:[/bold] {self.download_tracker.tracker_file}\n"
            f"[bold]Resume Capability:[/bold] ✅ Enabled\n"
            f"[bold]Next Run:[/bold] Will skip {stats['processed']} already processed datasets",
            title="[bold green]RESUME INFORMATION[/bold green]",
            border_style="green"
        )
        self.console.print(tracker_info)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python streaming_pipeline.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    pipeline = StreamingPipeline(config_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
