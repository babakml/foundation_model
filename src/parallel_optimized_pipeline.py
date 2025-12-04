#!/usr/bin/env python3
"""
Parallel Optimized Foundation Model Pipeline
- Parallel download and processing
- Optimized memory usage (utilize more than 1.68%)
- Enhanced nested archive handling
- Synapse, GEO, and SRA download support
- No email addresses
- Faster execution
"""

import os
import sys
import json
import time
import shutil
import logging
import threading
import queue
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
import psutil
import configparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parallel_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the parallel pipeline"""
    base_dir: str
    dataset_list_path: str
    max_download_threads: int = 4
    max_processing_threads: int = 8
    memory_limit_gb: int = 200  # Use more memory
    batch_size: int = 1
    download_timeout: int = 3600
    processing_timeout: int = 7200
    quality_control: Dict = None
    normalization: Dict = None
    storage_management: Dict = None

class DownloadTracker:
    """Enhanced download tracker with thread safety"""
    
    def __init__(self, tracker_path: str):
        self.tracker_path = Path(tracker_path)
        self.lock = threading.Lock()
        self.data = self.load_tracker()
    
    def load_tracker(self) -> Dict:
        """Load existing tracker or create new one"""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load tracker: {e}")
        return {}
    
    def save_tracker(self):
        """Save tracker with thread safety"""
        with self.lock:
            try:
                with open(self.tracker_path, 'w') as f:
                    json.dump(self.data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save tracker: {e}")
    
    def is_downloaded(self, dataset_id: str) -> bool:
        """Check if dataset is fully downloaded and processed"""
        with self.lock:
            return self.data.get(dataset_id, {}).get('processing_complete', False)
    
    def is_downloaded_only(self, dataset_id: str) -> bool:
        """Check if dataset is downloaded but not processed"""
        with self.lock:
            entry = self.data.get(dataset_id, {})
            return entry.get('download_complete', False) and not entry.get('processing_complete', False)
    
    def mark_downloaded(self, dataset_id: str, dataset_path: str, data_type: str):
        """Mark dataset as downloaded"""
        with self.lock:
            if dataset_id not in self.data:
                self.data[dataset_id] = {}
            self.data[dataset_id].update({
                'download_complete': True,
                'dataset_path': dataset_path,
                'data_type': data_type,
                'download_timestamp': datetime.now().isoformat()
            })
        self.save_tracker()
    
    def mark_processed(self, dataset_id: str, processed_path: str, n_cells: int, n_genes: int):
        """Mark dataset as processed"""
        with self.lock:
            if dataset_id not in self.data:
                self.data[dataset_id] = {}
            self.data[dataset_id].update({
                'processing_complete': True,
                'processed_path': processed_path,
                'n_cells': n_cells,
                'n_genes': n_genes,
                'processing_timestamp': datetime.now().isoformat()
            })
        self.save_tracker()
    
    def mark_failed(self, dataset_id: str, error: str):
        """Mark dataset as failed"""
        with self.lock:
            if dataset_id not in self.data:
                self.data[dataset_id] = {}
            self.data[dataset_id].update({
                'failed': True,
                'error': error,
                'failure_timestamp': datetime.now().isoformat()
            })
        self.save_tracker()

class ParallelOptimizedPipeline:
    """Parallel optimized pipeline for foundation model"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.console = Console()
        self.setup_directories()
        self.download_tracker = DownloadTracker(self.config['base_dir'] + '/download_tracker.json')
        
        # Threading components
        self.download_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.status_lock = threading.Lock()
        self.status_info = {
            'downloading': 0,
            'processing': 0,
            'completed': 0,
            'failed': 0,
            'current_download': '',
            'current_processing': ''
        }
        
        # Performance monitoring
        self.start_time = time.time()
        self.processed_count = 0
        
        logger.info("🚀 Parallel Optimized Pipeline initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_directories(self):
        """Setup directory structure"""
        base_dir = Path(self.config['base_dir'])
        self.dirs = {
            'base': base_dir,
            'raw': base_dir / 'data' / 'raw',
            'processed': base_dir / 'data' / 'processed',
            'logs': base_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_dataset_list(self) -> List[Dict]:
        """Load dataset list from CSV"""
        try:
            df = pd.read_csv(
                self.config['dataset_list_path'], 
                sep=';;', 
                engine='python', 
                encoding='latin-1',
                error_bad_lines=False,
                warn_bad_lines=False
            )
            # Clean up column names (remove trailing ;;)
            df.columns = df.columns.str.rstrip(';').str.strip()
            # Remove unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Ensure synapse_id column exists (even if empty)
            if 'synapse_id' not in df.columns:
                df['synapse_id'] = ''
            
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to load dataset list: {e}")
            return []
    
    def categorize_datasets(self, datasets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Categorize datasets into download and processing queues"""
        download_queue = []
        processing_queue = []
        
        for dataset in datasets:
            dataset_id = dataset['dataset_id']
            
            if self.download_tracker.is_downloaded(dataset_id):
                # Skip fully processed datasets
                continue
            elif self.download_tracker.is_downloaded_only(dataset_id):
                # Add to processing queue
                processing_queue.append(dataset)
            else:
                # Add to download queue
                download_queue.append(dataset)
        
        logger.info(f"📊 Dataset categorization:")
        logger.info(f"  - To download: {len(download_queue)}")
        logger.info(f"  - To process: {len(processing_queue)}")
        logger.info(f"  - Already completed: {len(datasets) - len(download_queue) - len(processing_queue)}")
        
        return download_queue, processing_queue
    
    def download_dataset_worker(self, dataset_info: Dict) -> bool:
        """Worker function for downloading datasets"""
        dataset_id = dataset_info['dataset_id']
        data_type = dataset_info.get('data_type', '')
        synapse_id = dataset_info.get('synapse_id', '').strip() if dataset_info.get('synapse_id') else None
        
        try:
            with self.status_lock:
                self.status_info['downloading'] += 1
                self.status_info['current_download'] = dataset_id
            
            logger.info(f"📥 Starting download: {dataset_id}")
            
            # Create dataset directory
            dataset_dir = self.dirs['raw'] / dataset_id
            dataset_dir.mkdir(exist_ok=True)
            
            # Try different download strategies
            success = False
            
            # Strategy 1: Try Synapse download first (if synapse_id is available)
            if synapse_id and synapse_id.lower().startswith('syn'):
                logger.info(f"🔗 Synapse ID found for {dataset_id}: {synapse_id}")
                success = self.download_synapse_dataset(synapse_id, dataset_dir, dataset_id)
            
            # Strategy 2: Try GEO download
            if not success:
                success = self.download_geo_dataset(dataset_id, dataset_dir, data_type)
            
            # Strategy 3: Try SRA download as fallback
            if not success:
                success = self.download_sra_dataset(dataset_id, dataset_dir, data_type)
            
            if success:
                self.download_tracker.mark_downloaded(dataset_id, str(dataset_dir), data_type)
                logger.info(f"✅ Download completed: {dataset_id}")
                return True
            else:
                self.download_tracker.mark_failed(dataset_id, "Download failed")
                logger.error(f"❌ Download failed: {dataset_id}")
                return False
                
        except Exception as e:
            self.download_tracker.mark_failed(dataset_id, str(e))
            logger.error(f"❌ Download error for {dataset_id}: {e}")
            return False
        finally:
            with self.status_lock:
                self.status_info['downloading'] -= 1
                if self.status_info['current_download'] == dataset_id:
                    self.status_info['current_download'] = ''
    
    def download_geo_dataset(self, dataset_id: str, output_dir: Path, data_type: str) -> bool:
        """Download dataset from GEO using FTP"""
        try:
            # Use FTP for faster downloads
            ftp_url = f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{dataset_id[:6]}nnn/{dataset_id}/suppl/"
            
            # Try to download files
            cmd = [
                'wget', '-r', '-nH', '--cut-dirs=4', '--no-parent',
                '--reject="index.html*"', '--timeout=300', '--tries=3',
                ftp_url, '-P', str(output_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                # Check if we got any files
                files = list(output_dir.rglob('*'))
                if len(files) > 1:  # More than just the directory itself
                    logger.info(f"✅ GEO download successful: {dataset_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"GEO download failed for {dataset_id}: {e}")
            return False
    
    def download_synapse_dataset(self, synapse_id: str, output_dir: Path, dataset_id: str) -> bool:
        """Download dataset from Synapse"""
        try:
            import synapseclient
            from synapseclient import Synapse
            
            # Initialize Synapse client (reuse if available, otherwise create new)
            if not hasattr(self, '_synapse_client'):
                self._synapse_client = self._init_synapse_client()
            
            if not self._synapse_client:
                logger.warning(f"Synapse client not available for {dataset_id}")
                return False
            
            logger.info(f"📥 Downloading from Synapse: {synapse_id} for {dataset_id}")
            
            # Get entity info first
            entity = self._synapse_client.get(synapse_id, downloadFile=False)
            entity_type = entity.concreteType if hasattr(entity, 'concreteType') else 'unknown'
            
            # If it's a folder or project, download recursively
            if 'Folder' in entity_type or 'Project' in entity_type:
                # Download to dataset directory
                self._synapse_client.get(synapse_id, downloadLocation=str(output_dir), recursive=True)
                
                # Verify download
                files = list(output_dir.rglob('*'))
                if len(files) > 1:  # More than just the directory itself
                    logger.info(f"✅ Synapse download successful: {dataset_id} ({len(files)} files)")
                    return True
            else:
                # It's a file, download it
                self._synapse_client.get(synapse_id, downloadLocation=str(output_dir))
                if (output_dir / (entity.name if hasattr(entity, 'name') else synapse_id)).exists():
                    logger.info(f"✅ Synapse download successful: {dataset_id}")
                    return True
            
            return False
            
        except ImportError:
            logger.warning(f"synapseclient not installed for {dataset_id}")
            return False
        except Exception as e:
            logger.warning(f"Synapse download failed for {dataset_id}: {e}")
            return False
    
    def _init_synapse_client(self):
        """Initialize Synapse client with credentials"""
        try:
            import synapseclient
            from synapseclient import Synapse
            
            # Read credentials from config file
            config_path = Path.home() / '.synapseConfig'
            if config_path.exists():
                config = configparser.ConfigParser()
                config.read(config_path)
                # Try different profiles
                for profile_name in ['profile j', 'default']:
                    if profile_name in config:
                        username = config[profile_name].get('username')
                        authtoken = config[profile_name].get('authtoken')
                        if username and authtoken:
                            syn = Synapse()
                            syn.login(authToken=authtoken, silent=True)
                            logger.info("✅ Synapse client initialized")
                            return syn
                # Try any other profile
                for section in config.sections():
                    if section.startswith('profile ') or section == 'default':
                        username = config[section].get('username')
                        authtoken = config[section].get('authtoken')
                        if username and authtoken:
                            syn = Synapse()
                            syn.login(authToken=authtoken, silent=True)
                            logger.info("✅ Synapse client initialized")
                            return syn
            # Fallback to default login
            syn = Synapse()
            syn.login(silent=True)
            logger.info("✅ Synapse client initialized")
            return syn
        except ImportError:
            logger.warning("synapseclient not installed. Synapse downloads will be skipped.")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize Synapse client: {e}")
            return None
    
    def download_sra_dataset(self, dataset_id: str, output_dir: Path, data_type: str) -> bool:
        """Download dataset from SRA using enhanced methods"""
        try:
            # Import the SRA downloader
            from sra_download_fix import SRADownloader
            
            # Create SRA downloader
            sra_downloader = SRADownloader(output_dir.parent)
            
            # Try to download SRA data
            success = sra_downloader.download_sra_data(dataset_id)
            
            if success:
                logger.info(f"✅ SRA download successful: {dataset_id}")
                return True
            
            return False
            
        except ImportError:
            logger.warning(f"SRA downloader not available for {dataset_id}")
            return False
        except Exception as e:
            logger.warning(f"SRA download failed for {dataset_id}: {e}")
            return False
    
    def process_dataset_worker(self, dataset_info: Dict) -> bool:
        """Worker function for processing datasets"""
        dataset_id = dataset_info['dataset_id']
        
        try:
            with self.status_lock:
                self.status_info['processing'] += 1
                self.status_info['current_processing'] = dataset_id
            
            logger.info(f"🔄 Starting processing: {dataset_id}")
            
            # Find the dataset directory
            dataset_dir = self.dirs['raw'] / dataset_id
            if not dataset_dir.exists():
                logger.error(f"❌ Dataset directory not found: {dataset_id}")
                return False
            
            # Process the dataset
            adata = self.process_dataset(str(dataset_dir), dataset_info)
            
            if adata is not None:
                # Save processed data
                processed_path = self.dirs['processed'] / f"{dataset_id}_processed.h5ad"
                adata.write_h5ad(processed_path)
                
                # Update tracker
                self.download_tracker.mark_processed(dataset_id, str(processed_path), adata.n_obs, adata.n_vars)
                
                logger.info(f"✅ Processing completed: {dataset_id} ({adata.n_obs} cells, {adata.n_vars} genes)")
                
                with self.status_lock:
                    self.status_info['completed'] += 1
                    self.processed_count += 1
                
                return True
            else:
                logger.error(f"❌ Processing failed: {dataset_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Processing error for {dataset_id}: {e}")
            return False
        finally:
            with self.status_lock:
                self.status_info['processing'] -= 1
                if self.status_info['current_processing'] == dataset_id:
                    self.status_info['current_processing'] = ''
    
    def process_dataset(self, dataset_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Process a single dataset with enhanced nested archive support"""
        dataset_id = dataset_info['dataset_id']
        data_type = dataset_info.get('data_type', '')
        
        try:
            # Extract nested archives first
            self.extract_nested_archives(Path(dataset_path))
            
            # Find data files
            data_files = self.find_data_files(dataset_path, data_type)
            
            if not data_files:
                logger.warning(f"No data files found for {dataset_id}")
                return None
            
            # Load the first valid data file
            for file_path in data_files:
                adata = self.load_data_file(file_path, dataset_info)
                if adata is not None:
                    # Apply quality control
                    adata = self.apply_quality_control(adata, dataset_info)
                    
                    # Apply normalization
                    adata = self.apply_normalization(adata)
                    
                    return adata
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_id}: {e}")
            return None
    
    def extract_nested_archives(self, dataset_dir: Path):
        """Extract nested archives (.tar.gz files inside tar files)"""
        logger.info(f"🔍 Checking for nested archives in {dataset_dir}")
        
        # Find all .tar.gz files
        tar_gz_files = list(dataset_dir.rglob("*.tar.gz"))
        
        for tar_gz_file in tar_gz_files:
            try:
                # Create extraction directory
                extract_dir = tar_gz_file.parent / f"{tar_gz_file.stem}_extracted"
                extract_dir.mkdir(exist_ok=True)
                
                logger.info(f"📦 Extracting nested archive: {tar_gz_file.name}")
                
                # Extract .tar.gz file
                import tarfile
                with tarfile.open(tar_gz_file, 'r:gz') as tar:
                    tar.extractall(extract_dir)
                
                logger.info(f"✅ Successfully extracted {tar_gz_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to extract {tar_gz_file}: {e}")
                continue
    
    def find_data_files(self, dataset_path: str, data_type: str) -> List[str]:
        """Find data files with enhanced pattern matching"""
        dataset_dir = Path(dataset_path)
        data_files = []
        
        # Define comprehensive file patterns
        file_patterns = {
            'H5': ['*.h5', '*.h5ad', '*feature_bc_matrix.h5', '*filtered_feature_bc_matrix.h5'],
            'H5AD': ['*.h5ad', '*.h5'],
            'MTX': ['matrix.mtx*', '*.mtx', '*.mtx.gz', 'matrix.mtx.gz'],
            'TSV': ['*.tsv', '*.tsv.gz', '*.txt', '*.txt.gz'],
            'CSV': ['*.csv', '*.csv.gz'],
            'RDS': ['*.rds'],
            'LOOM': ['*.loom']
        }
        
        # Look for files based on data_type
        for data_type_key in file_patterns.keys():
            if data_type_key.upper() in data_type.upper():
                for pattern in file_patterns[data_type_key]:
                    data_files.extend(list(dataset_dir.rglob(pattern)))
        
        # If no specific files found, look for common formats
        if not data_files:
            common_formats = ['*.h5ad', '*.h5', '*.loom', '*.rds', '*.csv', '*.tsv', '*.mtx']
            for root, dirs, files in os.walk(dataset_dir):
                root_path = Path(root)
                for pattern in common_formats:
                    data_files.extend(list(root_path.glob(pattern)))
        
        # Look for 10X directories
        tenx_dirs = self.find_10x_directories(dataset_dir)
        data_files.extend([str(d) for d in tenx_dirs])
        
        # Remove duplicates and prioritize (normalize to strings)
        data_files = list({str(p) for p in data_files})
        return self.prioritize_data_files(data_files)
    
    def find_10x_directories(self, dataset_dir: Path) -> List[Path]:
        """Find directories containing 10X Genomics matrix files"""
        tenx_dirs = []
        
        for root, dirs, files in os.walk(dataset_dir):
            root_path = Path(root)
            
            # Check for 10X matrix files
            has_matrix = any(f.startswith('matrix.mtx') for f in files)
            has_barcodes = any(f.startswith('barcodes.tsv') for f in files)
            has_features = any(f.startswith('features.tsv') or f.startswith('genes.tsv') for f in files)
            
            if has_matrix and (has_barcodes or has_features):
                tenx_dirs.append(root_path)
                logger.info(f"🔍 Found 10X directory: {root_path}")
        
        return tenx_dirs
    
    def prioritize_data_files(self, data_files: List[str]) -> List[str]:
        """Prioritize data files by format preference (robust to Path objects)"""
        priority_order = ['.h5ad', '.h5', '.loom', '.rds', '.csv', '.tsv', '.mtx']

        # Ensure string paths
        str_files = [str(fp) for fp in data_files]

        def get_priority(file_path_str: str) -> int:
            lower_path = file_path_str.lower()
            for i, ext in enumerate(priority_order):
                if ext in lower_path:
                    return i
            return len(priority_order)

        return sorted(str_files, key=get_priority)
    
    def load_data_file(self, file_path: str, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Load data from various file formats"""
        file_path = Path(file_path)
        
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
                return sc.read_10x_mtx(file_path.parent)
            elif file_path.suffix == '.gz':
                return self.read_compressed_file(file_path, dataset_info)
            elif file_path.is_dir():
                # Handle 10X directories
                return sc.read_10x_mtx(str(file_path))
            else:
                logger.warning(f"Unknown file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None
    
    def read_compressed_file(self, file_path: Path, dataset_info: Dict) -> Optional[ad.AnnData]:
        """Read compressed files (.gz)"""
        try:
            import gzip
            import pandas as pd
            import numpy as np
            from scipy import sparse
            
            # Handle compressed MTX files
            if 'matrix.mtx' in file_path.name:
                return self.read_compressed_mtx(file_path)
            
            # Handle compressed CSV/TXT files
            elif file_path.suffix == '.gz' and any(ext in file_path.name.lower() for ext in ['.csv', '.tsv', '.txt']):
                return self.read_compressed_matrix(file_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read compressed file {file_path}: {e}")
            return None
    
    def read_compressed_mtx(self, file_path: Path) -> Optional[ad.AnnData]:
        """Read compressed MTX files"""
        try:
            import gzip
            import pandas as pd
            import numpy as np
            from scipy import sparse
            
            # Read the compressed MTX file
            with gzip.open(file_path, 'rt') as f:
                # Skip header lines
                for line in f:
                    if line.startswith('%'):
                        continue
                    else:
                        rows, cols, entries = map(int, line.strip().split())
                        break
                
                # Read the data
                data = []
                row_indices = []
                col_indices = []
                
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        row_idx = int(parts[0]) - 1
                        col_idx = int(parts[1]) - 1
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
            self.add_barcodes_and_features(adata, dataset_dir)
            
            return adata
            
        except Exception as e:
            logger.error(f"Failed to read compressed MTX {file_path}: {e}")
            return None
    
    def read_compressed_matrix(self, file_path: Path) -> Optional[ad.AnnData]:
        """Read compressed CSV/TXT files"""
        try:
            import gzip
            import pandas as pd
            
            # Determine separator
            if '.csv' in file_path.name.lower():
                sep = ','
            elif '.tsv' in file_path.name.lower():
                sep = '\t'
            else:
                sep = '\t'
            
            # Read compressed file
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f, sep=sep, index_col=0)
                
                if df.shape[0] > 10 and df.shape[1] > 10:
                    # Transpose to get cells as rows, genes as columns
                    adata = ad.AnnData(
                        X=df.T.values,
                        var=pd.DataFrame(index=df.T.columns),
                        obs=pd.DataFrame(index=df.T.index)
                    )
                    return adata
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read compressed matrix {file_path}: {e}")
            return None
    
    def add_barcodes_and_features(self, adata: ad.AnnData, dataset_dir: Path):
        """Add barcodes and features to AnnData object"""
        try:
            # Look for barcodes file
            barcodes_file = None
            for pattern in ['barcodes.tsv', 'barcodes.tsv.gz', 'barcodes.txt', 'barcodes.txt.gz']:
                candidates = list(dataset_dir.glob(f"*{pattern}"))
                if candidates:
                    barcodes_file = candidates[0]
                    break
            
            # Look for features file
            features_file = None
            for pattern in ['features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz']:
                candidates = list(dataset_dir.glob(f"*{pattern}"))
                if candidates:
                    features_file = candidates[0]
                    break
            
            # Load barcodes
            if barcodes_file:
                if barcodes_file.suffix == '.gz':
                    import gzip
                    with gzip.open(barcodes_file, 'rt') as f:
                        barcodes = [line.strip() for line in f]
                else:
                    with open(barcodes_file, 'r') as f:
                        barcodes = [line.strip() for line in f]
                adata.obs_names = barcodes
            
            # Load features
            if features_file:
                if features_file.suffix == '.gz':
                    import gzip
                    with gzip.open(features_file, 'rt') as f:
                        features_df = pd.read_csv(f, sep='\t', header=None)
                else:
                    features_df = pd.read_csv(features_file, sep='\t', header=None)
                
                if features_df.shape[1] >= 2:
                    adata.var_names = features_df.iloc[:, 1].astype(str)
                    adata.var['gene_ids'] = features_df.iloc[:, 0].astype(str)
                else:
                    adata.var_names = features_df.iloc[:, 0].astype(str)
            
        except Exception as e:
            logger.warning(f"Failed to add barcodes/features: {e}")
    
    def read_rds_file(self, file_path: Path) -> Optional[ad.AnnData]:
        """Read RDS files using rpy2"""
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            
            # Load RDS file
            ro.r(f'data <- readRDS("{file_path}")')
            
            # Convert to pandas DataFrame
            df = ro.r('as.data.frame(data)')
            
            # Convert to AnnData
            adata = ad.AnnData(X=df.values)
            adata.obs_names = df.index
            adata.var_names = df.columns
            
            return adata
            
        except Exception as e:
            logger.error(f"Failed to read RDS file {file_path}: {e}")
            return None
    
    def read_matrix_file(self, file_path: Path) -> Optional[ad.AnnData]:
        """Read CSV/TSV matrix files"""
        try:
            # Determine separator
            if file_path.suffix == '.csv':
                sep = ','
            else:
                sep = '\t'
            
            # Read file
            df = pd.read_csv(file_path, sep=sep, index_col=0)
            
            # Convert to AnnData
            adata = ad.AnnData(X=df.values)
            adata.obs_names = df.index
            adata.var_names = df.columns
            
            return adata
            
        except Exception as e:
            logger.error(f"Failed to read matrix file {file_path}: {e}")
            return None
    
    def apply_quality_control(self, adata: ad.AnnData, dataset_info: Dict) -> ad.AnnData:
        """Apply quality control filters"""
        try:
            # Get QC parameters from config
            qc_config = self.config.get('quality_control', {})
            min_genes = qc_config.get('min_genes', 200)
            max_genes = qc_config.get('max_genes', 5000)
            max_mt_percent = qc_config.get('max_mt_percent', 20)
            min_cells = qc_config.get('min_cells', 3)
            
            # Calculate QC metrics
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
            
            # Filter cells
            sc.pp.filter_cells(adata, min_genes=min_genes)
            sc.pp.filter_genes(adata, min_cells=min_cells)
            
            # Filter by mitochondrial percentage
            if 'pct_counts_mt' in adata.obs.columns:
                adata = adata[adata.obs.pct_counts_mt < max_mt_percent, :]
            
            # Filter by gene count
            adata = adata[adata.obs.n_genes_by_counts < max_genes, :]
            
            logger.info(f"QC applied: {adata.n_obs} cells, {adata.n_vars} genes")
            return adata
            
        except Exception as e:
            logger.error(f"QC failed: {e}")
            return adata
    
    def apply_normalization(self, adata: ad.AnnData) -> ad.AnnData:
        """Apply normalization"""
        try:
            # Get normalization parameters from config
            norm_config = self.config.get('normalization', {})
            target_sum = norm_config.get('target_sum', 10000)
            log_transform = norm_config.get('log_transform', True)
            scale = norm_config.get('scale', True)
            max_value = norm_config.get('max_value', 10)
            
            # Normalize
            sc.pp.normalize_total(adata, target_sum=target_sum)
            
            if log_transform:
                sc.pp.log1p(adata)
            
            if scale:
                sc.pp.scale(adata, max_value=max_value)
            
            return adata
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return adata
    
    def print_status(self):
        """Print current status"""
        with self.status_lock:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            
            status_text = f"""
🚀 PARALLEL OPTIMIZED PIPELINE STATUS
=====================================
⏱️  Runtime: {elapsed/3600:.1f}h
📊 Processed: {self.processed_count} datasets
⚡ Rate: {rate:.2f} datasets/hour

📥 DOWNLOADING: {self.status_info['downloading']}
   Current: {self.status_info['current_download'] or 'None'}

🔄 PROCESSING: {self.status_info['processing']}
   Current: {self.status_info['current_processing'] or 'None'}

✅ COMPLETED: {self.status_info['completed']}
❌ FAILED: {self.status_info['failed']}

💾 Memory Usage: {psutil.virtual_memory().percent:.1f}%
            """
            
            self.console.print(Panel(status_text, title="Pipeline Status", border_style="blue"))
    
    def run_pipeline(self):
        """Run the parallel optimized pipeline"""
        logger.info("🚀 Starting Parallel Optimized Pipeline")
        
        # Load dataset list
        datasets = self.load_dataset_list()
        if not datasets:
            logger.error("No datasets to process")
            return
        
        # Categorize datasets
        download_queue, processing_queue = self.categorize_datasets(datasets)
        
        # Start status display thread
        status_thread = threading.Thread(target=self.status_display_loop, daemon=True)
        status_thread.start()
        
        # Run parallel processing
        with ThreadPoolExecutor(max_workers=self.config.get('max_download_threads', 4) + 
                               self.config.get('max_processing_threads', 8)) as executor:
            
            # Submit download tasks
            download_futures = []
            for dataset in download_queue:
                future = executor.submit(self.download_dataset_worker, dataset)
                download_futures.append(future)
            
            # Submit processing tasks
            processing_futures = []
            for dataset in processing_queue:
                future = executor.submit(self.process_dataset_worker, dataset)
                processing_futures.append(future)
            
            # Wait for completion
            all_futures = download_futures + processing_futures
            
            for future in as_completed(all_futures):
                try:
                    result = future.result()
                    if result:
                        logger.info("✅ Task completed successfully")
                    else:
                        logger.error("❌ Task failed")
                except Exception as e:
                    logger.error(f"❌ Task error: {e}")
        
        # Final status
        self.print_status()
        logger.info("🎉 Pipeline completed!")
    
    def status_display_loop(self):
        """Status display loop"""
        while True:
            self.print_status()
            time.sleep(30)  # Update every 30 seconds

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python parallel_optimized_pipeline.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    pipeline = ParallelOptimizedPipeline(config_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
