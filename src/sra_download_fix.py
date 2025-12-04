#!/usr/bin/env python3
"""
SRA Download Fix for ALS Foundation Model Pipeline
Based on NCBI SRA documentation: https://www.ncbi.nlm.nih.gov/sra/docs/sradownload/
"""

import os
import sys
import json
import time
import logging
import subprocess
import requests
import time
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class SRADownloader:
    """Enhanced SRA downloader with multiple fallback methods"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.sra_toolkit_available = self.check_sra_toolkit()
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.api_key = os.environ.get("NCBI_API_KEY")
        # Basic rate limit: default 3 req/sec without API key, 10 req/sec with API key
        self.min_interval = 0.35 if not self.api_key else 0.1
        self._last_request_ts = 0.0

    # Global semaphore to limit concurrent Entrez requests across threads
    _entrez_sem = threading.Semaphore(2)

    def _entrez_get(self, endpoint: str, params: Dict, max_retries: int = 5) -> requests.Response:
        """Entrez GET with rate limiting and exponential backoff (handles 429)."""
        url = f"{self.base_url}{endpoint}"
        if self.api_key:
            params = {**params, 'api_key': self.api_key}
        backoff = 1.0
        for attempt in range(max_retries):
            # Rate limit
            now = time.time()
            sleep_needed = self.min_interval - (now - self._last_request_ts)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            try:
                # Limit concurrency
                with SRADownloader._entrez_sem:
                    resp = requests.get(url, params=params, timeout=30)
                self._last_request_ts = time.time()
                if resp.status_code == 429:
                    # Too many requests
                    logger.warning("Entrez 429 rate limit hit; backing off")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
        raise RuntimeError("Unreachable: exhausted retries without raising")
        
    def check_sra_toolkit(self) -> bool:
        """Check if SRA toolkit is available"""
        try:
            result = subprocess.run(['prefetch', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ SRA Toolkit is available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("❌ SRA Toolkit not found - will use alternative methods")
        return False
    
    def search_sra_data(self, dataset_id: str) -> List[str]:
        """Search for SRA data using Entrez API and return SRR accessions."""
        try:
            # Search for SRA data
            params = {
                'db': 'sra',
                'term': dataset_id,
                'retmode': 'json',
                'retmax': 100
            }
            response = self._entrez_get('esearch.fcgi', params)
            data = response.json()
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                uid_list = data['esearchresult']['idlist']
                # Resolve UIDs to SRR accessions via efetch (XML)
                srr_accessions: List[str] = []
                for uid in uid_list:
                    try:
                        efetch_params = {'db': 'sra', 'id': uid, 'retmode': 'xml'}
                        efetch_resp = self._entrez_get('efetch.fcgi', efetch_params)
                        xml_text = efetch_resp.text
                        # Extract SRR accessions from RUN accession attributes
                        srrs = re.findall(r'RUN[^>]*accession="(SRR\d+)"', xml_text)
                        if srrs:
                            srr_accessions.extend(srrs)
                    except Exception as _e:
                        logger.warning(f"Failed to efetch UID {uid}: {_e}")
                        continue
                # Deduplicate, keep order
                seen = set()
                srr_list = [s for s in srr_accessions if not (s in seen or seen.add(s))]
                logger.info(f"Resolved {len(srr_list)} SRR runs for {dataset_id}")
                return srr_list
            return []
            
        except Exception as e:
            logger.warning(f"Failed to search SRA data for {dataset_id}: {e}")
            return []
    
    def get_run_info(self, run_id: str) -> Optional[Dict]:
        """Get metadata for a specific SRA run"""
        try:
            # Get run info
            params = {
                'db': 'sra',
                'id': run_id,
                'retmode': 'xml'
            }
            response = self._entrez_get('efetch.fcgi', params)
            # Parse XML to extract basic info
            # This is a simplified version - in practice you'd use proper XML parsing
            content = response.text
            if 'RUN' in content and 'accession' in content:
                return {
                    'run_id': run_id,
                    'size': self.extract_size_from_xml(content),
                    'platform': self.extract_platform_from_xml(content)
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get run info for {run_id}: {e}")
            return None
    
    def extract_size_from_xml(self, xml_content: str) -> str:
        """Extract size information from SRA XML"""
        # Simplified extraction - in practice use proper XML parsing
        if 'size="' in xml_content:
            start = xml_content.find('size="') + 6
            end = xml_content.find('"', start)
            return xml_content[start:end]
        return "unknown"
    
    def extract_platform_from_xml(self, xml_content: str) -> str:
        """Extract platform information from SRA XML"""
        # Simplified extraction - in practice use proper XML parsing
        if 'PLATFORM' in xml_content:
            if 'ILLUMINA' in xml_content:
                return 'ILLUMINA'
            elif 'PACBIO' in xml_content:
                return 'PACBIO'
            elif 'OXFORD_NANOPORE' in xml_content:
                return 'OXFORD_NANOPORE'
        return 'unknown'
    
    def download_with_sra_toolkit(self, run_id: str, output_dir: Path) -> bool:
        """Download SRA data using SRA toolkit"""
        try:
            # Create output directory
            run_dir = output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Use prefetch to download
            cmd = ['prefetch', run_id, '--output-directory', str(run_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                logger.info(f"✅ SRA download successful: {run_id}")
                
                # Convert to FASTQ if needed
                sra_file = run_dir / f"{run_id}.sra"
                if sra_file.exists():
                    self.convert_sra_to_fastq(sra_file, run_dir)
                
                return True
            else:
                logger.error(f"❌ SRA download failed: {run_id} - {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ SRA download timeout: {run_id}")
            return False
        except Exception as e:
            logger.error(f"❌ SRA download error: {run_id} - {e}")
            return False
    
    def convert_sra_to_fastq(self, sra_file: Path, output_dir: Path):
        """Convert SRA file to FASTQ format"""
        try:
            cmd = ['fasterq-dump', '--split-files', '--outdir', str(output_dir), str(sra_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✅ SRA to FASTQ conversion successful: {sra_file.name}")
            else:
                logger.warning(f"⚠️ SRA to FASTQ conversion failed: {sra_file.name} - {result.stderr}")
                
        except Exception as e:
            logger.warning(f"⚠️ SRA to FASTQ conversion error: {e}")
    
    def download_with_run_browser(self, run_id: str, output_dir: Path) -> bool:
        """Download SRA data using Run Browser (HTTP) - limited to <5 Gbases"""
        try:
            # Check if run is small enough for HTTP download
            run_info = self.get_run_info(run_id)
            if not run_info:
                return False
            
            # For now, we'll skip HTTP download as it's limited
            # In practice, you'd implement the Run Browser API calls here
            logger.info(f"ℹ️ Run Browser download not implemented for {run_id}")
            return False
            
        except Exception as e:
            logger.warning(f"Run Browser download failed for {run_id}: {e}")
            return False
    
    def download_sra_data(self, dataset_id: str) -> bool:
        """Main method to download SRA data with multiple fallback strategies"""
        logger.info(f"🔍 Searching for SRA data: {dataset_id}")
        
        # Search for SRA runs - this returns SRR accessions
        srr_accessions = self.search_sra_data(dataset_id)
        if not srr_accessions:
            logger.info(f"ℹ️ No SRA data found for {dataset_id}")
            return False
        
        # Create dataset directory
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        
        # Use SRR accessions (not numeric UIDs) for downloads
        for srr_id in srr_accessions[:3]:
            logger.info(f"📥 Downloading SRA run: {srr_id}")
            
            # Try SRA toolkit first
            if self.sra_toolkit_available:
                if self.download_with_sra_toolkit(srr_id, dataset_dir):
                    success_count += 1
                    continue
            
            # Fallback to Run Browser
            if self.download_with_run_browser(srr_id, dataset_dir):
                success_count += 1
                continue
            
            logger.warning(f"⚠️ All download methods failed for {srr_id}")
        
        if success_count > 0:
            logger.info(f"✅ Successfully downloaded {success_count}/{len(srr_accessions)} SRA runs for {dataset_id}")
            return True
        else:
            logger.error(f"❌ Failed to download any SRA data for {dataset_id}")
            return False

def install_sra_toolkit_instructions():
    """Provide instructions for installing SRA toolkit"""
    instructions = """
    🔧 SRA TOOLKIT INSTALLATION INSTRUCTIONS
    ========================================
    
    To fix the SRA download issues, you need to install the SRA Toolkit on the cluster:
    
    1. Download SRA Toolkit:
       wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-centos_linux64.tar.gz
    
    2. Extract:
       tar -xzf sratoolkit.current-centos_linux64.tar.gz
    
    3. Configure:
       cd sratoolkit.current-centos_linux64
       ./bin/vdb-config --interactive
    
    4. Add to PATH:
       export PATH=$PATH:$(pwd)/bin
    
    5. Test:
       prefetch --version
    
    Alternative: Use conda to install:
    conda install -c bioconda sra-tools
    
    """
    return instructions

def main():
    """Test the SRA downloader"""
    output_dir = Path("test_sra_downloads")
    downloader = SRADownloader(output_dir)
    
    # Test with a known SRA dataset
    test_dataset = "SRP123456"  # Replace with actual dataset ID
    success = downloader.download_sra_data(test_dataset)
    
    if success:
        print("✅ SRA download test successful!")
    else:
        print("❌ SRA download test failed!")
        print(install_sra_toolkit_instructions())

if __name__ == "__main__":
    main()
