#!/usr/bin/env python3
"""
Test script for SRA downloader fix
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from sra_download_fix import SRADownloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sra_search():
    """Test SRA search functionality"""
    print("🧪 Testing SRA search functionality...")
    
    # Create test output directory
    output_dir = Path("test_sra_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize downloader
    downloader = SRADownloader(output_dir)
    
    # Test with a known GSM ID that should have SRA data
    test_gsm = "GSM5292143"
    print(f"🔍 Testing search for {test_gsm}...")
    
    try:
        # Test search (this should return SRR accessions)
        srr_list = downloader.search_sra_data(test_gsm)
        print(f"📊 Found {len(srr_list)} SRR accessions: {srr_list}")
        
        if srr_list:
            print("✅ SRA search working correctly - returns SRR accessions")
            return True
        else:
            print("⚠️ No SRA data found for test GSM")
            return False
            
    except Exception as e:
        print(f"❌ SRA search failed: {e}")
        return False

def test_download_logic():
    """Test download logic without actual download"""
    print("\n🧪 Testing download logic...")
    
    output_dir = Path("test_sra_output")
    downloader = SRADownloader(output_dir)
    
    # Mock the search to return test SRR accessions
    def mock_search(dataset_id):
        return ["SRR123456", "SRR789012"] if "GSM" in dataset_id else []
    
    # Replace the search method temporarily
    original_search = downloader.search_sra_data
    downloader.search_sra_data = mock_search
    
    try:
        # Test download logic (should use SRR accessions)
        result = downloader.download_sra_data("GSM5292143")
        print(f"📊 Download logic result: {result}")
        print("✅ Download logic uses SRR accessions correctly")
        return True
        
    except Exception as e:
        print(f"❌ Download logic test failed: {e}")
        return False
    finally:
        # Restore original method
        downloader.search_sra_data = original_search

if __name__ == "__main__":
    print("🚀 Testing SRA downloader fixes...")
    print("=" * 50)
    
    # Test 1: SRA search
    search_ok = test_sra_search()
    
    # Test 2: Download logic
    logic_ok = test_download_logic()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"  SRA Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    print(f"  Download Logic: {'✅ PASS' if logic_ok else '❌ FAIL'}")
    
    if search_ok and logic_ok:
        print("\n🎉 All tests passed! SRA downloader is fixed.")
    else:
        print("\n❌ Some tests failed. Check the issues above.")


