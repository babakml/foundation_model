#!/usr/bin/env python3
"""
Analyze storage cleanup opportunities
- Find successfully processed datasets
- Identify raw files that can be removed
- Calculate space savings
"""

import json
import os
from pathlib import Path
import subprocess

def get_file_size(path):
    """Get file size in GB"""
    try:
        result = subprocess.run(['du', '-sh', str(path)], capture_output=True, text=True)
        if result.returncode == 0:
            size_str = result.stdout.split()[0]
            # Convert to GB
            if size_str.endswith('G'):
                return float(size_str[:-1])
            elif size_str.endswith('M'):
                return float(size_str[:-1]) / 1024
            elif size_str.endswith('K'):
                return float(size_str[:-1]) / (1024 * 1024)
            else:
                return float(size_str) / (1024 * 1024 * 1024)
    except:
        pass
    return 0

def find_processed_datasets():
    """Find datasets that have been successfully processed"""
    processed_dir = Path("data/processed")
    processed_files = []
    
    if processed_dir.exists():
        for h5ad_file in processed_dir.glob("*.h5ad"):
            dataset_id = h5ad_file.stem.replace("_processed", "")
            processed_files.append({
                'dataset_id': dataset_id,
                'processed_file': h5ad_file,
                'size_gb': get_file_size(h5ad_file)
            })
    
    return processed_files

def find_raw_files_for_cleanup(dataset_id):
    """Find raw files that can be removed for a processed dataset"""
    raw_dir = Path("data/raw") / dataset_id
    cleanup_candidates = []
    
    if not raw_dir.exists():
        return cleanup_candidates
    
    # Look for common raw file types that can be removed after processing
    patterns = [
        "*.mtx*",      # Matrix files
        "*.h5",        # H5 files (not h5ad)
        "*.tar.gz",    # Compressed archives
        "*.tar",       # Tar archives
        "*.gz",        # Gzipped files
        "barcodes.*",  # Barcode files
        "features.*",  # Feature files
        "genes.*"      # Gene files
    ]
    
    for pattern in patterns:
        for file_path in raw_dir.rglob(pattern):
            # Skip directories
            if file_path.is_file():
                cleanup_candidates.append({
                    'file': file_path,
                    'size_gb': get_file_size(file_path),
                    'type': pattern.replace("*", "").replace(".", "")
                })
    
    return cleanup_candidates

def analyze_cleanup_opportunities():
    """Analyze what can be cleaned up"""
    print("🔍 Analyzing storage cleanup opportunities...")
    print("=" * 60)
    
    # Find processed datasets
    processed_datasets = find_processed_datasets()
    print(f"📊 Found {len(processed_datasets)} successfully processed datasets")
    
    total_cleanup_size = 0
    cleanup_summary = []
    
    for dataset in processed_datasets:
        dataset_id = dataset['dataset_id']
        raw_files = find_raw_files_for_cleanup(dataset_id)
        
        if raw_files:
            dataset_cleanup_size = sum(f['size_gb'] for f in raw_files)
            total_cleanup_size += dataset_cleanup_size
            
            cleanup_summary.append({
                'dataset_id': dataset_id,
                'processed_size_gb': dataset['size_gb'],
                'raw_files_count': len(raw_files),
                'cleanup_size_gb': dataset_cleanup_size,
                'files': raw_files
            })
    
    # Print summary
    print(f"\n📈 CLEANUP SUMMARY:")
    print(f"  - Processed datasets: {len(processed_datasets)}")
    print(f"  - Datasets with cleanup opportunities: {len(cleanup_summary)}")
    print(f"  - Total space that can be freed: {total_cleanup_size:.2f} GB")
    
    # Detailed breakdown
    print(f"\n📋 DETAILED BREAKDOWN:")
    for item in cleanup_summary[:10]:  # Show first 10
        print(f"  {item['dataset_id']}:")
        print(f"    - Processed: {item['processed_size_gb']:.2f} GB")
        print(f"    - Raw files: {item['raw_files_count']} files, {item['cleanup_size_gb']:.2f} GB")
        print(f"    - Space savings: {item['cleanup_size_gb']:.2f} GB")
    
    if len(cleanup_summary) > 10:
        print(f"  ... and {len(cleanup_summary) - 10} more datasets")
    
    # Generate cleanup script
    generate_cleanup_script(cleanup_summary)
    
    return cleanup_summary, total_cleanup_size

def generate_cleanup_script(cleanup_summary):
    """Generate a script to safely remove raw files"""
    script_content = """#!/bin/bash
# cleanup_raw_files.sh - Remove raw files for successfully processed datasets

echo "🧹 Cleaning up raw files for processed datasets..."
echo "This will remove raw files (mtx, h5, tar.gz, etc.) for datasets that have been successfully processed to H5AD format."
echo ""

# Safety check - only proceed if processed files exist
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed/*.h5ad 2>/dev/null)" ]; then
    echo "❌ No processed H5AD files found. Aborting cleanup."
    exit 1
fi

echo "✅ Found processed H5AD files. Proceeding with cleanup..."
echo ""

"""
    
    for item in cleanup_summary:
        dataset_id = item['dataset_id']
        script_content += f"""
# Cleanup {dataset_id} ({item['cleanup_size_gb']:.2f} GB)
echo "🧹 Cleaning {dataset_id}..."
if [ -d "data/raw/{dataset_id}" ]; then
    # Remove specific file types
    find "data/raw/{dataset_id}" -name "*.mtx*" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "*.h5" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "*.tar.gz" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "*.tar" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "*.gz" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "barcodes.*" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "features.*" -type f -delete 2>/dev/null
    find "data/raw/{dataset_id}" -name "genes.*" -type f -delete 2>/dev/null
    
    # Remove empty directories
    find "data/raw/{dataset_id}" -type d -empty -delete 2>/dev/null
    
    echo "  ✅ Cleaned {dataset_id}"
else
    echo "  ⚠️  Directory not found: data/raw/{dataset_id}"
fi
"""
    
    script_content += """
echo ""
echo "🎉 Cleanup completed!"
echo "Run 'du -sh data/raw data/processed' to check new sizes."
"""
    
    with open("cleanup_raw_files.sh", "w") as f:
        f.write(script_content)
    
    print(f"\n📝 Generated cleanup script: cleanup_raw_files.sh")
    print(f"   Run: chmod +x cleanup_raw_files.sh && ./cleanup_raw_files.sh")

if __name__ == "__main__":
    cleanup_summary, total_size = analyze_cleanup_opportunities()
    
    if total_size > 0:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Run the generated cleanup script to free up {total_size:.2f} GB of space")
        print(f"   This will remove raw files for {len(cleanup_summary)} successfully processed datasets")
    else:
        print(f"\n💡 No cleanup opportunities found")


