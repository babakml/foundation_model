#!/usr/bin/env python3
"""
Simple test script for data loading without heavy dependencies
"""

import pandas as pd
import json
from pathlib import Path

def test_csv_loading():
    """Test loading the CSV file"""
    print("Testing CSV file loading...")
    
    try:
        df = pd.read_csv('data_list_full.csv', sep=';;', encoding='latin-1', engine='python')
        print(f"✅ Successfully loaded CSV: {df.shape[0]} datasets, {df.shape[1]} columns")
        
        # Check columns
        expected_columns = ['Title', 'Organism', 'dataset_id', 'repository', 'disease_status', 'tissue_type', 'data_type']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            print(f"⚠️  Missing columns: {missing_columns}")
        else:
            print("✅ All expected columns present")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return None

def analyze_data(df):
    """Analyze the loaded data"""
    print("\nAnalyzing dataset...")
    
    # Repository analysis
    print(f"📊 Repositories:")
    repo_counts = df['repository'].value_counts()
    for repo, count in repo_counts.items():
        print(f"   - {repo}: {count} datasets")
    
    # Organism analysis
    print(f"\n🧬 Organisms:")
    org_counts = df['Organism'].value_counts()
    for org, count in org_counts.items():
        print(f"   - {org}: {count} datasets")
    
    # Disease status analysis
    print(f"\n🏥 Disease Status (top 10):")
    disease_counts = df['disease_status'].value_counts()
    for status, count in disease_counts.head(10).items():
        print(f"   - {status}: {count} datasets")
    
    # Data type analysis
    print(f"\n📁 Data Types (top 10):")
    data_type_counts = df['data_type'].value_counts()
    for dtype, count in data_type_counts.head(10).items():
        print(f"   - {dtype}: {count} datasets")
    
    # Tissue type analysis
    print(f"\n🔬 Tissue Types (top 10):")
    tissue_counts = df['tissue_type'].value_counts()
    for tissue, count in tissue_counts.head(10).items():
        print(f"   - {tissue}: {count} datasets")

def test_processed_data_detection(df):
    """Test processed data detection logic"""
    print(f"\n🔍 Testing processed data detection...")
    
    processed_indicators = ['H5', 'MTX', 'TSV', 'RDS', 'seurat', 'processed']
    
    processed_count = 0
    raw_count = 0
    
    for _, row in df.iterrows():
        data_type = str(row.get('data_type', '')).upper()
        if any(indicator in data_type for indicator in processed_indicators):
            processed_count += 1
        else:
            raw_count += 1
    
    print(f"✅ Processed data available: {processed_count} datasets")
    print(f"✅ Raw data only: {raw_count} datasets")
    print(f"✅ Processed data percentage: {processed_count/(processed_count+raw_count)*100:.1f}%")

def test_batch_creation(df):
    """Test batch creation logic"""
    print(f"\n📦 Testing batch creation...")
    
    batch_size = 3
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"✅ Total datasets: {len(df)}")
    print(f"✅ Batch size: {batch_size}")
    print(f"✅ Total batches: {total_batches}")
    
    # Show first batch
    first_batch = df.head(batch_size)
    print(f"\n📋 First batch ({len(first_batch)} datasets):")
    for i, (_, row) in enumerate(first_batch.iterrows()):
        print(f"   {i+1}. {row['dataset_id']} - {row['Title'][:50]}...")

def create_sample_config():
    """Create a sample configuration file"""
    print(f"\n⚙️  Creating sample configuration...")
    
    config = {
        "base_dir": "/scratch/username/als_foundation",
        "dataset_list_path": "data_list_full.csv",
        "batch_size": 3,
        "max_storage_gb": 550,
        "min_free_space_gb": 50,
        "download_timeout": 3600,
        "processing_timeout": 7200,
        "model_config": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "n_layers": 4,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "n_epochs": 10
        },
        "quality_control": {
            "min_genes": 200,
            "max_genes": 5000,
            "max_mt_percent": 20,
            "min_cells": 3
        },
        "normalization": {
            "target_sum": 10000,
            "log_transform": True,
            "scale": True,
            "max_value": 10
        },
        "storage_management": {
            "keep_raw_data": False,
            "keep_processed_data": False,
            "keep_checkpoints": True,
            "max_checkpoints": 5,
            "compress_metadata": True
        }
    }
    
    with open('sample_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Sample configuration created: sample_config.json")

def main():
    """Run all tests"""
    print("=" * 60)
    print("ALS FOUNDATION MODEL - DATA LOADING TEST")
    print("=" * 60)
    
    # Test CSV loading
    df = test_csv_loading()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    # Analyze data
    analyze_data(df)
    
    # Test processed data detection
    test_processed_data_detection(df)
    
    # Test batch creation
    test_batch_creation(df)
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("🎉 Data loading test completed successfully!")
    print("✅ Your CSV file is ready for the streaming pipeline")
    print("✅ Sample configuration created")
    print("=" * 60)

if __name__ == "__main__":
    main()
