#!/usr/bin/env python3
"""
Test script for the streaming pipeline with actual dataset list
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from streaming_pipeline import StreamingPipeline

def test_dataset_loading():
    """Test loading the dataset list"""
    print("Testing dataset list loading...")
    
    # Test CSV loading
    try:
        df = pd.read_csv('data_list_full.csv', sep=';;', encoding='latin-1', engine='python')
        print(f"✅ Successfully loaded CSV: {df.shape[0]} datasets")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Repositories: {df['repository'].value_counts().to_dict()}")
        print(f"   Organisms: {df['Organism'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    
    try:
        # Create a test config
        test_config = {
            'base_dir': '/tmp/test_als_foundation',
            'dataset_list_path': 'data_list_full.csv',
            'batch_size': 2,
            'max_storage_gb': 100,
            'min_free_space_gb': 10
        }
        
        # Save test config
        import json
        with open('test_config.json', 'w') as f:
            json.dump(test_config, f, indent=2)
        
        # Initialize pipeline
        pipeline = StreamingPipeline('test_config.json')
        print("✅ Pipeline initialized successfully")
        
        # Test dataset loading
        datasets = pipeline.load_dataset_list()
        print(f"✅ Loaded {len(datasets)} datasets from pipeline")
        
        # Test batch creation
        batch = pipeline.get_next_batch(2)
        print(f"✅ Created batch with {len(batch)} datasets")
        
        if batch:
            print("   Sample batch data:")
            for i, dataset in enumerate(batch[:2]):
                print(f"   {i+1}. {dataset.get('dataset_id', 'N/A')} - {dataset.get('Title', 'N/A')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_data_type_parsing():
    """Test data type parsing"""
    print("\nTesting data type parsing...")
    
    try:
        df = pd.read_csv('data_list_full.csv', sep=';;', encoding='latin-1', engine='python')
        
        # Test data type parsing
        data_types = df['data_type'].value_counts()
        print(f"✅ Found {len(data_types)} unique data types")
        print("   Top data types:")
        for dt, count in data_types.head(10).items():
            print(f"   - {dt}: {count} datasets")
        
        # Test processed data detection
        pipeline = StreamingPipeline('test_config.json')
        processed_count = 0
        for _, row in df.iterrows():
            if pipeline.check_processed_data_available(row['dataset_id'], row['data_type']):
                processed_count += 1
        
        print(f"✅ {processed_count}/{len(df)} datasets have processed data available")
        
        return True
        
    except Exception as e:
        print(f"❌ Data type parsing failed: {e}")
        return False

def test_disease_status_parsing():
    """Test disease status parsing"""
    print("\nTesting disease status parsing...")
    
    try:
        df = pd.read_csv('data_list_full.csv', sep=';;', encoding='latin-1', engine='python')
        
        # Analyze disease status
        disease_status = df['disease_status'].value_counts()
        print(f"✅ Found {len(disease_status)} unique disease statuses")
        print("   Top disease statuses:")
        for status, count in disease_status.head(10).items():
            print(f"   - {status}: {count} datasets")
        
        # Check for ALS datasets
        als_datasets = df[df['disease_status'].str.contains('ALS', case=False, na=False)]
        print(f"✅ Found {len(als_datasets)} datasets containing ALS")
        
        # Check for control datasets
        control_datasets = df[df['disease_status'].str.contains('control', case=False, na=False)]
        print(f"✅ Found {len(control_datasets)} datasets containing control")
        
        return True
        
    except Exception as e:
        print(f"❌ Disease status parsing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ALS FOUNDATION MODEL - PIPELINE TESTING")
    print("=" * 60)
    
    tests = [
        test_dataset_loading,
        test_pipeline_initialization,
        test_data_type_parsing,
        test_disease_status_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
