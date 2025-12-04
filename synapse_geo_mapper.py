#!/usr/bin/env python3
"""
Script to map GEO IDs (GSM/GSE) to Synapse IDs and add them to the dataset CSV.
Reads data_list_full.csv, queries Synapse for each GEO ID, and exports enhanced CSV.
"""

import json
import os
import sys
import time
from typing import Dict, Optional, List

# Check NumPy version compatibility before importing pandas
try:
    import numpy as np
    numpy_version = np.__version__
    if numpy_version.startswith('2.'):
        print("⚠️  WARNING: NumPy 2.x detected. This may cause compatibility issues.")
        print("   Solution: Downgrade NumPy to 1.x:")
        print("   pip install 'numpy<2'")
        print("   OR")
        print("   conda install 'numpy<2'")
        print("\n   Attempting to continue anyway...\n")
except ImportError:
    pass

try:
    import pandas as pd
except (AttributeError, ImportError) as e:
    if '_ARRAY_API' in str(e) or 'NumPy' in str(e):
        print("\n❌ ERROR: NumPy compatibility issue detected!")
        print("   This is caused by NumPy 2.x incompatibility with pandas/numexpr/bottleneck.")
        print("\n   To fix this, run one of the following:")
        print("   pip install 'numpy<2'")
        print("   OR")
        print("   conda install 'numpy<2'")
        print("\n   Then try running this script again.\n")
        sys.exit(1)
    else:
        raise

try:
    import synapseclient
    from synapseclient import Synapse
except ImportError:
    print("❌ ERROR: synapseclient not installed!")
    print("   To install, run: pip install synapseclient")
    sys.exit(1)


class SynapseGEOMapper:
    """Maps GEO IDs to Synapse IDs using Synapse query API."""
    
    def __init__(self):
        """Initialize Synapse client."""
        self.syn = None
        self._login()
        self.mapping_cache = {}
        self.query_count = 0
        self.success_count = 0
        self.fail_count = 0
    
    def _login(self):
        """Login to Synapse. Prompts for credentials if not cached."""
        # Try different approaches to login
        login_success = False
        last_error = None
        
        # Approach 1: Try reading config file directly and using credentials
        try:
            import configparser
            config_path = os.path.expanduser('~/.synapseConfig')
            if os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                # Try profile 'j' first
                if 'profile j' in config:
                    username = config['profile j'].get('username')
                    authtoken = config['profile j'].get('authtoken')
                    if username and authtoken:
                        self.syn = Synapse()
                        self.syn.login(authToken=authtoken, silent=True)
                        print("✅ Successfully connected to Synapse (using credentials from config)")
                        login_success = True
                # Try default profile if 'j' didn't work
                elif 'default' in config:
                    username = config['default'].get('username')
                    authtoken = config['default'].get('authtoken')
                    if username and authtoken:
                        self.syn = Synapse()
                        self.syn.login(authToken=authtoken, silent=True)
                        print("✅ Successfully connected to Synapse (using default profile)")
                        login_success = True
        except Exception as e:
            last_error = e
            # Try with default profile using Synapse's built-in config reading
            try:
                self.syn = Synapse()
                self.syn.login(silent=True)
                print("✅ Successfully connected to Synapse (default profile)")
                login_success = True
            except:
                pass
        
        # Approach 2: Try interactive login
        if not login_success:
            try:
                print("\n⚠️  Config file issue detected. Attempting interactive login...")
                self.syn = Synapse()
                self.syn.login()
                print("✅ Successfully connected to Synapse (interactive login)")
                login_success = True
            except Exception as e2:
                last_error = e2
        
        if not login_success:
            print("\n❌ Failed to authenticate with Synapse.")
            print("\n   Please try one of the following:")
            print("   1. Delete the config file and recreate it:")
            print(f"      rm ~/.synapseConfig")
            print("      synapse config")
            print("\n   2. Or login interactively:")
            print("      python -c \"import synapseclient; syn = synapseclient.Synapse(); syn.login()\"")
            print(f"\n   Last error: {last_error}\n")
            sys.exit(1)
    
    def query_synapse_by_geo_id(self, geo_id: str) -> Optional[str]:
        """
        Query Synapse for entities with a specific GEO ID annotation.
        
        Args:
            geo_id: GEO ID (e.g., 'GSM5292144' or 'GSE275999')
            
        Returns:
            Synapse ID (e.g., 'syn53416892') if found, None otherwise
        """
        if not geo_id or pd.isna(geo_id) or str(geo_id).strip() == '':
            return None
            
        geo_id = str(geo_id).strip()
        
        # Check cache first
        if geo_id in self.mapping_cache:
            return self.mapping_cache[geo_id]
        
        self.query_count += 1
        
        # Try different entity types and annotation field names
        entity_types = ['entity', 'folder', 'project', 'file']
        annotation_fields = [
            'GEO_ID', 'geo_id', 'GEO', 'geo',
            'GSM_ID', 'gsm_id', 'GSM', 'gsm',
            'GSE_ID', 'gse_id', 'GSE', 'gse',
            'dataset_id', 'Dataset_ID', 'DATASET_ID', 'datasetId'
        ]
        
        # First, try searching by annotations
        for entity_type in entity_types:
            for field in annotation_fields:
                try:
                    query = f"SELECT id, name FROM {entity_type} WHERE {field}='{geo_id}'"
                    results = self.syn.query(query)
                    if results and len(results.get('results', [])) > 0:
                        synapse_id = results['results'][0]['id']
                        name = results['results'][0].get('name', 'Unknown')
                        print(f"  ✅ [{self.query_count}] Found {geo_id} → {synapse_id} ({name})")
                        self.mapping_cache[geo_id] = synapse_id
                        self.success_count += 1
                        time.sleep(0.1)
                        return synapse_id
                except Exception as e:
                    # Silently continue to next pattern
                    continue
        
        # Try searching in name/alias across all entity types
        for entity_type in entity_types:
            try:
                # Search in name
                query = f"SELECT id, name FROM {entity_type} WHERE name LIKE '%{geo_id}%'"
                results = self.syn.query(query)
                if results and len(results.get('results', [])) > 0:
                    # Check if any result actually contains the GEO ID
                    for result in results.get('results', []):
                        result_name = result.get('name', '')
                        if geo_id in result_name:
                            synapse_id = result['id']
                            print(f"  ⚠️  [{self.query_count}] Found approximate match {geo_id} → {synapse_id} ({result_name})")
                            self.mapping_cache[geo_id] = synapse_id
                            self.success_count += 1
                            time.sleep(0.1)
                            return synapse_id
            except Exception as e:
                continue
        
        # Last resort: try searching without entity type restriction
        try:
            query = f"SELECT id, name FROM entity WHERE name LIKE '%{geo_id}%' OR alias LIKE '%{geo_id}%'"
            results = self.syn.query(query)
            if results and len(results.get('results', [])) > 0:
                synapse_id = results['results'][0]['id']
                name = results['results'][0].get('name', 'Unknown')
                print(f"  ⚠️  [{self.query_count}] Found by alias/name search {geo_id} → {synapse_id} ({name})")
                self.mapping_cache[geo_id] = synapse_id
                self.success_count += 1
                time.sleep(0.1)
                return synapse_id
        except Exception as e:
            pass
        
        print(f"  ❌ [{self.query_count}] No Synapse ID found for {geo_id}")
        self.mapping_cache[geo_id] = None
        self.fail_count += 1
        time.sleep(0.1)
        return None
    
    def extract_geo_id(self, row: pd.Series) -> Optional[str]:
        """
        Extract GEO ID from a row, trying multiple possible column names.
        
        Args:
            row: DataFrame row
            
        Returns:
            GEO ID if found, None otherwise
        """
        # Try common column names for GEO IDs
        possible_columns = [
            'dataset_id', 'GEO_ID', 'geo_id', 'GEO', 'geo',
            'GSM_ID', 'gsm_id', 'GSM', 'gsm',
            'GSE_ID', 'gse_id', 'GSE', 'gse',
            'id', 'ID', 'dataset_name', 'name'
        ]
        
        for col in possible_columns:
            if col in row and pd.notna(row[col]):
                value = str(row[col]).strip()
                # Skip empty values
                if not value or value == '' or value.lower() == 'nan':
                    continue
                # Check if it looks like a GEO ID (starts with GSM, GSE, E-MTAB)
                if value.startswith(('GSM', 'GSE', 'E-MTAB')):
                    return value
                # If the column name suggests it's a GEO ID, use it (even if it doesn't start with GSM/GSE)
                if col in ['GEO_ID', 'geo_id', 'GSM_ID', 'gsm_id', 'GSE_ID', 'gse_id', 'dataset_id']:
                    return value
        
        return None
    
    def process_csv(self, input_csv: str, output_csv: str):
        """
        Read CSV, query Synapse for each GEO ID, and export enhanced CSV.
        
        Args:
            input_csv: Path to input CSV file (data_list_full.csv)
            output_csv: Path to output CSV file with Synapse IDs added
        """
        print(f"📖 Reading {input_csv}...")
        # Try different encodings and parsing options
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        last_error = None
        
        # Check pandas version to use correct parameter
        # on_bad_lines was introduced in pandas 1.3.0, error_bad_lines was removed in 2.0.0
        pandas_version = pd.__version__
        version_parts = [int(x) for x in pandas_version.split('.')[:2]]
        use_on_bad_lines = (version_parts[0] > 1) or (version_parts[0] == 1 and version_parts[1] >= 3)
        
        for encoding in encodings:
            try:
                # Try with semicolon delimiter first (since CSV uses ;;)
                read_kwargs = {
                    'encoding': encoding,
                    'engine': 'python',
                    'sep': ';',  # Use semicolon as delimiter
                    'quotechar': '"',
                    'skipinitialspace': True
                }
                # Use appropriate parameter based on pandas version
                if use_on_bad_lines:
                    read_kwargs['on_bad_lines'] = 'skip'
                else:
                    read_kwargs['error_bad_lines'] = False
                    read_kwargs['warn_bad_lines'] = False
                
                df = pd.read_csv(input_csv, **read_kwargs)
                # Clean up column names (remove trailing ;; and whitespace)
                df.columns = df.columns.str.rstrip(';').str.strip()
                print(f"✅ Loaded {len(df)} rows (using {encoding} encoding, semicolon delimiter)")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except (pd.errors.ParserError, ValueError, TypeError) as e:
                last_error = e
                # Try with comma delimiter
                try:
                    read_kwargs = {
                        'encoding': encoding,
                        'engine': 'python',
                        'sep': ',',
                        'quotechar': '"',
                        'skipinitialspace': True
                    }
                    if use_on_bad_lines:
                        read_kwargs['on_bad_lines'] = 'skip'
                    else:
                        read_kwargs['error_bad_lines'] = False
                        read_kwargs['warn_bad_lines'] = False
                    
                    df = pd.read_csv(input_csv, **read_kwargs)
                    df.columns = df.columns.str.rstrip(';').str.strip()
                    print(f"✅ Loaded {len(df)} rows (using {encoding} encoding, comma delimiter)")
                    break
                except Exception as e2:
                    last_error = e2
                    # Try with auto-detected separator
                    try:
                        read_kwargs = {
                            'encoding': encoding,
                            'engine': 'python',
                            'sep': None,  # Auto-detect separator
                            'quotechar': '"',
                            'skipinitialspace': True
                        }
                        if use_on_bad_lines:
                            read_kwargs['on_bad_lines'] = 'skip'
                        else:
                            read_kwargs['error_bad_lines'] = False
                            read_kwargs['warn_bad_lines'] = False
                        
                        df = pd.read_csv(input_csv, **read_kwargs)
                        df.columns = df.columns.str.rstrip(';').str.strip()
                        print(f"✅ Loaded {len(df)} rows (using {encoding} encoding, auto-separator)")
                        break
                    except Exception as e3:
                        last_error = e3
                        continue
        
        if df is None:
            print(f"\n❌ Could not read {input_csv}")
            print(f"   Last error: {last_error}")
            print("\n   Trying alternative approach with more lenient parsing...")
            try:
                # Last resort: very lenient parsing
                read_kwargs = {
                    'encoding': 'latin-1',
                    'engine': 'python',
                    'sep': None
                }
                if use_on_bad_lines:
                    read_kwargs['on_bad_lines'] = 'skip'
                else:
                    read_kwargs['error_bad_lines'] = False
                    read_kwargs['warn_bad_lines'] = False
                
                df = pd.read_csv(input_csv, **read_kwargs)
                df.columns = df.columns.str.rstrip(';').str.strip()
                print(f"✅ Loaded {len(df)} rows (using lenient parsing)")
            except Exception as e:
                raise ValueError(f"Could not read {input_csv}. Error: {e}")
        
        print(f"📋 Columns: {', '.join(df.columns.tolist())}\n")
        
        # Initialize synapse_id column
        df['synapse_id'] = None
        
        # Process each row
        print(f"🔍 Querying Synapse for {len(df)} datasets...\n")
        
        for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
            geo_id = self.extract_geo_id(row)
            
            if geo_id:
                print(f"[{row_num}/{len(df)}] Processing {geo_id}...")
                synapse_id = self.query_synapse_by_geo_id(geo_id)
                df.at[idx, 'synapse_id'] = synapse_id if synapse_id else ''
            else:
                print(f"[{row_num}/{len(df)}] ⚠️  No GEO ID found in row, skipping...")
                df.at[idx, 'synapse_id'] = ''
        
        # Save enhanced CSV with same format as input (double semicolon-delimited)
        print(f"\n💾 Saving enhanced CSV to {output_csv}...")
        # Save with double semicolon delimiter to match input format exactly
        # Use a custom separator that pandas doesn't natively support, so we'll write manually
        with open(output_csv, 'w', encoding='latin-1', newline='') as f:
            # Write header with double semicolons
            columns = df.columns.tolist()
            header = ';;'.join(columns) + ';;\n'
            f.write(header)
            
            # Write each row with double semicolons
            for idx, row in df.iterrows():
                row_values = []
                for col in columns:
                    value = str(row[col]) if pd.notna(row[col]) else ''
                    # Escape quotes and wrap in quotes if contains semicolon or quote
                    if ';' in value or '"' in value or '\n' in value:
                        value = '"' + value.replace('"', '""') + '"'
                    row_values.append(value)
                row_line = ';;'.join(row_values) + ';;\n'
                f.write(row_line)
        
        # Print summary
        found_count = df['synapse_id'].notna().sum() - (df['synapse_id'] == '').sum()
        print(f"\n📊 Summary:")
        print(f"   Total datasets: {len(df)}")
        print(f"   Synapse IDs found: {found_count}")
        print(f"   Synapse IDs not found: {len(df) - found_count}")
        print(f"   Total queries: {self.query_count}")
        print(f"   Successful queries: {self.success_count}")
        print(f"   Failed queries: {self.fail_count}")
        print(f"✅ Enhanced CSV saved to {output_csv}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map GEO IDs to Synapse IDs and add to CSV')
    parser.add_argument('--input', type=str, default='data_list_full.csv',
                       help='Input CSV file path (default: data_list_full.csv)')
    parser.add_argument('--output', type=str, default='data_list_full_with_synapse.csv',
                       help='Output CSV file path (default: data_list_full_with_synapse.csv)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    mapper = SynapseGEOMapper()
    mapper.process_csv(args.input, args.output)


if __name__ == '__main__':
    main()

