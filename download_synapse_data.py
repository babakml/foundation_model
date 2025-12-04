#!/usr/bin/env python3
"""
Script to read data_list_full.csv, find a Synapse ID, and download it.
Works with semicolon-delimited CSV files.
"""

import os
import sys
import re
import pandas as pd
import synapseclient
from synapseclient import Synapse
import configparser


class SynapseDownloader:
    """Downloads data from Synapse using Synapse IDs found in CSV file."""
    
    def __init__(self):
        """Initialize Synapse client."""
        self.syn = None
        self._login()
    
    def _login(self):
        """Login to Synapse using credentials from config file."""
        try:
            # Read credentials from config file
            config_path = os.path.expanduser('~/.synapseConfig')
            if os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                # Try profile 'j' first, then 'default', then any profile
                for profile_name in ['profile j', 'default']:
                    if profile_name in config:
                        username = config[profile_name].get('username')
                        authtoken = config[profile_name].get('authtoken')
                        if username and authtoken:
                            self.syn = Synapse()
                            self.syn.login(authToken=authtoken, silent=True)
                            print("✅ Successfully connected to Synapse")
                            return
                # Try any other profile
                for section in config.sections():
                    if section.startswith('profile ') or section == 'default':
                        username = config[section].get('username')
                        authtoken = config[section].get('authtoken')
                        if username and authtoken:
                            self.syn = Synapse()
                            self.syn.login(authToken=authtoken, silent=True)
                            print("✅ Successfully connected to Synapse")
                            return
            # Fallback to default login
            self.syn = Synapse()
            self.syn.login(silent=True)
            print("✅ Successfully connected to Synapse")
        except Exception as e:
            print(f"❌ Failed to login to Synapse: {e}")
            print("\n   Please ensure your Synapse credentials are configured.")
            print("   Run: synapse config")
            sys.exit(1)
    
    def find_synapse_ids(self, csv_file: str):
        """
        Read CSV file and find all Synapse IDs.
        
        Args:
            csv_file: Path to CSV file (semicolon-delimited)
            
        Returns:
            List of tuples (row_index, synapse_id, context)
        """
        print(f"📖 Reading {csv_file}...")
        
        # Try different encodings for CSV
        encodings = ['latin-1', 'utf-8', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        df = None
        last_error = None
        
        for encoding in encodings:
            try:
                # Try semicolon delimiter first (based on your CSV format)
                df = pd.read_csv(
                    csv_file,
                    sep=';',
                    encoding=encoding,
                    engine='python',
                    error_bad_lines=False,
                    warn_bad_lines=False
                )
                # Clean up column names (remove trailing ;;)
                df.columns = df.columns.str.rstrip(';').str.strip()
                print(f"✅ Loaded {len(df)} rows (using {encoding} encoding, semicolon delimiter)")
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                # Try comma delimiter as fallback
                try:
                    df = pd.read_csv(
                        csv_file,
                        sep=',',
                        encoding=encoding,
                        engine='python',
                        error_bad_lines=False,
                        warn_bad_lines=False
                    )
                    print(f"✅ Loaded {len(df)} rows (using {encoding} encoding, comma delimiter)")
                    break
                except:
                    continue
        
        if df is None:
            print(f"❌ Error reading CSV file: {last_error}")
            print("   Tried encodings: " + ", ".join(encodings))
            sys.exit(1)
        
        print(f"✅ Loaded {len(df)} rows")
        print(f"📋 Columns: {', '.join(df.columns.tolist())}\n")
        
        # Pattern to match Synapse IDs (syn followed by numbers)
        synapse_id_pattern = re.compile(r'syn\d+', re.IGNORECASE)
        synapse_ids = []
        
        # Search through all columns
        for col in df.columns:
            for idx, value in df[col].items():
                if pd.notna(value):
                    value_str = str(value)
                    # Search for Synapse ID pattern
                    matches = synapse_id_pattern.findall(value_str)
                    for match in matches:
                        synapse_id = match.lower()  # Normalize to lowercase
                        # Get some context (row data)
                        context = f"Column: {col}, Row: {idx+1}"
                        synapse_ids.append((idx, synapse_id, context, df.iloc[idx].to_dict()))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for item in synapse_ids:
            synapse_id = item[1]
            if synapse_id not in seen:
                seen.add(synapse_id)
                unique_ids.append(item)
        
        return unique_ids
    
    def download_synapse_entity(self, synapse_id: str, output_dir: str = "synapse_downloads"):
        """
        Download a Synapse entity.
        
        Args:
            synapse_id: Synapse ID (e.g., 'syn53421674')
            output_dir: Directory to download to
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\n📥 Downloading {synapse_id}...")
            
            # Get entity info first to check its type
            entity = self.syn.get(synapse_id, downloadFile=False)
            entity_type = entity.concreteType if hasattr(entity, 'concreteType') else 'unknown'
            
            print(f"   Entity type: {entity_type}")
            print(f"   Entity name: {entity.name if hasattr(entity, 'name') else 'N/A'}")
            
            # If it's a folder or project, download recursively
            if 'Folder' in entity_type or 'Project' in entity_type:
                print(f"   Detected folder/project - downloading recursively...")
                # Create a subdirectory for this entity
                entity_dir = os.path.join(output_dir, entity.name if hasattr(entity, 'name') else synapse_id)
                os.makedirs(entity_dir, exist_ok=True)
                
                # Use syn.getChildren to get all files in the folder
                try:
                    children = list(self.syn.getChildren(synapse_id))
                    print(f"   Found {len(children)} items in folder")
                    
                    if not children:
                        print(f"⚠️  Folder is empty")
                        return True
                    
                    # Download each child
                    downloaded_files = []
                    for child in children:
                        child_id = child.get('id')
                        child_name = child.get('name', '')
                        child_type = child.get('type', [])
                        
                        if 'file' in child_type or 'FileEntity' in str(child_type):
                            print(f"   Downloading file: {child_name}...")
                            try:
                                file_entity = self.syn.get(child_id, downloadLocation=entity_dir)
                                file_path = os.path.join(entity_dir, file_entity.name if hasattr(file_entity, 'name') else child_name)
                                if os.path.exists(file_path):
                                    downloaded_files.append(file_path)
                            except Exception as e:
                                print(f"     ⚠️  Failed to download {child_name}: {e}")
                        elif 'folder' in child_type or 'Folder' in str(child_type):
                            # Recursively download subfolder
                            print(f"   Downloading subfolder: {child_name}...")
                            try:
                                subfolder_dir = os.path.join(entity_dir, child_name)
                                self.syn.get(child_id, downloadLocation=subfolder_dir, recursive=True)
                                # Count files in subfolder
                                for root, dirs, filenames in os.walk(subfolder_dir):
                                    downloaded_files.extend([os.path.join(root, f) for f in filenames])
                            except Exception as e:
                                print(f"     ⚠️  Failed to download subfolder {child_name}: {e}")
                    
                    # Verify download
                    if downloaded_files:
                        print(f"✅ Successfully downloaded {synapse_id}")
                        print(f"   Location: {entity_dir}")
                        print(f"   Files downloaded: {len(downloaded_files)}")
                        print(f"   Sample files:")
                        for f in downloaded_files[:5]:  # Show first 5 files
                            rel_path = os.path.relpath(f, entity_dir)
                            size = os.path.getsize(f) / (1024*1024)  # Size in MB
                            print(f"     - {rel_path} ({size:.2f} MB)")
                        if len(downloaded_files) > 5:
                            print(f"     ... and {len(downloaded_files) - 5} more files")
                    else:
                        # Try alternative method - direct recursive download
                        print(f"   Trying alternative download method...")
                        try:
                            self.syn.get(synapse_id, downloadLocation=entity_dir, recursive=True)
                            # Check again
                            files = []
                            for root, dirs, filenames in os.walk(entity_dir):
                                for filename in filenames:
                                    files.append(os.path.join(root, filename))
                            
                            if files:
                                print(f"✅ Successfully downloaded {synapse_id}")
                                print(f"   Location: {entity_dir}")
                                print(f"   Files downloaded: {len(files)}")
                            else:
                                print(f"⚠️  Download completed but no files found")
                                print(f"   The folder might be empty or you may not have access to the files")
                        except Exception as e:
                            print(f"⚠️  Alternative download method failed: {e}")
                except Exception as e:
                    print(f"⚠️  Error getting folder contents: {e}")
                    # Fallback to recursive download
                    try:
                        self.syn.get(synapse_id, downloadLocation=entity_dir, recursive=True)
                        files = []
                        for root, dirs, filenames in os.walk(entity_dir):
                            for filename in filenames:
                                files.append(os.path.join(root, filename))
                        if files:
                            print(f"✅ Downloaded {len(files)} files")
                        else:
                            print(f"⚠️  No files found after download")
                    except Exception as e2:
                        print(f"❌ Download failed: {e2}")
            else:
                # It's a file, download it
                entity = self.syn.get(synapse_id, downloadLocation=output_dir)
                download_path = os.path.join(output_dir, entity.name if hasattr(entity, 'name') else synapse_id)
                
                if os.path.exists(download_path):
                    size = os.path.getsize(download_path) / (1024*1024)  # Size in MB
                    print(f"✅ Successfully downloaded {synapse_id}")
                    print(f"   Location: {download_path}")
                    print(f"   Size: {size:.2f} MB")
                else:
                    print(f"⚠️  Download reported success but file not found at {download_path}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to download {synapse_id}: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Synapse data from CSV file')
    parser.add_argument('--input', type=str, default='data_list_full.csv',
                       help='Input CSV file path (default: data_list_full.csv)')
    parser.add_argument('--output-dir', type=str, default='synapse_downloads',
                       help='Output directory for downloads (default: synapse_downloads)')
    parser.add_argument('--synapse-id', type=str, default=None,
                       help='Specific Synapse ID to download (if not provided, will find one from file)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list found Synapse IDs, do not download')
    
    args = parser.parse_args()
    
    # Handle both absolute and relative paths
    if not os.path.isabs(args.input):
        # Try current directory first
        if not os.path.exists(args.input):
            # Try in common locations
            possible_paths = [
                os.path.join(os.getcwd(), args.input),
                os.path.join(os.path.expanduser('~'), 'als_foundation_model', args.input),
                os.path.join('/home/ul/ul_neurorku/ul_oqn09/als_foundation_model', args.input),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    args.input = path
                    break
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        sys.exit(1)
    
    print(f"📁 Using input file: {args.input}")
    
    # Create output directory (handle both absolute and relative paths)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory: {args.output_dir}")
    
    downloader = SynapseDownloader()
    
    # Find Synapse IDs in the file
    synapse_ids = downloader.find_synapse_ids(args.input)
    
    if not synapse_ids:
        print("\n❌ No Synapse IDs found in the file!")
        print("   Looking for IDs in format: syn[numbers] (e.g., syn53421674)")
        sys.exit(1)
    
    if args.list_only:
        print(f"\n📊 Found {len(synapse_ids)} unique Synapse ID(s):\n")
        for i, (row_idx, synapse_id, context, row_data) in enumerate(synapse_ids, 1):
            # Try to get dataset_id or other identifier from row
            dataset_id = row_data.get('dataset_id', row_data.get('dataset_ID', row_data.get('Dataset_ID', 'N/A')))
            print(f"  {i}. {synapse_id} ({context})")
            if dataset_id and str(dataset_id) != 'nan' and str(dataset_id).strip():
                print(f"     Dataset: {dataset_id}")
        print("\n✅ Listing complete. Use --synapse-id to download a specific one.")
        return
    else:
        # Only show summary when not listing
        print(f"\n📊 Found {len(synapse_ids)} unique Synapse ID(s) in file")
    
    # Download
    if args.synapse_id:
        # Download specific ID
        if args.synapse_id.lower() not in [sid[1] for sid in synapse_ids]:
            print(f"\n⚠️  Warning: {args.synapse_id} not found in file, but attempting download anyway...")
        downloader.download_synapse_entity(args.synapse_id, args.output_dir)
    else:
        # Download the first one found
        if synapse_ids:
            first_id = synapse_ids[0][1]
            print(f"\n📥 Downloading first Synapse ID found: {first_id}")
            downloader.download_synapse_entity(first_id, args.output_dir)
        else:
            print("\n❌ No Synapse IDs to download!")


if __name__ == '__main__':
    main()

