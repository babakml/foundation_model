#!/usr/bin/env python3
"""
Storage monitoring utility for ALS Foundation Model pipeline
"""

import os
import psutil
import json
import time
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageMonitor:
    """Monitor storage usage and provide recommendations"""
    
    def __init__(self, base_dir: str, max_gb: float = 550):
        self.base_dir = Path(base_dir)
        self.max_gb = max_gb
        self.max_bytes = max_gb * (1024**3)
        
    def get_directory_size(self, path: Path) -> float:
        """Get directory size in GB"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except (OSError, IOError) as e:
            logger.warning(f"Error calculating size for {path}: {e}")
        
        return total_size / (1024**3)
    
    def get_storage_breakdown(self) -> dict:
        """Get detailed storage breakdown"""
        breakdown = {}
        
        if not self.base_dir.exists():
            return breakdown
        
        for subdir in self.base_dir.iterdir():
            if subdir.is_dir():
                size_gb = self.get_directory_size(subdir)
                breakdown[subdir.name] = {
                    'size_gb': round(size_gb, 2),
                    'percentage': round((size_gb / self.max_gb) * 100, 2)
                }
        
        return breakdown
    
    def get_system_storage(self) -> dict:
        """Get system storage information"""
        usage = psutil.disk_usage(self.base_dir)
        
        return {
            'total_gb': round(usage.total / (1024**3), 2),
            'used_gb': round((usage.total - usage.free) / (1024**3), 2),
            'free_gb': round(usage.free / (1024**3), 2),
            'usage_percentage': round(((usage.total - usage.free) / usage.total) * 100, 2)
        }
    
    def get_large_files(self, min_size_gb: float = 1.0) -> list:
        """Find large files that could be cleaned up"""
        large_files = []
        min_size_bytes = min_size_gb * (1024**3)
        
        try:
            for dirpath, dirnames, filenames in os.walk(self.base_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        size = os.path.getsize(filepath)
                        if size > min_size_bytes:
                            large_files.append({
                                'path': filepath,
                                'size_gb': round(size / (1024**3), 2),
                                'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                            })
        except (OSError, IOError) as e:
            logger.warning(f"Error scanning for large files: {e}")
        
        return sorted(large_files, key=lambda x: x['size_gb'], reverse=True)
    
    def get_cleanup_recommendations(self) -> list:
        """Get recommendations for cleaning up storage"""
        recommendations = []
        breakdown = self.get_storage_breakdown()
        system_storage = self.get_system_storage()
        
        # Check if approaching limit
        if system_storage['usage_percentage'] > 80:
            recommendations.append({
                'priority': 'high',
                'action': 'cleanup_immediate',
                'message': f"Storage usage at {system_storage['usage_percentage']}%. Immediate cleanup needed."
            })
        
        # Check specific directories
        if 'raw' in breakdown and breakdown['raw']['size_gb'] > 100:
            recommendations.append({
                'priority': 'medium',
                'action': 'cleanup_raw_data',
                'message': f"Raw data directory using {breakdown['raw']['size_gb']}GB. Consider cleaning processed files."
            })
        
        if 'cache' in breakdown and breakdown['cache']['size_gb'] > 50:
            recommendations.append({
                'priority': 'low',
                'action': 'cleanup_cache',
                'message': f"Cache directory using {breakdown['cache']['size_gb']}GB. Safe to clean."
            })
        
        # Check for old checkpoints
        checkpoints_dir = self.base_dir / 'model' / 'checkpoints'
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob('*.pkl'))
            if len(checkpoint_files) > 5:
                recommendations.append({
                    'priority': 'medium',
                    'action': 'cleanup_old_checkpoints',
                    'message': f"Found {len(checkpoint_files)} checkpoint files. Consider keeping only recent ones."
                })
        
        return recommendations
    
    def generate_report(self) -> dict:
        """Generate comprehensive storage report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_storage': self.get_system_storage(),
            'directory_breakdown': self.get_storage_breakdown(),
            'large_files': self.get_large_files(),
            'recommendations': self.get_cleanup_recommendations()
        }
        
        return report
    
    def print_report(self):
        """Print formatted storage report"""
        report = self.generate_report()
        
        print("=" * 60)
        print("ALS FOUNDATION MODEL - STORAGE REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        # System storage
        sys_storage = report['system_storage']
        print("SYSTEM STORAGE:")
        print(f"  Total: {sys_storage['total_gb']} GB")
        print(f"  Used:  {sys_storage['used_gb']} GB ({sys_storage['usage_percentage']}%)")
        print(f"  Free:  {sys_storage['free_gb']} GB")
        print()
        
        # Directory breakdown
        print("DIRECTORY BREAKDOWN:")
        for dir_name, info in report['directory_breakdown'].items():
            print(f"  {dir_name:15} {info['size_gb']:8.2f} GB ({info['percentage']:5.1f}%)")
        print()
        
        # Large files
        if report['large_files']:
            print("LARGE FILES (>1GB):")
            for file_info in report['large_files'][:10]:  # Show top 10
                print(f"  {file_info['size_gb']:6.2f} GB - {file_info['path']}")
            print()
        
        # Recommendations
        if report['recommendations']:
            print("RECOMMENDATIONS:")
            for rec in report['recommendations']:
                priority_color = {
                    'high': '\033[91m',    # Red
                    'medium': '\033[93m',  # Yellow
                    'low': '\033[92m'      # Green
                }.get(rec['priority'], '')
                reset_color = '\033[0m'
                
                print(f"  {priority_color}[{rec['priority'].upper()}]{reset_color} {rec['message']}")
            print()
        
        print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor storage for ALS Foundation Model')
    parser.add_argument('--base-dir', default='/scratch/username/als_foundation',
                       help='Base directory to monitor')
    parser.add_argument('--max-gb', type=float, default=550,
                       help='Maximum storage limit in GB')
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode - update every 30 seconds')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    monitor = StorageMonitor(args.base_dir, args.max_gb)
    
    if args.watch:
        try:
            while True:
                os.system('clear')
                monitor.print_report()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    elif args.json:
        report = monitor.generate_report()
        print(json.dumps(report, indent=2))
    else:
        monitor.print_report()

if __name__ == "__main__":
    main()
