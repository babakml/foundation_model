#!/usr/bin/env python3
"""
Test script to demonstrate the status display functionality
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

def demo_status_display():
    """Demonstrate the status display functionality"""
    console = Console()
    
    # Demo 1: Basic status panel
    console.print("\n[bold green]Demo 1: Basic Status Panel[/bold green]")
    status_text = """[bold blue]Downloading dataset[/bold blue] - [yellow]GSE306676[/yellow]
[dim]Repository: GEO, Type: H5, seurat[/dim]
[dim]Elapsed: 0:02:15 | Storage: 45.2GB used, 504.8GB free[/dim]
[dim]Overall Progress: 1/88 (1.1%)[/dim]
[dim]Batch Progress: 2/3 (66.7%)[/dim]"""
    
    panel = Panel(
        status_text,
        title="[bold green]ALS Foundation Model Pipeline[/bold green]",
        border_style="green"
    )
    console.print(panel)
    
    # Demo 2: Progress table
    console.print("\n[bold green]Demo 2: Batch Progress Table[/bold green]")
    table = Table(title="Batch 1 - 3 datasets")
    table.add_column("Dataset ID", style="cyan")
    table.add_column("Title", style="white", max_width=50)
    table.add_column("Repository", style="magenta")
    table.add_column("Data Type", style="yellow")
    table.add_column("Status", style="green")
    
    table.add_row(
        "GSE306676",
        "An emergent disease-associated motor neuron state...",
        "GEO",
        "H5, seurat",
        "✅ Complete"
    )
    table.add_row(
        "GSE306675",
        "An emergent disease-associated motor neuron state...",
        "GEO",
        "H5, TBI, TSV",
        "🔄 Processing"
    )
    table.add_row(
        "GSE298187",
        "A Mechanistic basis of fast myofiber vulnerability...",
        "GEO",
        "MTX, TSV, seurat",
        "⏳ Pending"
    )
    
    console.print(table)
    
    # Demo 3: Different status phases
    console.print("\n[bold green]Demo 3: Different Status Phases[/bold green]")
    
    phases = [
        ("Initializing", "Setting up directories and configuration"),
        ("Loading dataset list", "Reading CSV file with 262 datasets"),
        ("Downloading dataset", "Repository: GEO, Type: H5, seurat", "GSE306676"),
        ("Downloading from GEO", "Trying tar file first", "GSE306676"),
        ("Tar download successful", "Extracting archive", "GSE306676"),
        ("Processing dataset", "Data type: H5, seurat", "GSE306676"),
        ("Finding data files", "Scanning directory structure", "GSE306676"),
        ("Loading data file", "Found 3 files, using data.h5ad", "GSE306676"),
        ("Quality control", "Starting with 15420 cells, 2000 genes", "GSE306676"),
        ("Normalizing data", "Applying log transform and scaling", "GSE306676"),
        ("Extracting features", "Finding highly variable genes", "GSE306676"),
        ("Saving processed data", "Writing to processed/GSE306676_processed.h5ad", "GSE306676"),
        ("Processing complete", "Final: 12000 cells, 1500 genes", "GSE306676"),
        ("Integrating batch", "Combining 3 datasets"),
        ("Updating model", "Incremental training on new data"),
        ("Batch integration complete", "Model updated with 35000 cells"),
        ("Cleaning up", "Removing temporary files"),
        ("Batch complete", "Finished batch 1/88")
    ]
    
    for i, phase_info in enumerate(phases):
        if len(phase_info) == 2:
            phase, details = phase_info
            dataset_id = None
        else:
            phase, details, dataset_id = phase_info
        
        # Simulate status display
        status_text = f"[bold blue]{phase}[/bold blue]"
        if dataset_id:
            status_text += f" - [yellow]{dataset_id}[/yellow]"
        if details:
            status_text += f"\n[dim]{details}[/dim]"
        
        status_text += f"\n[dim]Elapsed: 0:{i//60:02d}:{i%60:02d} | Storage: {45.2 + i*0.1:.1f}GB used, {504.8 - i*0.1:.1f}GB free[/dim]"
        status_text += f"\n[dim]Overall Progress: {i+1}/88 ({(i+1)/88*100:.1f}%)[/dim]"
        status_text += f"\n[dim]Batch Progress: {i%3 + 1}/3 ({(i%3 + 1)/3*100:.1f}%)[/dim]"
        
        panel = Panel(
            status_text,
            title="[bold green]ALS Foundation Model Pipeline[/bold green]",
            border_style="green"
        )
        
        console.print(panel)
        time.sleep(0.5)  # Simulate processing time
    
    # Demo 4: Error status
    console.print("\n[bold green]Demo 4: Error Status[/bold green]")
    error_status = """[bold red]Download failed[/bold red] - [yellow]GSE123456[/yellow]
[dim]No data available from any source[/dim]
[dim]Elapsed: 0:05:30 | Storage: 45.5GB used, 504.5GB free[/dim]
[dim]Overall Progress: 5/88 (5.7%)[/dim]
[dim]Batch Progress: 2/3 (66.7%)[/dim]"""
    
    error_panel = Panel(
        error_status,
        title="[bold green]ALS Foundation Model Pipeline[/bold green]",
        border_style="red"
    )
    console.print(error_panel)
    
    # Demo 5: Completion status
    console.print("\n[bold green]Demo 5: Completion Status[/bold green]")
    completion_status = """[bold green]Pipeline complete[/bold green]
[dim]All processing finished successfully[/dim]
[dim]Elapsed: 2:15:30 | Storage: 180.5GB used, 369.5GB free[/dim]
[dim]Overall Progress: 88/88 (100.0%)[/dim]
[dim]Total datasets processed: 262[/dim]
[dim]Model checkpoints saved: 88[/dim]"""
    
    completion_panel = Panel(
        completion_status,
        title="[bold green]ALS Foundation Model Pipeline[/bold green]",
        border_style="green"
    )
    console.print(completion_panel)

def main():
    """Run the demo"""
    console = Console()
    
    console.print("=" * 60)
    console.print("[bold green]ALS FOUNDATION MODEL - STATUS DISPLAY DEMO[/bold green]")
    console.print("=" * 60)
    
    console.print("\nThis demo shows how the pipeline will display real-time status updates:")
    console.print("• [bold blue]Current phase[/bold blue] - What the pipeline is doing")
    console.print("• [yellow]Dataset ID[/yellow] - Which dataset is being processed")
    console.print("• [dim]Details[/dim] - Additional information about the current operation")
    console.print("• [dim]Progress tracking[/dim] - Overall and batch progress")
    console.print("• [dim]Storage monitoring[/dim] - Real-time disk usage")
    console.print("• [dim]Elapsed time[/dim] - How long the pipeline has been running")
    
    demo_status_display()
    
    console.print("\n" + "=" * 60)
    console.print("[bold green]Status display demo completed![/bold green]")
    console.print("The pipeline will show these status updates in real-time during execution.")
    console.print("=" * 60)

if __name__ == "__main__":
    main()
