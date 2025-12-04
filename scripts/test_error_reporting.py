#!/usr/bin/env python3
"""
Test script to demonstrate the error reporting functionality
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

def demo_error_reporting():
    """Demonstrate the error reporting functionality"""
    console = Console()
    
    # Demo 1: Download failure
    console.print("\n[bold red]Demo 1: Download Failure[/bold red]")
    error_text = """❌ [bold red]Download Failures[/bold red]
[yellow]Dataset:[/yellow] GSE123456
[red]Error:[/red] No data available from any source
[dim]Details:[/dim] Tried tar files, processed data, and SRA fallback
[dim]Batch:[/dim] 1 | [dim]Time:[/dim] 2024-01-15T10:30:45
[dim]Total Issues:[/dim] 1"""
    
    panel = Panel(
        error_text,
        title="[bold red]ISSUE REPORTED[/bold red]",
        border_style="red"
    )
    console.print(panel)
    
    # Demo 2: Processing failure
    console.print("\n[bold orange3]Demo 2: Processing Failure[/bold orange3]")
    error_text = """⚠️ [bold orange3]Processing Failures[/bold orange3]
[yellow]Dataset:[/yellow] GSE789012
[red]Error:[/red] Failed to load data file
[dim]Details:[/dim] File: /data/GSE789012/data.h5ad
[dim]Batch:[/dim] 2 | [dim]Time:[/dim] 2024-01-15T10:35:20
[dim]Total Issues:[/dim] 2"""
    
    panel = Panel(
        error_text,
        title="[bold orange3]ISSUE REPORTED[/bold orange3]",
        border_style="orange3"
    )
    console.print(panel)
    
    # Demo 3: Storage warning
    console.print("\n[bold yellow]Demo 3: Storage Warning[/bold yellow]")
    error_text = """💾 [bold yellow]Storage Warnings[/bold yellow]
[yellow]Dataset:[/yellow] SYSTEM
[red]Error:[/red] Low storage space: 45.2GB free (minimum: 50GB)
[dim]Details:[/dim] Used: 504.8GB, Total: 550GB
[dim]Batch:[/dim] 3 | [dim]Time:[/dim] 2024-01-15T10:40:15
[dim]Total Issues:[/dim] 3"""
    
    panel = Panel(
        error_text,
        title="[bold yellow]ISSUE REPORTED[/bold yellow]",
        border_style="yellow"
    )
    console.print(panel)
    
    # Demo 4: Data quality issue
    console.print("\n[bold magenta]Demo 4: Data Quality Issue[/bold magenta]")
    error_text = """🔍 [bold magenta]Data Quality Issues[/bold magenta]
[yellow]Dataset:[/yellow] GSE555666
[red]Error:[/red] No data files found after download
[dim]Details:[/dim] Expected formats: H5, seurat
[dim]Batch:[/dim] 4 | [dim]Time:[/dim] 2024-01-15T10:45:30
[dim]Total Issues:[/dim] 4"""
    
    panel = Panel(
        error_text,
        title="[bold magenta]ISSUE REPORTED[/bold magenta]",
        border_style="magenta"
    )
    console.print(panel)
    
    # Demo 5: Network issue
    console.print("\n[bold blue]Demo 5: Network Issue[/bold blue]")
    error_text = """🌐 [bold blue]Network Issues[/bold blue]
[yellow]Dataset:[/yellow] GSE777888
[red]Error:[/red] Tar file download failed: Connection timeout
[dim]Details:[/dim] Network or server issue
[dim]Batch:[/dim] 5 | [dim]Time:[/dim] 2024-01-15T10:50:45
[dim]Total Issues:[/dim] 5"""
    
    panel = Panel(
        error_text,
        title="[bold blue]ISSUE REPORTED[/bold blue]",
        border_style="blue"
    )
    console.print(panel)

def demo_issue_summary():
    """Demonstrate the issue summary functionality"""
    console = Console()
    
    console.print("\n[bold green]Demo 6: Issue Summary[/bold green]")
    
    # Create issue summary table
    table = Table(title="Issue Summary - 5 Total Issues")
    table.add_column("Issue Type", style="cyan")
    table.add_column("Count", style="yellow")
    table.add_column("Examples", style="white", max_width=60)
    
    table.add_row(
        "Download Failures",
        "1",
        "GSE123456: No data available from any source..."
    )
    table.add_row(
        "Processing Failures",
        "1",
        "GSE789012: Failed to load data file..."
    )
    table.add_row(
        "Storage Warnings",
        "1",
        "SYSTEM: Low storage space: 45.2GB free..."
    )
    table.add_row(
        "Data Quality Issues",
        "1",
        "GSE555666: No data files found after download..."
    )
    table.add_row(
        "Network Issues",
        "1",
        "GSE777888: Tar file download failed: Connection timeout..."
    )
    
    console.print(table)
    
    # Print detailed issue breakdown
    console.print("\n[bold]Download Failures:[/bold]")
    console.print("  • [yellow]GSE123456[/yellow]: No data available from any source")
    
    console.print("\n[bold]Processing Failures:[/bold]")
    console.print("  • [yellow]GSE789012[/yellow]: Failed to load data file")
    
    console.print("\n[bold]Storage Warnings:[/bold]")
    console.print("  • [yellow]SYSTEM[/yellow]: Low storage space: 45.2GB free (minimum: 50GB)")
    
    console.print("\n[bold]Data Quality Issues:[/bold]")
    console.print("  • [yellow]GSE555666[/yellow]: No data files found after download")
    
    console.print("\n[bold]Network Issues:[/bold]")
    console.print("  • [yellow]GSE777888[/yellow]: Tar file download failed: Connection timeout")

def demo_success_case():
    """Demonstrate the success case with no issues"""
    console = Console()
    
    console.print("\n[bold green]Demo 7: Success Case (No Issues)[/bold green]")
    
    success_panel = Panel(
        "[bold green]✅ No issues encountered![/bold green]\nAll datasets processed successfully.",
        title="[bold green]PIPELINE SUCCESS[/bold green]",
        border_style="green"
    )
    console.print(success_panel)

def demo_real_time_flow():
    """Demonstrate real-time error reporting flow"""
    console = Console()
    
    console.print("\n[bold green]Demo 8: Real-Time Error Flow[/bold green]")
    
    # Simulate a real pipeline run with errors
    scenarios = [
        ("Starting pipeline", "Initializing streaming data processing", None),
        ("Downloading dataset", "Repository: GEO, Type: H5, seurat", "GSE306676"),
        ("Download complete", "Data saved to /data/GSE306676", "GSE306676"),
        ("Processing dataset", "Data type: H5, seurat", "GSE306676"),
        ("Processing complete", "Final: 12000 cells, 1500 genes", "GSE306676"),
        ("Downloading dataset", "Repository: GEO, Type: SRA", "GSE123456"),
        ("Download failed", "No data available from any source", "GSE123456"),  # ERROR
        ("Downloading dataset", "Repository: GEO, Type: MTX, TSV", "GSE789012"),
        ("Processing dataset", "Data type: MTX, TSV", "GSE789012"),
        ("Processing failed", "Failed to load data file", "GSE789012"),  # ERROR
        ("Checking storage", "Monitoring disk usage", None),
        ("Low storage detected", "Cleaning up batch data", None),  # WARNING
        ("Batch complete", "Finished batch 1/88", None),
    ]
    
    for i, (phase, details, dataset_id) in enumerate(scenarios):
        # Check if this is an error/warning
        if "failed" in phase.lower() or "low storage" in phase.lower():
            if "failed" in phase.lower():
                if "download" in phase.lower():
                    error_type = "download_failures"
                    icon = "❌"
                    color = "red"
                else:
                    error_type = "processing_failures"
                    icon = "⚠️"
                    color = "orange3"
            else:
                error_type = "storage_warnings"
                icon = "💾"
                color = "yellow"
            
            # Print error report
            error_text = f"{icon} [bold {color}]{error_type.replace('_', ' ').title()}[/bold {color}]"
            error_text += f"\n[yellow]Dataset:[/yellow] {dataset_id or 'SYSTEM'}"
            error_text += f"\n[red]Error:[/red] {details}"
            error_text += f"\n[dim]Batch:[/dim] 1 | [dim]Time:[/dim] {datetime.now().isoformat()}"
            error_text += f"\n[dim]Total Issues:[/dim] {i//3 + 1}"
            
            panel = Panel(
                error_text,
                title=f"[bold {color}]ISSUE REPORTED[/bold {color}]",
                border_style=color
            )
            console.print(panel)
        else:
            # Print normal status
            status_text = f"[bold blue]{phase}[/bold blue]"
            if dataset_id:
                status_text += f" - [yellow]{dataset_id}[/yellow]"
            if details:
                status_text += f"\n[dim]{details}[/dim]"
            status_text += f"\n[dim]Elapsed: 0:{i//60:02d}:{i%60:02d} | Storage: {45.2 + i*0.1:.1f}GB used, {504.8 - i*0.1:.1f}GB free[/dim]"
            status_text += f"\n[dim]Overall Progress: {i+1}/88 ({(i+1)/88*100:.1f}%)[/dim]"
            
            panel = Panel(
                status_text,
                title="[bold green]ALS Foundation Model Pipeline[/bold green]",
                border_style="green"
            )
            console.print(panel)
        
        time.sleep(0.3)  # Simulate processing time

def main():
    """Run the demo"""
    console = Console()
    
    console.print("=" * 60)
    console.print("[bold green]ALS FOUNDATION MODEL - ERROR REPORTING DEMO[/bold green]")
    console.print("=" * 60)
    
    console.print("\nThis demo shows how the pipeline reports issues in real-time:")
    console.print("• [bold red]❌ Download Failures[/bold red] - When datasets can't be downloaded")
    console.print("• [bold orange3]⚠️ Processing Failures[/bold orange3] - When data processing fails")
    console.print("• [bold yellow]💾 Storage Warnings[/bold yellow] - When storage space is low")
    console.print("• [bold magenta]🔍 Data Quality Issues[/bold magenta] - When data quality is poor")
    console.print("• [bold blue]🌐 Network Issues[/bold blue] - When network/server problems occur")
    
    demo_error_reporting()
    demo_issue_summary()
    demo_success_case()
    demo_real_time_flow()
    
    console.print("\n" + "=" * 60)
    console.print("[bold green]Error reporting demo completed![/bold green]")
    console.print("The pipeline will show these error reports in real-time during execution.")
    console.print("=" * 60)

if __name__ == "__main__":
    main()
