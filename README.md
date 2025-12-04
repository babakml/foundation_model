# Foundation Model

Single-cell RNA-seq data processing pipeline for biomedical research.

## Overview

This project provides a comprehensive pipeline for downloading, processing, and analyzing single-cell RNA-seq datasets. It includes:

- Automated dataset download from GEO and SRA
- Data processing and quality control
- Integration of multiple datasets
- Synapse repository integration for processed 10x Genomics data

## Features

- **Streaming Data Processing**: Memory-efficient processing of large datasets
- **Parallel Processing**: Concurrent download and processing for faster execution
- **SRA Integration**: Automated download and conversion of SRA data to FASTQ
- **Synapse Integration**: Download processed 10x Genomics data from Synapse
- **Reference Genome Alignment**: STAR alignment for FASTQ files
- **Download Tracking**: Resume functionality for interrupted runs

## Requirements

- Python 3.9+
- Conda environment (see `requirements.txt`)
- SLURM (for cluster execution)
- SRA Toolkit (for SRA data download)
- STAR aligner (for FASTQ alignment)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/foundation_model.git
cd foundation_model
```

2. Set up conda environment:
```bash
conda env create -f environment.yml
conda activate foundation_model
```

3. Install SRA Toolkit (on cluster):
```bash
bash scripts/install_sra_toolkit_fixed.sh
```

4. Configure Synapse (if using Synapse downloads):
```bash
synapse config
```

## Usage

### Main Pipeline

Run the parallel optimized pipeline:
```bash
sbatch scripts/run_parallel_optimized_pipeline.slurm
```

### Synapse Data Download

Download data from Synapse using IDs from CSV:
```bash
python download_synapse_data.py --input data_list_full.csv --synapse-id syn53421674
```

### Reference Genome Setup

Download and build STAR indices:
```bash
sbatch scripts/run_reference_download.slurm
```

### FASTQ Alignment

Align FASTQ files to reference genomes:
```bash
bash scripts/align_fastq_and_cleanup.sh /path/to/STAR/index 32
```

## Project Structure

```
foundation_model/
├── src/                    # Source code
│   ├── parallel_optimized_pipeline.py
│   ├── sra_download_fix.py
│   └── ...
├── scripts/                # Utility scripts
│   ├── run_parallel_optimized_pipeline.slurm
│   ├── download_references_and_indices.sh
│   └── ...
├── configs/                # Configuration files
│   └── streaming_config.json
├── data/                   # Data directories
│   ├── raw/               # Raw downloaded data
│   └── processed/         # Processed data
└── logs/                  # Log files
```

## Configuration

Edit `configs/streaming_config.json` to customize:
- Download and processing parameters
- Memory and CPU limits
- Storage management settings
- Quality control and normalization parameters

## License

[Add your license here]

## Contact

[Add contact information]
