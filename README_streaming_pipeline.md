# Streaming Data Processing Pipeline for ALS Foundation Model

## Overview

This pipeline processes single-cell transcriptomics datasets in batches to work within storage constraints (550GB) while building a comprehensive ALS foundation model. The pipeline downloads, processes, integrates, and trains on data incrementally, then cleans up to free space for the next batch.

## Key Features

- **Storage Efficient**: Never exceeds 550GB limit
- **Incremental Learning**: Model improves with each batch
- **Resumable**: Can restart from any batch if interrupted
- **Automated Cleanup**: Removes processed data to free space
- **Cluster Optimized**: Designed for HPC environments

## Pipeline Architecture

```
Download → Process → Integrate → Train → Cleanup → Repeat
    ↓         ↓         ↓         ↓        ↓
  50-100GB  30-50GB   20-30GB   10-20GB   Free Space
```

## Storage Allocation (550GB Total)

- **Data Directory (200GB)**: Current batch processing
- **Model Directory (200GB)**: Checkpoints and embeddings
- **Cache Directory (100GB)**: Temporary processing
- **Outputs Directory (50GB)**: Final results

## Quick Start

### 1. Setup Environment

```bash
# On the cluster
sbatch scripts/setup_environment.sh
```

### 2. Prepare Dataset List

Create an Excel file with your datasets containing columns:
- `dataset_id`: Unique identifier
- `download_url`: URL or accession number
- `type`: 'geo', 'direct', or 'sra'
- `disease_status`: 'ALS', 'control', etc.
- `tissue_type`: 'spinal_cord', 'motor_cortex', etc.

### 3. Configure Pipeline

Edit `configs/streaming_config.json`:
```json
{
  "base_dir": "/scratch/username/als_foundation",
  "dataset_list_path": "/home/username/als_datasets.xlsx",
  "batch_size": 3,
  "max_storage_gb": 550
}
```

### 4. Run Pipeline

```bash
# Submit job to cluster
sbatch scripts/run_streaming_pipeline.sh

# Monitor progress
tail -f streaming_pipeline_*.out
```

### 5. Monitor Storage

```bash
# Check storage usage
python scripts/monitor_storage.py --base-dir /scratch/username/als_foundation

# Watch mode (updates every 30 seconds)
python scripts/monitor_storage.py --watch
```

## Pipeline Components

### 1. Streaming Pipeline (`src/streaming_pipeline.py`)

Main pipeline class that orchestrates the entire process:
- Downloads datasets in batches
- Performs quality control and normalization
- Integrates datasets within each batch
- Updates the foundation model incrementally
- Cleans up processed data

### 2. Configuration (`configs/streaming_config.json`)

Centralized configuration for:
- Storage paths and limits
- Processing parameters
- Model architecture
- Quality control thresholds
- Cluster settings

### 3. Cluster Scripts

- `scripts/setup_environment.sh`: Environment setup
- `scripts/run_streaming_pipeline.sh`: SLURM job script
- `scripts/monitor_storage.py`: Storage monitoring utility

## Data Processing Workflow

### 1. Download Phase
- Downloads 2-3 datasets per batch (50-100GB)
- Supports GEO, SRA, and direct downloads
- Handles different file formats (FASTQ, H5AD, Loom, RDS)

### 2. Processing Phase
- Quality control (cell/gene filtering)
- Normalization and scaling
- Feature extraction
- Batch correction

### 3. Integration Phase
- Concatenates datasets within batch
- Performs batch correction
- Adds metadata annotations

### 4. Training Phase
- Incremental model updates
- Embedding learning
- Checkpoint saving

### 5. Cleanup Phase
- Removes raw data files
- Removes processed data (keeps latest)
- Cleans cache directories
- Archives old checkpoints

## Storage Management

### Automatic Cleanup
- Raw data removed after processing
- Processed data removed after integration
- Cache cleared between batches
- Old checkpoints archived

### Manual Cleanup
```bash
# Check what can be cleaned
python scripts/monitor_storage.py

# Clean specific directories
rm -rf /scratch/username/als_foundation/data/raw/*
rm -rf /scratch/username/als_foundation/cache/*
```

### Storage Monitoring
```bash
# Real-time monitoring
python scripts/monitor_storage.py --watch

# Generate report
python scripts/monitor_storage.py --json > storage_report.json
```

## Model Architecture

The foundation model is designed for incremental learning:

- **Input**: Single-cell gene expression matrices
- **Embedding Layer**: Maps genes to dense representations
- **Transformer Layers**: Captures gene-gene interactions
- **Output**: Cell type predictions and gene embeddings

### Incremental Training
- New data is integrated with existing model
- Embeddings are updated incrementally
- Model weights are fine-tuned on new batches
- Checkpoints are saved after each batch

## Error Handling and Recovery

### Resumable Pipeline
- Saves progress after each batch
- Can restart from last completed batch
- Maintains model state across restarts

### Error Recovery
- Failed downloads are retried
- Processing errors are logged and skipped
- Model state is preserved on errors

### Logging
- Comprehensive logging to file and console
- Progress tracking and error reporting
- Storage usage monitoring

## Performance Optimization

### Parallel Processing
- Downloads can run in parallel
- Processing uses multiple CPU cores
- GPU acceleration for model training

### Memory Management
- Processes data in chunks
- Clears memory between batches
- Efficient data structures

### Storage Optimization
- Compressed data formats
- Efficient file I/O
- Minimal data duplication

## Troubleshooting

### Common Issues

1. **Storage Full**
   ```bash
   # Check storage usage
   python scripts/monitor_storage.py
   
   # Clean up manually
   rm -rf /scratch/username/als_foundation/data/raw/*
   ```

2. **Download Failures**
   - Check network connectivity
   - Verify dataset URLs
   - Check SRA toolkit installation

3. **Processing Errors**
   - Check data format compatibility
   - Verify quality control parameters
   - Check memory usage

4. **Model Training Issues**
   - Check GPU availability
   - Verify CUDA installation
   - Check model configuration

### Log Analysis
```bash
# Check pipeline logs
tail -f streaming_pipeline.log

# Check SLURM logs
tail -f streaming_pipeline_*.out
tail -f streaming_pipeline_*.err
```

## Best Practices

1. **Start Small**: Begin with 2-3 datasets to test the pipeline
2. **Monitor Storage**: Use the monitoring script regularly
3. **Backup Checkpoints**: Keep important model checkpoints
4. **Test Configuration**: Verify settings before long runs
5. **Document Changes**: Keep track of configuration modifications

## Future Enhancements

- Support for additional data formats
- Advanced batch correction methods
- Distributed training across multiple nodes
- Real-time model evaluation
- Integration with cloud storage

## Support

For issues or questions:
1. Check the logs for error messages
2. Use the storage monitoring script
3. Verify cluster resource availability
4. Check configuration parameters
