#!/bin/bash
# align_fastq_and_cleanup.sh
# Align FASTQ files to reference genome (STAR) and remove FASTQs upon success
# Usage: align_fastq_and_cleanup.sh <genomeDir> [threads]

set -euo pipefail

GENOME_DIR=${1:-""}
THREADS=${2:-16}
ROOT_DIR=${ROOT_DIR:-"$HOME/foundation_model"}
RAW_DIR="$ROOT_DIR/data/raw"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ -z "$GENOME_DIR" ]]; then
  echo "Usage: $0 <STAR_genomeDir> [threads]"
  exit 1
fi

if ! command -v STAR >/dev/null 2>&1; then
  echo "❌ STAR not found in PATH. Please load your module or conda env."
  exit 1
fi

echo "🧬 Using STAR genomeDir: $GENOME_DIR"
echo "🧵 Threads: $THREADS"

datasets=$(find "$RAW_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

for ds in $datasets; do
  ds_id=$(basename "$ds")
  echo "\n==== Dataset: $ds_id ====" | tee -a "$LOG_DIR/align_$ds_id.log"

  # Find FASTQ files
  mapfile -t fq_files < <(find "$ds" -type f \( -name "*.fastq" -o -name "*.fastq.gz" -o -name "*.fq" -o -name "*.fq.gz" \) | sort)
  if [[ ${#fq_files[@]} -eq 0 ]]; then
    echo "  ⏭️  No FASTQ files found, skipping" | tee -a "$LOG_DIR/align_$ds_id.log"
    continue
  fi

  echo "  📁 Found ${#fq_files[@]} FASTQ files" | tee -a "$LOG_DIR/align_$ds_id.log"

  work_dir="$ds/alignment_output"
  mkdir -p "$work_dir"

  # Determine if gzip
  READ_CMD=""
  if ls "$ds"/*.gz >/dev/null 2>&1; then
    READ_CMD="--readFilesCommand zcat"
  fi

  # Align
  set +e
  STAR \
    --runThreadN "$THREADS" \
    --genomeDir "$GENOME_DIR" \
    --readFilesIn "${fq_files[@]}" \
    $READ_CMD \
    --outFileNamePrefix "$work_dir/" \
    --outSAMtype BAM SortedByCoordinate \
    --quantMode GeneCounts 2>&1 | tee -a "$LOG_DIR/align_$ds_id.log"
  status=$?
  set -e

  if [[ $status -eq 0 && -s "$work_dir/Aligned.sortedByCoord.out.bam" ]]; then
    echo "  ✅ Alignment succeeded for $ds_id" | tee -a "$LOG_DIR/align_$ds_id.log"

    # Remove FASTQs
    echo "  🗑️  Removing FASTQ files..." | tee -a "$LOG_DIR/align_$ds_id.log"
    find "$ds" -type f \( -name "*.fastq" -o -name "*.fastq.gz" -o -name "*.fq" -o -name "*.fq.gz" \) -delete
    echo "  ✅ FASTQ files removed for $ds_id" | tee -a "$LOG_DIR/align_$ds_id.log"
  else
    echo "  ❌ Alignment failed for $ds_id (status=$status)" | tee -a "$LOG_DIR/align_$ds_id.log"
  fi

done

echo "\n🎉 Alignment pass completed. Check $LOG_DIR for logs."






