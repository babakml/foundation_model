#!/bin/bash
# download_references_and_indices.sh
# Download reference genomes (FASTA + GTF) from Ensembl and build STAR indices
# Targets: Human (GRCh38), Mouse (GRCm39), Rat (mRatBN7.2)

set -euo pipefail

ROOT_DIR=${1:-"$HOME/als_foundation_model"}
REF_DIR="$ROOT_DIR/references"
LOG_DIR="$ROOT_DIR/logs"
THREADS=${THREADS:-32}

mkdir -p "$REF_DIR" "$LOG_DIR"

log() { echo "[$(date +'%F %T')] $*" | tee -a "$LOG_DIR/ref_download.log"; }

# Ensembl current base (adjust release if needed)
ENSEMBL_BASE="https://ftp.ensembl.org/pub/current_fasta"
ENSEMBL_GTF_BASE="https://ftp.ensembl.org/pub/current_gtf"

# Species map: name|fasta_subpath|gtf_species_dir|index_dir
SPECIES=(
  "homo_sapiens|homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz|homo_sapiens|GRCh38_STAR"
  "mus_musculus|mus_musculus/dna/Mus_musculus.GRCm39.dna.primary_assembly.fa.gz|mus_musculus|GRCm39_STAR"
  "rattus_norvegicus|rattus_norvegicus/dna/Rattus_norvegicus.mRatBN7.2.dna.primary_assembly.fa.gz|rattus_norvegicus|mRatBN7.2_STAR"
)

# Helper to fetch the latest non-abinitio GTF filename dynamically
fetch_gtf() {
  local species_key="$1" # e.g., homo_sapiens
  local listing
  listing=$(curl -sL "${ENSEMBL_GTF_BASE}/${species_key}/" || true)
  # Prefer non-abinitio .gtf.gz
  local gtf
  gtf=$(printf "%s" "$listing" | grep -Eo 'href="[^" ]+\.gtf\.gz"' | sed 's/href="//;s/"//' | grep -v abinitio | head -n1)
  if [[ -z "$gtf" ]]; then
    # Fallback to any .gtf.gz (including abinitio)
    gtf=$(printf "%s" "$listing" | grep -Eo 'href="[^" ]+\.gtf\.gz"' | sed 's/href="//;s/"//' | head -n1)
  fi
  printf "%s" "$gtf"
}

for entry in "${SPECIES[@]}"; do
  IFS='|' read -r sp_key fasta_rel gtf_species idx_dir <<<"$entry"
  sp_dir="$REF_DIR/$sp_key"
  idx_path="$sp_dir/$idx_dir"
  mkdir -p "$sp_dir"

  log "==== ${sp_key} ===="

  # Resolve FASTA URL
  FASTA_URL="${ENSEMBL_BASE}/${fasta_rel}"
  FASTA_FILE="$sp_dir/$(basename "$FASTA_URL")"

  # Resolve GTF URL dynamically (prefer non-abinitio)
  log "Discovering latest GTF for ${sp_key}..."
  latest_gtf=$(fetch_gtf "$gtf_species" || true)
  if [[ -z "$latest_gtf" ]]; then
    log "❌ Could not find GTF for ${sp_key}. Skipping."
    continue
  fi
  GTF_URL="${ENSEMBL_GTF_BASE}/${gtf_species}/${latest_gtf}"
  GTF_FILE="$sp_dir/$(basename "$GTF_URL")"

  # Download FASTA
  if [[ ! -s "$FASTA_FILE" ]]; then
    log "📥 Downloading FASTA: $FASTA_URL"
    curl -L "$FASTA_URL" -o "$FASTA_FILE"
  else
    log "✅ FASTA exists: $(basename "$FASTA_FILE")"
  fi

  # Download GTF
  if [[ ! -s "$GTF_FILE" ]]; then
    log "📥 Downloading GTF: $GTF_URL"
    curl -L "$GTF_URL" -o "$GTF_FILE"
  else
    log "✅ GTF exists: $(basename "$GTF_FILE")"
  fi

  # Decompress if needed
  if [[ "$FASTA_FILE" == *.gz ]]; then
    if [[ ! -s "${FASTA_FILE%.gz}" ]]; then
      log "🗜️  Decompressing FASTA..."
      gunzip -c "$FASTA_FILE" > "${FASTA_FILE%.gz}"
    fi
    FASTA_PLAIN="${FASTA_FILE%.gz}"
  else
    FASTA_PLAIN="$FASTA_FILE"
  fi

  if [[ "$GTF_FILE" == *.gz ]]; then
    if [[ ! -s "${GTF_FILE%.gz}" ]]; then
      log "🗜️  Decompressing GTF..."
      gunzip -c "$GTF_FILE" > "${GTF_FILE%.gz}"
    fi
    GTF_PLAIN="${GTF_FILE%.gz}"
  else
    GTF_PLAIN="$GTF_FILE"
  fi

  # Validate GTF contains exon lines
  if ! grep -q $'\texon\t' "$GTF_PLAIN"; then
    log "❌ GTF has no exon features: $(basename "$GTF_PLAIN"). Trying to fetch a different GTF."
    # Remove current GTF and refetch (forcing non-abinitio fallback)
    rm -f "$GTF_FILE" "$GTF_PLAIN"
    latest_gtf=$(fetch_gtf "$gtf_species" || true)
    if [[ -n "$latest_gtf" ]]; then
      GTF_URL="${ENSEMBL_GTF_BASE}/${gtf_species}/${latest_gtf}"
      GTF_FILE="$sp_dir/$(basename "$GTF_URL")"
      curl -L "$GTF_URL" -o "$GTF_FILE"
      if [[ "$GTF_FILE" == *.gz ]]; then gunzip -c "$GTF_FILE" > "${GTF_FILE%.gz}"; GTF_PLAIN="${GTF_FILE%.gz}"; else GTF_PLAIN="$GTF_FILE"; fi
    fi
  fi

  # Build STAR index
  if [[ -d "$idx_path" && -s "$idx_path/SAindex" ]]; then
    log "✅ STAR index exists: $idx_path"
  else
    log "🧱 Building STAR index: $idx_path"
    mkdir -p "$idx_path"
    if ! command -v STAR >/dev/null 2>&1; then
      log "❌ STAR not found in PATH. Please load your module or conda env."
      exit 1
    fi
    STAR \
      --runThreadN "$THREADS" \
      --runMode genomeGenerate \
      --genomeDir "$idx_path" \
      --genomeFastaFiles "$FASTA_PLAIN" \
      --sjdbGTFfile "$GTF_PLAIN" \
      --sjdbOverhang 99
    log "✅ STAR index built: $idx_path"
  fi

done

log "🎉 Reference downloads and STAR indices are ready in $REF_DIR"
