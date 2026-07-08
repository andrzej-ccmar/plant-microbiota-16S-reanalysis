#!/bin/bash
#SBATCH --job-name=total_unified
#SBATCH --output=unified.%j.log
#SBATCH --error=unified.%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=all
#SBATCH --nodelist=ceta2
#SBATCH --mem=80G

set -euo pipefail

# Optional: if your cluster uses modules
# module load python/3.10
# module load vsearch
# module load blast


MODE="missing"          # <<< CHANGE THIS
REDO_LIST="redo_list.txt"

RECALC_ARGS=()
if [[ "$MODE" == "all" ]]; then
  RECALC_ARGS+=(--recalc all)
elif [[ "$MODE" == "selected" ]]; then
  RECALC_ARGS+=(--recalc selected --redo-list "$REDO_LIST")
elif [[ "$MODE" == "missing" ]]; then
  RECALC_ARGS+=(--recalc missing)
else
  echo "ERROR: MODE must be one of: all | selected | missing"
  exit 1
fi

python3 unified.py \
  --root . \
  "${RECALC_ARGS[@]}" \
  \
  --run-asv \
  --all-fasta all_samples.fasta \
  --silva-db /home/andrzej/total/SILVA_db \
  --metadata metadata_V7.csv \
  --threads 48 \
  --min-len 200 --max-len 350 \
  --exclude "source_folder=apple-Papp-Phytob" \
  \
  --make-niche-subsets --niche-col niche_category \
  --make-species-subsets --species-col my_host \
  \
  --postprocess \
  --do-shannon \
  --do-anosim --anosim-factor niche_category \
  --anosim-merge "root=endosphere|rhizoplane;other=aerial root|flower|fruit|seed|stem" \
  --anosim-pairs other_plus_rest \
  \
  --do-corr --corr-mode topn --corr-topn 10 --corr-method spearman \
  |& tee unified_run.%j.console.txt

#############################################################
## If you want fixed taxa for the correlations instead:

#  --do-corr --corr-rank phylum \
#  --corr-mode fixed --corr-method spearman \
#  --corr-taxa "Proteobacteria,Actinobacteriota,Bacteroidota,Firmicutes"

