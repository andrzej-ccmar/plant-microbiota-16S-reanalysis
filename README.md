# plant-microbiota-16S-reanalysis
Code and workflows for cross-study reanalysis of publicly available plant-associated 16S rRNA microbiota datasets. Includes read processing, ASV inference, taxonomy assignment, diversity analysis, ordination, PERMANOVA/BETADISP, and visualisation of plant niche and host-associated microbiota patterns.


*** how to run it

### code 1

python3 code1_accession_to_final_fasta.py \
   --sra-table-glob "./code1_test/test_paper/SraRunTable*.csv"  \
   --outdir "./code1_out/studyname/"  \
   --workdir "./code1_test/test_paper" \
   --threads 4  \
   --mito-db "./Mito_db" \
   --chloro-db "./Chloro_db" \
   --clean





*** note - it is best to examine the SraRunTable.csv yourself and remove any sample you don't want. Also remove any WGS samples (they may be indicated as WGS or you may use the total file size) - as WGS will massively slow down the code with their download, merging etc. and will be deleted nevertheless


# code 2
python3 code2_rename_merge_check.py  \
 --root /path_to_fasta_files/ \
 --metadata /path_to_metadata_file \
 --min-len 220 --max-len 255


*** example
python3 code2_rename_merge_check.py  \
 --root /usr/local/storage/ebi-data/code1_out/ \
 --metadata /usr/local/storage/ebi-data/metadata_onlyhere/metadata_17March.csv \
 --min-len 220 --max-len 255


#code 3 - unified
The code is designed for HPC - the code was run using slurm

Copy codes into a folder containing all_samples.fasta and the metadata file. 
Make sure the SILVAdb (or other database of choice) is available
i.e. makeblastdb -dbtype nucl -in your_db_in_fasta_format -out SILVAdb

You may need to change the node and/or the input method to match your HPC specifications. Alternatively the code should be able to run on local PC, however it may run out of RAM memory 



#code4 - taxonomy - panel-based code - upon running a new tab in Firefox should open - if not copy the address manually into your web browser
#code5 - pcoa (as code4)
#code6 - shannon (as code5)
#code7 - shannon, betadisp, and within study permanova - run using slurm
python3 shannon_compute_all_hpc.py

echo "Job started on: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"

echo ""
echo "============================================================"
echo "Running PERMANOVA: ASV endosphere_noBoechera, host_category"
echo "============================================================"

python3 code_to_permanova_ASV_endosphere_noBoechera_host_category.py \
  --dataset-dir saved_matrices_ASV/endosphere_noBoechera \
  --workers ${SLURM_CPUS_PER_TASK} \
  --permutations 99900

echo ""
echo "============================================================"
echo "Running betadisper: ASV endosphere_noBoechera, host_category"
echo "============================================================"

python3 code_to_betadisper_ASV_endosphere_noBoechera_host_category_direct.py \
  --matrix saved_matrices_ASV/endosphere_noBoechera/bc_matrix_ASV.csv \
  --metadata metadata_17March_with_taxa.csv \
  --outdir saved_matrices_ASV/endosphere_noBoechera/permanova_reports \
  --workers ${SLURM_CPUS_PER_TASK} \
  --permutations 9999 \
  --min-group-size 2

echo ""
echo "End time: $(date)"
echo "Job finished"

#code7 global PERMANOVA - run using slurm

