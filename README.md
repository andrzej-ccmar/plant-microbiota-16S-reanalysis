# plant-microbiota-16S-reanalysis
Code and workflows for cross-study reanalysis of publicly available plant-associated 16S rRNA microbiota datasets. Includes read processing, ASV inference, taxonomy assignment, diversity analysis, ordination, PERMANOVA/BETADISP, and visualisation of plant niche and host-associated microbiota patterns.


*** how to run it

python3 code1_accession_to_final_fasta.py \
   --sra-table-glob "./code1_test/test_paper/SraRunTable*.csv"  \
   --outdir "./code1_out/studyname/"  \
   --workdir "./code1_test/test_paper" \
   --threads 4  \
   --mito-db "./Mito_db" \
   --chloro-db "./Chloro_db" \
   --clean





*** note - it is best to examine the SraRunTable.csv yourself and remove any sample you don't want. Also remove any WGS samples (they may be indicated as WGS or you may use the total file size) - as WGS will massively slow down the code with their download, merging etc. and will be deleted nevertheless




