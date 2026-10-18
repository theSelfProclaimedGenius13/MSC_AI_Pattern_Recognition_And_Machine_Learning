# Project 2: Prepare NCBI AMR data for the seven selected organism groups

This starter kit uses your existing `isolates.csv` export from the NCBI Pathogen Detection Isolates Browser. It deliberately keeps only these seven groups:

- Salmonella enterica
- E.coli and Shigella
- Campylobacter jejuni
- Klebsiella pneumoniae
- Staphylococcus aureus
- Mycobacterium tuberculosis
- Acinetobacter baumannii

## 1. Put the files together

Put `project2_download_amr_data.py` and your downloaded `isolates.csv` in the same folder.

## 2. Install Python dependency

Open PowerShell in that folder and run:

```powershell
py -m pip install pandas
```

## 3. Prepare and inspect the metadata first

```powershell
py .\project2_download_amr_data.py
```

This creates a `project2_data` folder with:
- `metadata_7_groups.csv` — metadata for the selected groups
- `phenotypes_long_all_statuses.csv` — one row per isolate/antibiotic result, including intermediate/other status codes
- `phenotypes_binary_SR.csv` — only clear susceptible (S) and resistant (R) labels
- `label_counts_by_group_antibiotic.csv` — label counts to help us check class balance
- `assemblies_to_download.txt` — unique assembly accessions associated with a clear S/R result

**Review the printed counts before downloading genome sequences.** The current CSV has around 29,000 rows, and the number of unique assemblies can still mean a large download.

## 4. Install NCBI Datasets CLI (only if you want genome FASTA files)

Follow NCBI's official installation instructions:
https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/

After installation, open a new PowerShell window and verify:

```powershell
datasets --version
```

## 5. Download the genome assemblies

```powershell
py .\project2_download_amr_data.py --download-genomes
```

The script downloads assemblies in batches of 100 and stores FASTA files under `project2_data\genomes`. It skips batches already marked as extracted, so rerunning it can resume after interruption.

If you want smaller batches:

```powershell
py .\project2_download_amr_data.py --download-genomes --batch-size 25
```

## Important scientific notes

- The metadata CSV may contain malformed lines. Pandas will warn and may skip malformed lines; the script reports the number of rows it actually reads.
- NCBI AST codes `S` and `R` are converted to binary labels (susceptible=0, resistant=1). `I`, `NS`, `SSD`, and undefined codes are preserved in the all-status output but excluded from the binary dataset. These categories should be reviewed before final modelling.
- Each isolate may have results for several antibiotics, so the long phenotype table has multiple rows per genome. It is not valid to randomly split these rows without grouping by genome/sample; otherwise the same genome can leak across train and test.
- The script downloads genome assemblies linked to records with clear S/R labels, not every assembly in NCBI. This is intentional for the first labelled modelling dataset.
- NCBI Datasets may not return every requested assembly, for example if an assembly is no longer available. The script prints a final reminder to compare requested accessions with downloaded files.
- This is for research/education, not clinical decision-making.

## Official NCBI references

- Pathogen Detection Isolates Browser: https://www.ncbi.nlm.nih.gov/pathogens/isolates/
- NCBI Datasets installation: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/
- NCBI Pathogen Detection BigQuery documentation: https://www.ncbi.nlm.nih.gov/pathogens/docs/isolates_gcp/
