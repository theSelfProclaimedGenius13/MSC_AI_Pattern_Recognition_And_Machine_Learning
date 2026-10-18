#!/usr/bin/env python3
"""
Prepare NCBI Pathogen Detection metadata for Project 2 and optionally download
the matching genome assemblies.

Input:
    isolates.csv  (the CSV exported by the user from NCBI Pathogen Detection)

Scope is deliberately limited to the seven organism groups in that CSV:
- Salmonella enterica
- E.coli and Shigella
- Campylobacter jejuni
- Klebsiella pneumoniae
- Staphylococcus aureus
- Mycobacterium tuberculosis
- Acinetobacter baumannii

Default mode prepares clean metadata and manifests only.
Use --download-genomes to download all unique assemblies with at least one
clear S/R AST label. Downloads are made in batches using the NCBI Datasets CLI.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ALLOWED_GROUPS = {
    "Salmonella enterica",
    "E.coli and Shigella",
    "Campylobacter jejuni",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus",
    "Mycobacterium tuberculosis",
    "Acinetobacter baumannii",
}

STATUS_MAP = {
    "R": "resistant",
    "S": "susceptible",
}
# I (intermediate), NS (nonsusceptible), SSD (susceptible-dose dependent),
# and undefined labels are retained in the raw metadata but not binarized.
AST_PATTERN = re.compile(r"^\s*(.*?)\s*=\s*(R|S|I|NS|SSD|N|ND)\s*$", re.I)


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Columns found: {list(df.columns)}"
    )


def parse_ast(value):
    """Parse strings such as 'ampicillin=R,tetracycline=S' into pairs."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    pairs = []
    # NCBI browser exports commonly store the AST field as comma-separated
    # antibiotic=status pairs. Split on commas and parse only recognized pairs.
    for item in text.split(","):
        match = AST_PATTERN.match(item)
        if match:
            antibiotic = match.group(1).strip()
            status = match.group(2).upper()
            if antibiotic:
                pairs.append((antibiotic, status))
    return pairs


def run_prepare(input_csv: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Reading: {input_csv}")
    print("Note: if pandas reports malformed CSV lines, those lines may be skipped.")
    df = pd.read_csv(input_csv, low_memory=False, on_bad_lines="warn", encoding="utf-8-sig")

    group_col = find_col(df, "#Organism group", "Organism group")
    ast_col = find_col(df, "AST phenotypes", "AST_phenotypes")
    assembly_col = find_col(df, "Assembly", "asm_acc", "Assembly accession")
    biosample_col = find_col(df, "BioSample", "biosample_acc")
    strain_col = next((c for c in df.columns if str(c).strip().lower() == "strain"), None)

    df[group_col] = df[group_col].astype("string").str.strip()
    original_rows = len(df)
    df = df[df[group_col].isin(ALLOWED_GROUPS)].copy()

    # Standardize key columns without changing the original data.
    df["organism_group"] = df[group_col].astype("string")
    df["biosample_accession"] = df[biosample_col].astype("string").str.strip()
    df["assembly_accession"] = df[assembly_col].astype("string").str.strip()
    df.loc[df["biosample_accession"].isin(["", "nan", "None", "<NA>"]), "biosample_accession"] = pd.NA
    df.loc[df["assembly_accession"].isin(["", "nan", "None", "<NA>"]), "assembly_accession"] = pd.NA

    df.to_csv(outdir / "metadata_7_groups.csv", index=False)

    long_rows = []
    for _, row in df.iterrows():
        pairs = parse_ast(row.get(ast_col))
        for antibiotic, status in pairs:
            item = {
                "organism_group": row["organism_group"],
                "biosample_accession": row["biosample_accession"],
                "assembly_accession": row["assembly_accession"],
                "antibiotic": antibiotic,
                "ast_status_original": status,
                "label": STATUS_MAP.get(status),  # only S/R become binary labels
                "label_binary": {"S": 0, "R": 1}.get(status),
            }
            if strain_col:
                item["strain"] = row.get(strain_col)
            long_rows.append(item)

    phenotypes = pd.DataFrame(long_rows)
    if phenotypes.empty:
        raise RuntimeError(
            "No AST pairs were parsed. Please check the exact format of the "
            "'AST phenotypes' column in isolates.csv."
        )

    phenotypes.to_csv(outdir / "phenotypes_long_all_statuses.csv", index=False)
    binary = phenotypes[phenotypes["label_binary"].notna()].copy()
    binary.to_csv(outdir / "phenotypes_binary_SR.csv", index=False)

    usable = binary.dropna(subset=["assembly_accession"]).copy()
    accessions = sorted(set(usable["assembly_accession"].astype(str)))
    (outdir / "assemblies_to_download.txt").write_text(
        "\n".join(accessions) + ("\n" if accessions else ""), encoding="utf-8"
    )

    summary = (
        binary.groupby(["organism_group", "antibiotic", "ast_status_original"])
        .size().rename("records").reset_index()
    )
    summary.to_csv(outdir / "label_counts_by_group_antibiotic.csv", index=False)

    print("\nPREPARATION COMPLETE")
    print(f"Rows read from CSV: {original_rows:,}")
    print(f"Rows in the selected seven groups: {len(df):,}")
    print(f"Parsed isolate-antibiotic AST records (all statuses): {len(phenotypes):,}")
    print(f"Binary S/R isolate-antibiotic records: {len(binary):,}")
    print(f"Binary S/R records with assembly accessions: {len(usable):,}")
    print(f"Unique assemblies to download: {len(accessions):,}")
    print(f"Output folder: {outdir.resolve()}")
    print("\nCreated:")
    for name in [
        "metadata_7_groups.csv",
        "phenotypes_long_all_statuses.csv",
        "phenotypes_binary_SR.csv",
        "label_counts_by_group_antibiotic.csv",
        "assemblies_to_download.txt",
    ]:
        print(f"  - {name}")

    print("\nOrganism groups retained:")
    for group, count in df["organism_group"].value_counts().items():
        print(f"  {group}: {count:,} metadata rows")

    print("\nImportant: the same genome can have several antibiotic labels. "
          "Each row in phenotypes_binary_SR.csv is one genome/sample–antibiotic "
          "result, not one independent genome.")


def run_download(outdir: Path, batch_size: int):
    datasets_cli = shutil.which("datasets")
    if not datasets_cli:
        print(
            "\nERROR: NCBI Datasets CLI ('datasets') was not found.\n"
            "Install it using NCBI's official instructions:\n"
            "https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/\n"
            "Then open a NEW PowerShell window and run this script again with "
            "--download-genomes.",
            file=sys.stderr,
        )
        sys.exit(2)

    accession_file = outdir / "assemblies_to_download.txt"
    if not accession_file.exists():
        raise FileNotFoundError(
            f"{accession_file} is missing. Run preparation mode first."
        )
    accessions = [
        line.strip() for line in accession_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not accessions:
        print("No assembly accessions found; nothing to download.")
        return

    genome_dir = outdir / "genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = outdir / "download_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    total = (len(accessions) + batch_size - 1) // batch_size
    print(f"\nDownloading {len(accessions):,} unique assemblies in {total} batches.")
    print("This may require many GB of disk space and may take a long time.")
    print(f"Batch size: {batch_size}. Files will be saved under: {genome_dir.resolve()}")

    for batch_num, start in enumerate(range(0, len(accessions), batch_size), start=1):
        batch = accessions[start:start + batch_size]
        batch_dir = batches_dir / f"batch_{batch_num:05d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_file = batch_dir / "accessions.txt"
        zip_file = batch_dir / "genomes.zip"
        input_file.write_text("\n".join(batch) + "\n", encoding="utf-8")

        # Skip completed batches to allow resuming after interruption.
        marker = batch_dir / "EXTRACTED_OK.txt"
        if marker.exists():
            print(f"[{batch_num}/{total}] already extracted; skipping")
            continue

        cmd = [
            datasets_cli, "download", "genome", "accession",
            "--inputfile", str(input_file),
            "--include", "genome",
            "--filename", str(zip_file),
        ]
        print(f"[{batch_num}/{total}] requesting {len(batch)} assemblies...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Batch {batch_num} failed. You can rerun the same command; "
                "already completed batches will be skipped.",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        try:
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(batch_dir / "extracted")
        except zipfile.BadZipFile:
            print(f"Downloaded archive for batch {batch_num} is not a valid ZIP.", file=sys.stderr)
            sys.exit(3)

        # Copy genome FASTA files into one searchable directory.
        extracted_root = batch_dir / "extracted"
        copied = 0
        for fasta in extracted_root.rglob("*.fna"):
            target = genome_dir / fasta.name
            if not target.exists():
                shutil.copy2(fasta, target)
                copied += 1
        marker.write_text(f"Extracted; copied {copied} FASTA files.\n", encoding="utf-8")
        print(f"  extracted; copied {copied} FASTA files")

    print("\nGENOME DOWNLOAD PROCESS FINISHED")
    print(f"Genome FASTA directory: {genome_dir.resolve()}")
    print("Check the NCBI package reports and compare downloaded assemblies with "
          "assemblies_to_download.txt; not every requested accession is guaranteed "
          "to have an available genome package.")


def main():
    parser = argparse.ArgumentParser(description="Prepare/download AMR genomes for seven NCBI organism groups.")
    parser.add_argument("--input", default="isolates.csv", help="Path to your NCBI export CSV (default: isolates.csv)")
    parser.add_argument("--outdir", default="project2_data", help="Output folder (default: project2_data)")
    parser.add_argument("--download-genomes", action="store_true", help="Also download genome FASTA files using NCBI Datasets CLI")
    parser.add_argument("--batch-size", type=int, default=100, help="Assemblies per download batch (default: 100)")
    args = parser.parse_args()

    input_csv = Path(args.input)
    outdir = Path(args.outdir)
    if not input_csv.exists():
        print(
            f"Input file not found: {input_csv.resolve()}\n"
            "Put isolates.csv beside this script or pass --input with the full path.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.batch_size < 1 or args.batch_size > 1000:
        parser.error("--batch-size must be between 1 and 1000")

    run_prepare(input_csv, outdir)
    if args.download_genomes:
        run_download(outdir, args.batch_size)


if __name__ == "__main__":
    main()
