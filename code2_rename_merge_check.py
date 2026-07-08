#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code 2: Rename reads, merge FASTA across studies, length-filter, and cross-check metadata.

What it does
------------
Given a directory like:

  ./code1_out/
      study1/
          SRR123_final.fasta
          ERR999_final.fasta
      study2/
          SRR555_final.fasta
          ...

This script will:

1) For each immediate subfolder under --root (e.g., study1, study2, ...):
   - For each *.fasta file in that folder:
       Rename FASTA record headers IN-PLACE to:
           ><RUN>.<i>
       where <RUN> is taken from the filename prefix before the first '_'.
       Example: SRR123_final.fasta -> headers become >SRR123.1, >SRR123.2, ...

2) Merge all renamed FASTA records from all subfolders into ONE merged FASTA in --root
   (default: all_samples.fasta).

3) Length-filter the merged FASTA to keep reads with length in [--min-len, --max-len]
   (default: 220..255), producing:
       all_samples_len<min>_<max>.fasta

4) Compare run IDs between FASTA files and metadata:
   - Write runs present as FASTA files but missing from metadata Run column
   - Write runs in metadata Run column that do not have a corresponding FASTA file

Outputs (in --root)
-------------------
- all_samples.fasta
- all_samples_len<min>_<max>.fasta
- runs_in_fastas_not_in_metadata.txt
- runs_in_metadata_not_in_fastas.txt

Notes
-----
- This script edits FASTA files in-place (safe atomic replace).
- It assumes each per-run file is named like: <RUN>_anything.fasta
  If no '_' exists, it uses the stem before ".fasta".
- It does NOT require Biopython.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def parse_args():
    ap = argparse.ArgumentParser(description="Rename per-run FASTA headers, merge, length-filter, and check metadata.")
    ap.add_argument("--root", default="./code1_out", help="Root directory that contains study subfolders.")
    ap.add_argument("--metadata", required=True, help="Metadata CSV file with a 'Run' column.")
    ap.add_argument("--run-col", default="Run", help="Column name in metadata that contains run IDs (default: Run).")

    ap.add_argument("--min-len", type=int, default=220, help="Minimum sequence length to keep in merged filtered FASTA.")
    ap.add_argument("--max-len", type=int, default=255, help="Maximum sequence length to keep in merged filtered FASTA.")

    ap.add_argument("--merged-name", default="all_samples.fasta", help="Filename for merged FASTA (written under root).")
    ap.add_argument("--filtered-name", default=None,
                    help="Filename for length-filtered merged FASTA (default auto: all_samples_len<min>_<max>.fasta).")

    ap.add_argument("--studies-glob", default="*", help="Which subfolders to treat as studies (default: '*').")
    ap.add_argument("--fasta-glob", default="*.fasta", help="Which FASTA files to process in each study (default: *.fasta).")

    ap.add_argument("--dry-run", action="store_true", help="Do not modify files; just report what would happen.")
    ap.add_argument("--skip-rename", action="store_true", help="Skip renaming step; just merge/filter/check.")
    ap.add_argument("--overwrite-merged", action="store_true", help="Overwrite merged/filtered outputs if they exist.")
    return ap.parse_args()


def run_id_from_filename(fa_path: Path) -> str:
    """Extract run prefix from filename: everything before first '_' else stem."""
    name = fa_path.name
    stem = name[:-len(".fasta")] if name.lower().endswith(".fasta") else fa_path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def iter_fasta_records(path: Path) -> Iterable[Tuple[str, str]]:
    """
    Stream FASTA records as (header, sequence).
    Header returned WITHOUT the leading '>'.
    """
    header = None
    seq_parts = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)


def write_fasta_records(records: Iterable[Tuple[str, str]], out_path: Path) -> int:
    """Write FASTA records (header without '>') and return count written."""
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for h, s in records:
            out.write(f">{h}\n{s}\n")
            n += 1
    return n


def rename_fasta_headers_inplace(fa_path: Path, run_id: str, dry_run: bool = False) -> int:
    """
    Replace each header in fa_path with >{run_id}.{i} where i starts at 1 within file.
    Uses atomic replace via temp file.
    Returns number of records processed.
    """
    tmp = fa_path.with_suffix(fa_path.suffix + ".tmp")
    i = 0

    if dry_run:
        # just count
        for _h, _s in iter_fasta_records(fa_path):
            i += 1
        return i

    with tmp.open("w", encoding="utf-8") as out:
        for _h, s in iter_fasta_records(fa_path):
            i += 1
            out.write(f">{run_id}.{i}\n{s}\n")

    tmp.replace(fa_path)
    return i


def collect_study_folders(root: Path, studies_glob: str) -> List[Path]:
    folders = [p for p in root.glob(studies_glob) if p.is_dir()]
    folders.sort()
    return folders


def collect_fasta_files(study_dir: Path, fasta_glob: str) -> List[Path]:
    files = [p for p in study_dir.glob(fasta_glob) if p.is_file()]
    files.sort()
    return files


def merge_fastas(fasta_paths: List[Path], merged_path: Path, overwrite: bool = False) -> int:
    if merged_path.exists() and not overwrite:
        raise FileExistsError(f"Merged file exists (use --overwrite-merged): {merged_path}")
    n = 0
    with merged_path.open("w", encoding="utf-8") as out:
        for fa in fasta_paths:
            with fa.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    out.write(line)
            n += 1
    return n


def length_filter_fasta(in_fa: Path, out_fa: Path, min_len: int, max_len: int, overwrite: bool = False) -> Tuple[int, int]:
    """
    Filter records by length inclusive. Returns (kept, total).
    """
    if out_fa.exists() and not overwrite:
        raise FileExistsError(f"Filtered file exists (use --overwrite-merged): {out_fa}")
    kept = 0
    total = 0
    with out_fa.open("w", encoding="utf-8") as out:
        for h, s in iter_fasta_records(in_fa):
            total += 1
            L = len(s)
            if min_len <= L <= max_len:
                out.write(f">{h}\n{s}\n")
                kept += 1
    return kept, total


def read_metadata_runs(meta_csv: Path, run_col: str) -> Set[str]:
    """
    Read Run IDs from metadata CSV (robust to commas/quotes using csv module).
    """
    runs: Set[str] = set()
    with meta_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file has no header: {meta_csv}")
        if run_col not in reader.fieldnames:
            raise ValueError(f"Metadata missing column '{run_col}'. Columns: {reader.fieldnames}")
        for row in reader:
            v = (row.get(run_col) or "").strip().strip('"')
            if v:
                runs.add(v)
    return runs


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    meta = Path(args.metadata).resolve()

    if not root.exists():
        sys.exit(f"Root folder does not exist: {root}")
    if not meta.exists():
        sys.exit(f"Metadata file does not exist: {meta}")

    filtered_name = args.filtered_name or f"all_samples_len{args.min_len}_{args.max_len}.fasta"
    merged_path = root / args.merged_name
    filtered_path = root / filtered_name

    study_dirs = collect_study_folders(root, args.studies_glob)
    if not study_dirs:
        sys.exit(f"No study subfolders found under {root} matching glob: {args.studies_glob}")

    all_fastas: List[Path] = []
    run_ids_from_files: Set[str] = set()

    log(f"[root] {root}")
    log(f"[meta] {meta}")
    log(f"[studies] found {len(study_dirs)} folder(s)")

    # 1) Rename in-place
    if not args.skip_rename:
        for sd in study_dirs:
            fas = collect_fasta_files(sd, args.fasta_glob)
            if not fas:
                continue
            log(f"\n[study] {sd.name}: {len(fas)} FASTA files")
            for fa in fas:
                run_id = run_id_from_filename(fa)
                run_ids_from_files.add(run_id)
                all_fastas.append(fa)

                nrec = rename_fasta_headers_inplace(fa, run_id, dry_run=args.dry_run)
                if args.dry_run:
                    log(f"  [DRY] would rename {fa.name} -> headers >{run_id}.1..{run_id}.{nrec}")
                else:
                    log(f"  renamed {fa.name}: {nrec} records -> >{run_id}.<i>")
    else:
        # just collect FASTAs and run IDs
        for sd in study_dirs:
            fas = collect_fasta_files(sd, args.fasta_glob)
            for fa in fas:
                run_id = run_id_from_filename(fa)
                run_ids_from_files.add(run_id)
                all_fastas.append(fa)

    if not all_fastas:
        sys.exit("No FASTA files found. Check --root / --studies-glob / --fasta-glob.")

    # 2) Merge to one file
    log(f"\n[merge] {len(all_fastas)} FASTA files -> {merged_path.name}")
    if args.dry_run:
        log("  [DRY] would create merged FASTA (skipping write)")
    else:
        if merged_path.exists() and not args.overwrite_merged:
            log(f"  merged already exists (skipping, use --overwrite-merged to rebuild): {merged_path}")
        else:
            merge_fastas(all_fastas, merged_path, overwrite=True)
            log(f"  wrote merged FASTA: {merged_path}")

    # 3) Length-filter merged
    log(f"\n[length-filter] keep {args.min_len}..{args.max_len} bp -> {filtered_path.name}")
    if args.dry_run:
        log("  [DRY] would length-filter merged FASTA (skipping write)")
    else:
        if not merged_path.exists() or merged_path.stat().st_size == 0:
            sys.exit(f"Merged FASTA missing or empty: {merged_path}")
        if filtered_path.exists() and not args.overwrite_merged:
            log(f"  filtered already exists (skipping, use --overwrite-merged to rebuild): {filtered_path}")
        else:
            kept, total = length_filter_fasta(merged_path, filtered_path, args.min_len, args.max_len, overwrite=True)
            log(f"  wrote filtered FASTA: {filtered_path}  (kept {kept}/{total} reads)")

    # 4) Cross-check with metadata
    log("\n[check] comparing FASTA run IDs vs metadata Run column")
    meta_runs = read_metadata_runs(meta, args.run_col)

    in_fastas_not_in_meta = sorted(run_ids_from_files - meta_runs)
    in_meta_not_in_fastas = sorted(meta_runs - run_ids_from_files)

    out1 = root / "runs_in_fastas_not_in_metadata.txt"
    out2 = root / "runs_in_metadata_not_in_fastas.txt"

    if args.dry_run:
        log(f"  [DRY] would write {out1.name} ({len(in_fastas_not_in_meta)} runs)")
        log(f"  [DRY] would write {out2.name} ({len(in_meta_not_in_fastas)} runs)")
    else:
        out1.write_text("\n".join(in_fastas_not_in_meta) + ("\n" if in_fastas_not_in_meta else ""), encoding="utf-8")
        out2.write_text("\n".join(in_meta_not_in_fastas) + ("\n" if in_meta_not_in_fastas else ""), encoding="utf-8")
        log(f"  wrote {out1} ({len(in_fastas_not_in_meta)} runs)")
        log(f"  wrote {out2} ({len(in_meta_not_in_fastas)} runs)")

    log("\n[done]")


if __name__ == "__main__":
    main()
