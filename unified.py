#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified.py

Unified HPC-friendly pipeline for amplicon microbiome processing + downstream matrices and analyses.

WHAT IT DOES
------------
A) ASV + taxonomy + base matrices (optional)
   - Filters reads by length from all_samples.fasta
   - Dereplicates, UNOISE clusters to ASVs (vsearch)
   - Maps all reads to ASVs (vsearch usearch_global)
   - BLAST ASVs against SILVA (blastn)
   - Filters ASVs flagged as chloroplast/mitochondria/Eukaryota
   - Optionally drops samples matching exclusion conditions from metadata
   - Computes, for each rank: ASV, kingdom, phylum, class, order, family, genus:
       rel_abundance_<rank>.csv
       bc_matrix_<rank>.csv
       pcoa_coordinates_<rank>.csv  (up to 10 PCs)
       eigenvalues_<rank>.csv
   - Enriches metadata with "most abundant <rank>" columns (optional)

B) Subfolder creation (optional)
   - Creates niche-based subsets under each saved_matrices_<rank>/<subset_name>/
   - Creates species-based subsets (plant_rice / plant_potato / plant_other) based on --species-col

C) Postprocess scan (optional)
   - Recursively scans all saved_matrices_* folders (including subfolders) and creates:
       shannon_<rank>.csv
       anosim_pairwise_<rank>_<factor>.csv (+ optional PNG)
       correlation outputs for a chosen rank only:
          rel_abundance_<rank>_<mode>_matrix.csv
          corr_<rank>_<method>_<mode>.csv
          corr_<rank>_<method>_<mode>.png

RECALCULATION POLICY (set in SLURM with --recalc)
-------------------------------------------------
--recalc missing   : only compute outputs that are missing / empty / tiny
--recalc all       : recompute everything possible
--recalc selected  : recompute targets listed in --redo-list, and ALSO compute missing outputs

Targets are internal IDs (human-readable). You can generate a redo list template:
    python3 unified.py --root . --make-redo-list

DEPENDENCIES
------------
External:
  - vsearch
  - blastn

Python:
  - biopython, pandas, numpy, scipy
  - matplotlib (optional; for PNG plots)
  - scikit-bio (optional; for ANOSIM)

Notes:
- Placeholder files are created for subsets with <2 samples, but postprocess will SKIP tiny/empty files.
- CSV parsing with commas in quoted fields can confuse simple `cut -d,` checks; use pandas/awk safeguards.
"""

import argparse
import gc
import os
import random
import re
import sys
import subprocess as sp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# Optional libs
HAVE_MPL = False
try:
    import matplotlib.pyplot as plt  # noqa: F401
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

HAVE_SKBIO = False
try:
    from skbio import DistanceMatrix  # noqa: F401
    from skbio.stats.distance import anosim  # noqa: F401
    HAVE_SKBIO = True
except Exception:
    HAVE_SKBIO = False

from Bio import SeqIO
from scipy.spatial.distance import pdist, squareform


# =============================
# Logging (tee stdout/stderr)
# =============================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def log(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# =============================
# Recalc policy
# =============================

def is_tiny(path: Path, min_bytes: int = 16) -> bool:
    try:
        return (not path.exists()) or path.stat().st_size < min_bytes
    except Exception:
        return True


def is_nonempty(path: Path, min_bytes: int = 16) -> bool:
    return (path.exists() and path.is_file() and (path.stat().st_size >= min_bytes))


def load_selected_targets(redo_list: Path) -> Set[str]:
    """
    redo_list format: one target ID per line (comments allowed with #).
    """
    selected: Set[str] = set()
    if not redo_list.exists():
        return selected
    for line in redo_list.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        selected.add(line)
    return selected


def policy_should_recompute(target_id: str, out_path: Path, policy: str, selected: Set[str]) -> bool:
    """
    Returns True if we should (re)compute target_id.
    policy:
      - all: always
      - missing: only if out missing/tiny
      - selected: if target in selected OR out missing/tiny
    """
    if policy == "all":
        return True
    if policy == "missing":
        return is_tiny(out_path)
    if policy == "selected":
        return (target_id in selected) or is_tiny(out_path)
    return is_tiny(out_path)


# =============================
# CLI
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="Unified ASV + taxonomy + BC/PCoA + subsets + postprocess (idempotent)")

    ap.add_argument("--root", default=".", help="Root working directory")
    ap.add_argument("--log-file", default=None, help="Mirror stdout/stderr to a log file (default: unified.log)")

    ap.add_argument("--recalc", choices=["missing", "all", "selected"], default="missing",
                    help="Recompute policy: missing | all | selected")
    ap.add_argument("--redo-list", default="redo_list.txt", help="Path to redo_list.txt (for --recalc selected)")
    ap.add_argument("--make-redo-list", action="store_true", help="Write a redo_list template and exit")

    ap.add_argument("--run-asv", action="store_true", help="Run ASV+taxonomy+base ranks")
    ap.add_argument("--make-niche-subsets", action="store_true", help="Create niche-based subset folders")
    ap.add_argument("--niche-col", default="niche_category", help="Metadata column used for niche subsets")
    ap.add_argument("--make-species-subsets", action="store_true", help="Create species-based subset folders")
    ap.add_argument("--species-col", default=None,
                    help="Metadata column used for species subsets; values like rice/potato/other (supports rice[01] etc.)")

    ap.add_argument("--postprocess", action="store_true", help="Scan folders and compute Shannon/ANOSIM/correlations")
    ap.add_argument("--do-shannon", action="store_true", help="Postprocess: compute Shannon")
    ap.add_argument("--do-anosim", action="store_true", help="Postprocess: compute ANOSIM")
    ap.add_argument("--do-corr", action="store_true", help="Postprocess: compute correlations")

    ap.add_argument("--all-fasta", default=None, help="Input reads FASTA (multi-sample combined)")
    ap.add_argument("--silva-db", default=None, help="BLAST DB basename path (no extensions)")
    ap.add_argument("--metadata", default="metadata.csv", help="Metadata CSV with 'Run' column")
    ap.add_argument("--metadata-out", default=None,
                    help="Output metadata filename (default: overwrite --metadata with enriched columns)")
    ap.add_argument("--threads", type=int, default=8, help="Threads for BLAST")
    ap.add_argument("--min-len", type=int, default=220)
    ap.add_argument("--max-len", type=int, default=254)
    ap.add_argument("--unoise-minsize", type=int, default=10)
    ap.add_argument("--unoise-alpha", type=float, default=2.0)
    ap.add_argument("--identity", type=float, default=0.97, help="vsearch --id for usearch_global")
    ap.add_argument("--postfilter-min-sample-sum", type=int, default=200,
                    help="Minimum total reads per sample after ASV filtering")

    ap.add_argument("--exclude", default="", help='Comma-separated exclusions like "source_folder=AppleStudy,project=Foo"')

    ap.add_argument("--anosim-factor", default="niche_category", help="Metadata grouping column for ANOSIM")
    ap.add_argument("--anosim-merge", default="",
                    help='Merge rules e.g. "root=endosphere|rhizoplane;other=aerial root|flower|fruit|seed|stem"')
    ap.add_argument("--anosim-pairs", choices=["allpairs", "other_plus_rest"], default="allpairs",
                    help="Pairwise strategy for ANOSIM")
    ap.add_argument("--anosim-perms", type=int, default=500)

    ap.add_argument("--corr-rank", default="phylum", help="Only run correlation for this rank")
    ap.add_argument("--corr-method", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--corr-mode", choices=["topn", "fixed"], default="topn")
    ap.add_argument("--corr-topn", type=int, default=10)
    ap.add_argument("--corr-taxa", default=None, help="Comma-separated taxa list for fixed mode")
    ap.add_argument("--corr-taxa-file", default=None, help="Text file (one taxon per line) for fixed mode")

    args = ap.parse_args()

    if args.postprocess and not (args.do_shannon or args.do_anosim or args.do_corr):
        args.do_shannon = True
        args.do_anosim = True
        args.do_corr = True

    return args


# =============================
# Helpers
# =============================

def run(cmd: List[str]):
    log("[RUN] " + " ".join(cmd))
    sp.run(cmd, check=True)


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if is_tiny(path):
        raise ValueError(f"tiny/empty file: {path}")
    return pd.read_csv(path, **kwargs)


# =============================
# ASV pipeline helpers
# =============================

TAX_COLS = ["ASV", "kingdom", "phylum", "class", "order", "family", "genus", "species/strain"]
MITO_CHLORO_PAT = re.compile(r"chloroplast|mitochondria|eukaryota", re.IGNORECASE)


def filter_fasta_length(in_fa: Path, out_fa: Path, min_len: int, max_len: int) -> int:
    kept = 0
    with open(out_fa, "w") as out:
        for rec in SeqIO.parse(str(in_fa), "fasta"):
            L = len(rec.seq)
            if min_len <= L <= max_len:
                SeqIO.write(rec, out, "fasta")
                kept += 1
    log(f"[fasta] kept {kept} reads in [{min_len},{max_len}] -> {out_fa}")
    return kept


def load_asv_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0)
    first = df.columns[0]
    if first.lower() in ["#otu id", "otu id", "asv", "asv_id"]:
        df.rename(columns={first: "ASV"}, inplace=True)
    df.set_index("ASV", inplace=True)
    return df


def parse_blast_taxonomy(blast_path: Path, out_tsv: Path) -> pd.DataFrame:
    if is_nonempty(out_tsv):
        return pd.read_csv(out_tsv, sep="\t")

    rows = []
    with open(blast_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            asv = parts[0].split(";", 1)[0]
            sseqid = parts[1]
            tax_str = sseqid.split("_", 1)[1] if "_" in sseqid else sseqid
            taxa = [t.strip() for t in tax_str.split(";") if t.strip()]
            taxa = (taxa[:7] + [None] * 7)[:7]
            if taxa[0] and "_" in str(taxa[0]):
                taxa[0] = str(taxa[0]).split("_", 1)[1]
            rows.append([asv] + taxa)

    df = pd.DataFrame(rows, columns=TAX_COLS)
    df.to_csv(out_tsv, sep="\t", index=False)
    return df


def asvs_to_exclude(taxa_df: pd.DataFrame, blast_path: Optional[Path] = None) -> Set[str]:
    bad: Set[str] = set()
    for _, row in taxa_df.iterrows():
        for c in TAX_COLS[1:]:
            v = row.get(c, None)
            if isinstance(v, str) and MITO_CHLORO_PAT.search(v):
                bad.add(row["ASV"])
                break
    if blast_path and blast_path.exists():
        with open(blast_path, "r") as f:
            for line in f:
                if MITO_CHLORO_PAT.search(line):
                    bad.add(line.split("\t", 1)[0].split(";", 1)[0])
    log(f"[filter] flagged {len(bad)} ASVs as chloroplast/mitochondria/Eukaryota")
    return bad


def filter_asv_table(asv_table: pd.DataFrame, bad_asvs: Set[str], min_sample_sum: int, out_path: Path) -> pd.DataFrame:
    before_rows = asv_table.shape[0]
    present_bad = sorted(list(set(asv_table.index).intersection(bad_asvs)))
    asv_table_f = asv_table.drop(index=present_bad)
    after_rows = asv_table_f.shape[0]
    log(f"[filter] ASV rows: {before_rows} -> {after_rows} (removed {before_rows - after_rows})")

    col_sums = asv_table_f.sum(axis=0)
    keep_cols = col_sums[col_sums >= min_sample_sum].index.tolist()
    drop_cols = [c for c in asv_table_f.columns if c not in keep_cols]
    if drop_cols:
        (out_path.parent / "dropped_samples_lowdepth.txt").write_text("\n".join(drop_cols) + "\n")
        log(f"[filter] Dropping {len(drop_cols)} low-depth samples (<{min_sample_sum}).")
    asv_table_f2 = asv_table_f[keep_cols]
    asv_table_f2.to_csv(out_path, sep="\t")
    return asv_table_f2


def parse_exclusions(exclude_str: str) -> List[Tuple[str, str]]:
    out = []
    s = (exclude_str or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad --exclude entry (expected col=value): {p}")
        col, val = p.split("=", 1)
        out.append((col.strip(), val.strip()))
    return out


def runs_to_exclude_from_metadata(metadata_csv: Path, rules: List[Tuple[str, str]]) -> Set[str]:
    if not rules:
        return set()
    md = pd.read_csv(metadata_csv)
    if "Run" not in md.columns:
        raise ValueError("metadata CSV must contain a 'Run' column")
    md["Run"] = md["Run"].astype(str).str.strip()

    mask = pd.Series(True, index=md.index)
    for col, val in rules:
        if col not in md.columns:
            raise ValueError(f"metadata missing column for exclusion: {col}")
        s = md[col].astype(str).str.strip().str.lower()
        mask = mask & (s == str(val).strip().lower())

    runs = md.loc[mask, "Run"].astype(str).str.strip()
    runs = runs[runs != ""]
    return set(runs.tolist())


def drop_samples_from_asv_table(asv_table: pd.DataFrame, runs_to_drop: Set[str], report_path: Path) -> pd.DataFrame:
    if not runs_to_drop:
        return asv_table
    cols_present = [c for c in asv_table.columns if c in runs_to_drop]
    report_path.write_text("\n".join(sorted(cols_present)) + "\n")
    log(f"[exclude] dropping {len(cols_present)} samples from ASV table due to exclusions")
    return asv_table.drop(columns=cols_present)


def aggregate_by_rank(asv_table: pd.DataFrame, asv_tax: pd.DataFrame, rank: str) -> pd.DataFrame:
    merged = pd.merge(asv_table.reset_index(), asv_tax, on="ASV", how="left")
    sample_cols = [c for c in merged.columns if c not in TAX_COLS]
    grouped = merged.groupby(rank, dropna=False)[sample_cols].sum()
    rel = grouped.div(grouped.sum(axis=0), axis=1).fillna(0)
    return rel


def compute_bc_matrix(rel_abundance_df: pd.DataFrame) -> pd.DataFrame:
    data = rel_abundance_df.T.values.astype(np.float32)
    dist_vec = pdist(data, metric="braycurtis").astype(np.float32)
    del data
    gc.collect()
    dist_mat = squareform(dist_vec).astype(np.float32)
    del dist_vec
    gc.collect()
    dist_mat = dist_mat.astype(np.float16)
    return pd.DataFrame(dist_mat, index=rel_abundance_df.columns, columns=rel_abundance_df.columns, dtype=np.float16)


def pcoa(distance_df: pd.DataFrame, n_components=10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    D = distance_df.values.astype(np.float64)
    n = D.shape[0]
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(columns=["Eigenvalue"])
    if n == 1:
        sid = distance_df.index[0]
        coords_df = pd.DataFrame([[0.0]], index=[sid], columns=["PC1"])
        eigvals_df = pd.DataFrame([0.0], index=["EV1"], columns=["Eigenvalue"])
        return coords_df, eigvals_df

    k = min(int(n_components), int(n))

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J.dot(D**2).dot(J)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    top_eigvals = eigvals[:k]
    clipped = np.clip(top_eigvals, a_min=0, a_max=None)
    coords = eigvecs[:, :k] * np.sqrt(clipped)[np.newaxis, :]
    coords_df = pd.DataFrame(coords, index=distance_df.index, columns=[f"PC{i+1}" for i in range(k)])
    eigvals_df = pd.DataFrame(eigvals, index=[f"EV{i+1}" for i in range(len(eigvals))], columns=["Eigenvalue"])
    return coords_df, eigvals_df


def compute_most_abundant(asv_table_filt: pd.DataFrame, asv_tax: pd.DataFrame,
                          metadata_in: Path, metadata_out: Path,
                          policy: str, selected: Set[str]) -> None:
    target_id = f"METADATA:enrich:{metadata_out}"
    if not policy_should_recompute(target_id, metadata_out, policy, selected):
        log(f"[metadata] skip (exists): {metadata_out}")
        return

    dfA = asv_table_filt.reset_index().merge(asv_tax, on="ASV", how="left").set_index("ASV")
    sample_cols = [c for c in dfA.columns if c not in TAX_COLS]
    tax_ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species/strain"]

    abundant_rows = []
    for sample in sample_cols:
        winners = {}
        for rank in tax_ranks:
            grouped = dfA.groupby(rank, dropna=False)[sample].sum().reset_index()
            grouped.sort_values(by=sample, ascending=False, inplace=True)
            winners[rank] = grouped.iloc[0][rank] if len(grouped) and grouped.iloc[0][sample] > 0 else None
        best_asv = dfA[sample].idxmax() if dfA[sample].max() > 0 else None
        abundant_rows.append([sample] + [winners[r] for r in tax_ranks] + [best_asv])

    winning_df = pd.DataFrame(
        abundant_rows,
        columns=["Run"] + tax_ranks + ["most_abundant_ASV"]
    ).rename(columns={
        "kingdom": "the most abundant kingdom",
        "phylum": "the most abundant phylum",
        "class": "the most abundant class",
        "order": "the most abundant order",
        "family": "the most abundant family",
        "genus": "the most abundant genus",
        "species/strain": "the most abundant species/strain",
        "most_abundant_ASV": "the most abundant ASV"
    })

    metadata = pd.read_csv(metadata_in)
    if "Run" not in metadata.columns:
        raise ValueError("metadata must contain Run column")
    metadata["Run"] = metadata["Run"].astype(str).str.strip()
    merged_metadata = metadata.merge(winning_df, on="Run", how="left")
    merged_metadata.to_csv(metadata_out, index=False)
    log(f"[metadata] wrote {metadata_out} ({len(merged_metadata)} rows)")


# =============================
# Subset creation
# =============================

def _ensure_placeholder_files(out_dir: Path, rank: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [f"rel_abundance_{rank}.csv", f"bc_matrix_{rank}.csv",
                 f"pcoa_coordinates_{rank}.csv", f"eigenvalues_{rank}.csv"]:
        p = out_dir / name
        if not p.exists():
            p.write_text("")


def subset_and_save_for_rank(rank: str, subset_name: str, sample_ids: List[str],
                             policy: str, selected: Set[str]):
    base = Path(f"saved_matrices_{rank}")
    base_rel_csv = base / f"rel_abundance_{rank}.csv"
    base_bc_csv = base / f"bc_matrix_{rank}.csv"

    out_dir = base / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_rel_csv = out_dir / f"rel_abundance_{rank}.csv"
    sub_bc_csv = out_dir / f"bc_matrix_{rank}.csv"
    sub_coord_csv = out_dir / f"pcoa_coordinates_{rank}.csv"
    sub_eig_csv = out_dir / f"eigenvalues_{rank}.csv"

    target_id = f"SUBSET:{subset_name}:{rank}:{out_dir}"
    if not policy_should_recompute(target_id, sub_coord_csv, policy, selected) and \
       all(is_nonempty(p) for p in [sub_rel_csv, sub_bc_csv, sub_coord_csv, sub_eig_csv]):
        log(f"[subset:{subset_name}] {rank}: skip (exists)")
        return

    if not is_nonempty(base_rel_csv):
        log(f"[subset:{subset_name}] {rank}: missing base rel -> placeholders")
        _ensure_placeholder_files(out_dir, rank)
        return

    rel = pd.read_csv(base_rel_csv, index_col=0)
    have = [s for s in sample_ids if s in rel.columns]
    if len(have) < 2:
        log(f"[subset:{subset_name}] {rank}: <2 samples -> placeholders")
        _ensure_placeholder_files(out_dir, rank)
        return

    rel_sub = rel[have]
    rel_sub.to_csv(sub_rel_csv)

    # subset BC if possible, else compute from rel
    bc_sub = None
    if is_nonempty(base_bc_csv):
        bc_full = pd.read_csv(base_bc_csv, index_col=0)
        present = [s for s in have if s in bc_full.index and s in bc_full.columns]
        if len(present) >= 2:
            bc_sub = bc_full.loc[present, present]
    if bc_sub is None:
        bc_sub = compute_bc_matrix(rel_sub)
    bc_sub.to_csv(sub_bc_csv)

    coords, eig = pcoa(bc_sub, n_components=10)
    coords.to_csv(sub_coord_csv)
    eig.to_csv(sub_eig_csv)
    log(f"[subset:{subset_name}] {rank}: wrote rel/bc/pcoa/eig")


def run_niche_subsets(metadata_csv: Path, ranks: List[str], niche_col: str,
                      policy: str, selected: Set[str]):
    md = pd.read_csv(metadata_csv)
    if "Run" not in md.columns:
        raise ValueError("metadata missing Run column")
    if niche_col not in md.columns:
        raise ValueError(f"metadata missing niche column '{niche_col}'")

    md["Run"] = md["Run"].astype(str).str.strip()
    niche = md.set_index("Run")[niche_col].astype(str).str.strip().str.lower()
    run_ids = niche.index.tolist()

    subsets = {
        "bulk_soil": ["bulk soil", "bulk-soil", "bulk_soil"],
        "endosphere": ["endosphere"],
        "rhizoplane": ["rhizoplane"],
        "rhizosphere": ["rhizosphere"],
        "nodule": ["nodule"],
        "xylem_sap": ["xylem sap"],
        "tuber": ["tuber"],
        "seed": ["seed"],
        "stem": ["stem"],
        "leaf": ["leaf"],
        "endosphere_and_rhizoplane": ["endosphere", "rhizoplane"],
        "endosphere_rhizoplane_rhizosphere": ["endosphere", "rhizoplane", "rhizosphere"],
        "leaf_and_stem": ["leaf", "stem"],
        "leaf_stem_fruit_seed": ["leaf", "stem", "fruit", "seed"],
    }

    def matches(v: str, wanted: List[str]) -> bool:
        vv = (v or "").strip().lower()
        return any(vv == w for w in wanted)

    for subset_name, wanted in subsets.items():
        chosen = [run for run in run_ids if matches(niche.get(run, ""), wanted)]
        if len(chosen) < 2:
            for rank in ranks:
                _ensure_placeholder_files(Path(f"saved_matrices_{rank}") / subset_name, rank)
            continue
        for rank in ranks:
            subset_and_save_for_rank(rank, subset_name, chosen, policy, selected)
    log("[done] niche subsets complete")


def _normalize_species_value(x: str) -> str:
    x = str(x).strip().lower()
    if "[" in x:
        x = x.split("[", 1)[0].strip()
    return x


def run_species_subsets(metadata_csv: Path, ranks: List[str], species_col: str,
                        policy: str, selected: Set[str]):
    md = pd.read_csv(metadata_csv)
    if "Run" not in md.columns:
        raise ValueError("metadata missing Run column")
    if species_col not in md.columns:
        raise ValueError(f"metadata missing species column '{species_col}'")

    md["Run"] = md["Run"].astype(str).str.strip()
    sp = md.set_index("Run")[species_col].apply(_normalize_species_value)

    runs = sp.index.tolist()
    ids_rice = [r for r in runs if sp.get(r, "") == "rice"]
    ids_potato = [r for r in runs if sp.get(r, "") == "potato"]
    ids_other = [r for r in runs if sp.get(r, "") not in ("rice", "potato")]

    groups = {
        "plant_rice": ids_rice,
        "plant_potato": ids_potato,
        "plant_other": ids_other,
    }

    for subset_name, ids in groups.items():
        if len(ids) < 2:
            for rank in ranks:
                _ensure_placeholder_files(Path(f"saved_matrices_{rank}") / subset_name, rank)
            continue
        for rank in ranks:
            subset_and_save_for_rank(rank, subset_name, ids, policy, selected)
    log("[done] species subsets complete")


# =============================
# Shannon
# =============================

def shannon_vectorized(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=0)
    sums_safe = sums.copy()
    zero_cols = sums_safe == 0
    sums_safe[zero_cols] = 1.0

    p = matrix / sums_safe
    if zero_cols.any():
        p[:, zero_cols] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(p)
    logp[~np.isfinite(logp)] = 0.0
    logp[p <= 0] = 0.0

    H = -(p * logp).sum(axis=0)
    H[zero_cols] = 0.0
    return H


def compute_shannon_from_rel(rel_path: Path, out_path: Path,
                             policy: str, selected: Set[str]):
    target_id = f"SHANNON:{rel_path.parent}:{rel_path.name}"
    if not policy_should_recompute(target_id, out_path, policy, selected):
        return
    try:
        df = safe_read_csv(rel_path, low_memory=False)
    except Exception as e:
        log(f"[shannon] SKIP {rel_path}: {e}")
        return
    if df.shape[1] < 3:
        log(f"[shannon] SKIP {rel_path}: <2 sample columns")
        return
    data = df.iloc[:, 1:].to_numpy(dtype="float32", copy=True)
    data[data < 0] = 0.0
    H = shannon_vectorized(data)
    out_df = pd.DataFrame({"Sample": list(df.columns[1:]), "Shannon": H})
    out_df.to_csv(out_path, index=False, float_format="%.6f")


# =============================
# ANOSIM
# =============================

def parse_merge_rules(s: str) -> Dict[str, List[str]]:
    rules: Dict[str, List[str]] = {}
    s = (s or "").strip()
    if not s:
        return rules
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad --anosim-merge chunk: {chunk}")
        target, srcs = chunk.split("=", 1)
        target = target.strip().lower()
        src_list = [x.strip().lower() for x in srcs.split("|") if x.strip()]
        rules[target] = src_list
    return rules


def apply_binning(groups: pd.Series, merge_map: Dict[str, List[str]]) -> pd.Series:
    mapping = {}
    for target, srcs in merge_map.items():
        for s in srcs:
            mapping[s] = target
    return groups.map(lambda g: mapping.get(str(g).strip().lower(), str(g).strip().lower()))


def load_bc_csv(bc_path: Path):
    df = safe_read_csv(bc_path, index_col=0)
    if df.shape[0] != df.shape[1]:
        raise ValueError("bc_matrix must be square")
    if not df.columns.equals(df.index):
        raise ValueError("bc_matrix labels must match index")
    arr = df.values.astype(float)
    np.fill_diagonal(arr, 0.0)
    ids = list(map(str, df.index.tolist()))
    if not HAVE_SKBIO:
        raise ValueError("scikit-bio not installed")
    return DistanceMatrix(arr, ids=ids)


def _extract_anosim(res, fallback_perms: int) -> Tuple[float, float, int]:
    def get(names, default=np.nan):
        for n in names:
            if hasattr(res, n):
                try:
                    return getattr(res, n)
                except Exception:
                    pass
            try:
                return res[n]
            except Exception:
                pass
        return default

    R = get(["statistic", "R", "test_statistic", "test statistic", "stat_value"])
    p = get(["p_value", "p-value", "p value"])
    perms = get(["permutations", "number_of_permutations", "number of permutations"], default=fallback_perms)

    try:
        R = float(R)
    except Exception:
        R = np.nan
    try:
        p = float(p)
    except Exception:
        p = np.nan
    try:
        perms = int(perms)
    except Exception:
        perms = int(fallback_perms)
    return R, p, perms


def run_pairwise_anosim_for_folder(bc_path: Path, metadata_path: Path,
                                  group_col: str, merge_map: Dict[str, List[str]],
                                  pair_mode: str, permutations: int,
                                  policy: str, selected: Set[str]):
    if not HAVE_SKBIO:
        log(f"[anosim] SKIP (no scikit-bio) {bc_path.parent}")
        return
    if is_tiny(bc_path):
        log(f"[anosim] SKIP {bc_path.parent}: empty placeholder bc_matrix")
        return

    m = re.match(r"bc_matrix_(.+)\.csv$", bc_path.name)
    rank = m.group(1) if m else "rank"

    out_csv = bc_path.parent / f"anosim_pairwise_{rank}_{group_col}.csv"
    out_png = bc_path.parent / f"anosim_pairwise_{rank}_{group_col}.png"
    target_id = f"ANOSIM:{bc_path.parent}:{rank}:{group_col}:{pair_mode}"
    if not policy_should_recompute(target_id, out_csv, policy, selected):
        return

    try:
        dm = load_bc_csv(bc_path)
    except Exception as e:
        log(f"[anosim] FAILED {bc_path.parent} {rank}: {e}")
        return

    if len(dm.ids) < 4:
        log(f"[anosim] SKIP {bc_path.parent} {rank}: <4 samples")
        return

    meta = pd.read_csv(metadata_path)
    if "Run" not in meta.columns:
        log(f"[anosim] FAILED {bc_path.parent} {rank}: metadata missing Run")
        return
    if group_col not in meta.columns:
        log(f"[anosim] FAILED {bc_path.parent} {rank}: metadata missing {group_col}")
        return
    meta["Run"] = meta["Run"].astype(str).str.strip()
    groups_full = meta.set_index("Run")[group_col].astype(str).str.strip().str.lower()

    dm_ids = list(map(str, dm.ids))
    overlap = [sid for sid in dm_ids if sid in set(groups_full.index)]
    if len(overlap) < 4:
        log(f"[anosim] SKIP {bc_path.parent} {rank}: <4 overlapping samples")
        return

    pos = {sid: i for i, sid in enumerate(dm_ids)}
    idx = [pos[sid] for sid in overlap]
    arr = dm.data[np.ix_(idx, idx)]
    dm = DistanceMatrix(arr, ids=overlap)
    groups = groups_full.reindex(overlap)

    bad = groups.isna() | (groups.str.strip() == "") | (groups.str.lower() == "nan")
    if bad.any():
        overlap2 = [sid for sid in dm.ids if not bad.loc[sid]]
        if len(overlap2) < 4:
            log(f"[anosim] SKIP {bc_path.parent} {rank}: too many missing group labels")
            return
        pos2 = {sid: i for i, sid in enumerate(dm.ids)}
        idx2 = [pos2[sid] for sid in overlap2]
        dm = DistanceMatrix(dm.data[np.ix_(idx2, idx2)], ids=overlap2)
        groups = groups.reindex(overlap2)

    if merge_map:
        groups = apply_binning(groups, merge_map)

    cats = sorted(pd.Series(groups).unique().tolist())
    ids_by = {c: [sid for sid in dm.ids if groups[sid] == c] for c in cats}

    pairs = []
    if pair_mode == "other_plus_rest" and "other" in ids_by:
        remaining = [c for c in cats if c != "other"]
        for b in remaining:
            pairs.append(("other", b))
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                pairs.append((remaining[i], remaining[j]))
    else:
        for i in range(len(cats)):
            for j in range(i + 1, len(cats)):
                pairs.append((cats[i], cats[j]))

    rows = []
    for a, b in pairs:
        A = ids_by.get(a, [])
        B = ids_by.get(b, [])
        if len(A) < 2 or len(B) < 2:
            continue

        keep_set = set(A + B)
        keep = [sid for sid in dm.ids if sid in keep_set]
        if len(keep) < 4:
            continue

        pos3 = {sid: i for i, sid in enumerate(dm.ids)}
        idx3 = [pos3[sid] for sid in keep]
        dm2 = DistanceMatrix(dm.data[np.ix_(idx3, idx3)], ids=keep)
        g2 = pd.Series([groups[sid] for sid in dm2.ids], index=dm2.ids)

        err = ""
        try:
            res = anosim(dm2, g2, permutations=int(permutations))
            R, p, perms_used = _extract_anosim(res, int(permutations))
        except Exception as e:
            R, p, perms_used = np.nan, np.nan, int(permutations)
            err = str(e)

        rows.append({
            "source": bc_path.name,
            "group_col": group_col,
            "A": a,
            "B": b,
            "R": R,
            "p_value": p,
            "permutations": perms_used,
            "n_samples": int(len(dm2.ids)),
            "bin_map": ";".join(f"{t}<-{','.join(v)}" for t, v in merge_map.items()) if merge_map else "",
            "error": err,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if HAVE_MPL and not df.empty:
        plt.figure(figsize=(10, max(3, 0.25 * len(df))))
        y = np.arange(len(df))
        plt.barh(y, df["R"].fillna(0.0).values)
        plt.yticks(y, [f'{a} vs {b}' for a, b in zip(df["A"], df["B"])])
        plt.xlabel("ANOSIM R")
        plt.title(f"{bc_path.parent.name} — ANOSIM ({rank}, {group_col})")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()


# =============================
# Correlation
# =============================

def clean_taxa_index(rel: pd.DataFrame) -> pd.DataFrame:
    rel.index = rel.index.astype(str).str.strip()
    bad = set(["", "nan", "none", "na", "n/a"])
    rel = rel[~rel.index.str.lower().isin(bad)]
    rel = rel[~pd.isna(rel.index)]
    return rel


def select_taxa_for_corr(rel: pd.DataFrame, mode: str, topn: int,
                         fixed: Optional[List[str]] = None) -> List[str]:
    if mode == "fixed":
        fixed = fixed or []
        fixed2 = [str(x).strip() for x in fixed if str(x).strip()]
        present = set(rel.index.astype(str).tolist())
        return [t for t in fixed2 if t in present]

    means = rel.mean(axis=1, numeric_only=True).sort_values(ascending=False)
    chosen = means.head(int(topn)).index.astype(str).tolist()
    chosen = [t for t in chosen if str(t).strip().lower() not in ("nan", "none", "")]
    return chosen


def run_corr_for_folder(rel_path: Path, rank: str,
                        method: str, mode: str,
                        topn: int, fixed_list: Optional[List[str]],
                        policy: str, selected: Set[str]):
    out_dir = rel_path.parent
    mat_out = out_dir / f"rel_abundance_{rank}_{mode}_matrix.csv"
    corr_out = out_dir / f"corr_{rank}_{method}_{mode}.csv"
    png_out = out_dir / f"corr_{rank}_{method}_{mode}.png"

    target_id = f"CORR:{out_dir}:{rank}:{method}:{mode}:{topn}"
    if not policy_should_recompute(target_id, corr_out, policy, selected):
        return

    try:
        rel = safe_read_csv(rel_path, index_col=0)
    except Exception as e:
        log(f"[corr] SKIP {out_dir} {rank}: {e}")
        return

    if rel.shape[1] < 2:
        log(f"[corr] SKIP {out_dir} {rank}: <2 samples")
        return

    rel = rel.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rel = clean_taxa_index(rel)
    if rel.shape[0] < 2:
        log(f"[corr] SKIP {out_dir} {rank}: <2 taxa after cleaning")
        return

    chosen = select_taxa_for_corr(rel, mode=mode, topn=topn, fixed=fixed_list)
    if len(chosen) < 2:
        log(f"[corr] SKIP {out_dir} {rank}: <2 taxa selected")
        return

    try:
        sub = rel.loc[chosen]
    except Exception as e:
        log(f"[corr] FAILED {out_dir} {rank}: {e}")
        return

    sub.to_csv(mat_out)
    corr = sub.T.corr(method=method)
    np.fill_diagonal(corr.values, np.nan)
    corr.to_csv(corr_out)

    if HAVE_MPL:
        plt.figure(figsize=(10, 8))
        data = np.ma.masked_invalid(corr.values)
        v = np.nanmax(np.abs(corr.values))
        if np.isnan(v) or v == 0:
            v = 1.0
        im = plt.imshow(data, aspect="equal", vmin=-v, vmax=v)
        plt.colorbar(im, label=f"{method.title()} correlation")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title(f"{rank}: {method} correlation ({mode})")
        plt.tight_layout()
        plt.savefig(png_out, dpi=300)
        plt.close()


# =============================
# Postprocess scan
# =============================

def is_plain_rel_abundance(fname: str) -> bool:
    if not (fname.startswith("rel_abundance_") and fname.endswith(".csv")):
        return False
    core = fname[len("rel_abundance_"):-len(".csv")]
    if "_" in core:
        return False
    return True


def discover_rel_files(root: Path) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    found = []
    for dirpath, _, filenames in os.walk(str(root)):
        fnset = set(filenames)
        for f in fnset:
            if is_plain_rel_abundance(f):
                rank = f[len("rel_abundance_"):-len(".csv")]
                rel_path = Path(dirpath) / f
                bc_path = Path(dirpath) / f"bc_matrix_{rank}.csv"
                found.append((rank, Path(dirpath), rel_path, bc_path if bc_path.exists() else None))
    return found


def postprocess_everywhere(root: Path, metadata_path: Path,
                           do_shannon: bool, do_anosim: bool, do_corr: bool,
                           anosim_factor: str, merge_map: Dict[str, List[str]],
                           anosim_pairs: str, anosim_perms: int,
                           corr_rank: str, corr_method: str, corr_mode: str, corr_topn: int,
                           corr_taxa_list: Optional[List[str]],
                           policy: str, selected: Set[str]):
    rel_targets = discover_rel_files(root)
    log(f"[postprocess] found {len(rel_targets)} rel_abundance_<rank>.csv targets under {root}")

    for rank, d, rel_path, bc_path in rel_targets:
        if do_shannon:
            out_sh = d / f"shannon_{rank}.csv"
            compute_shannon_from_rel(rel_path, out_sh, policy, selected)

        if do_anosim and bc_path is not None:
            run_pairwise_anosim_for_folder(bc_path, metadata_path,
                                           group_col=anosim_factor,
                                           merge_map=merge_map,
                                           pair_mode=anosim_pairs,
                                           permutations=anosim_perms,
                                           policy=policy, selected=selected)

        if do_corr and rank == corr_rank:
            run_corr_for_folder(rel_path, rank=rank,
                                method=corr_method, mode=corr_mode,
                                topn=corr_topn, fixed_list=corr_taxa_list,
                                policy=policy, selected=selected)

    log("[done] postprocess complete")


# =============================
# Redo list template
# =============================

def write_redo_list_template(root: Path, out_path: Path):
    rel_targets = discover_rel_files(root)

    lines = []
    lines.append("# redo_list.txt — targets to recompute when using: --recalc selected")
    lines.append("# One target ID per line. Lines starting with # are ignored.")
    lines.append("#")
    lines.append("# Examples (uncomment by removing '# '):")
    lines.append("# SHANNON:/path/to/folder:rel_abundance_phylum.csv")
    lines.append("# ANOSIM:/path/to/folder:ASV:niche_category:allpairs")
    lines.append("# CORR:/path/to/folder:phylum:spearman:topn:10")
    lines.append("#")
    lines.append("# ---- Auto-discovered candidates (copy/paste or edit) ----")
    lines.append("")

    for rank, d, rel_path, bc_path in rel_targets[:200]:
        lines.append(f"# SHANNON:{rel_path.parent}:{rel_path.name}")
    lines.append("")
    for rank, d, rel_path, bc_path in rel_targets[:200]:
        if bc_path is not None and bc_path.exists():
            lines.append(f"# ANOSIM:{bc_path.parent}:{rank}:niche_category:allpairs")
    lines.append("")
    for rank, d, rel_path, bc_path in rel_targets[:200]:
        if rank == "phylum":
            lines.append(f"# CORR:{rel_path.parent}:{rank}:spearman:topn:10")

    out_path.write_text("\n".join(lines) + "\n")
    log(f"[redo-list] wrote template: {out_path}")


# =============================
# Main ASV stage
# =============================

def run_asv_pipeline(args, policy: str, selected: Set[str]):
    root = Path(args.root).resolve()
    os.chdir(str(root))

    all_fasta = Path(args.all_fasta).resolve()
    silva_db = Path(args.silva_db).resolve()
    metadata_in = Path(args.metadata).resolve()
    metadata_out = Path(args.metadata_out).resolve() if args.metadata_out else metadata_in

    filt_fa = root / "all_samples_filtered.fasta"
    derep_fa = root / "dereplicated.fasta"
    derep2_fa = root / "dereplicated_no_singletons.fasta"
    asvs_fa = root / "ASVs.fasta"
    asv_table = root / "ASV_table.txt"
    blast_out = root / "ASV_blast.txt"
    taxa_tsv = root / "ASV_taxa_separated.txt"
    asv_table_filt_path = root / "ASV_table.filtered.txt"

    exclude_rules = parse_exclusions(args.exclude)
    runs_excl = runs_to_exclude_from_metadata(metadata_in, exclude_rules) if exclude_rules else set()
    if runs_excl:
        log(f"[exclude] will exclude {len(runs_excl)} runs")

    if policy_should_recompute(f"ASV:filter_fasta:{filt_fa}", filt_fa, policy, selected):
        filter_fasta_length(all_fasta, filt_fa, args.min_len, args.max_len)

    if policy_should_recompute(f"ASV:derep:{derep_fa}", derep_fa, policy, selected):
        run(["vsearch", "--derep_fulllength", str(filt_fa),
             "--output", str(derep_fa), "--sizeout", "--relabel", "ASV_"])

    if policy_should_recompute(f"ASV:sort:{derep2_fa}", derep2_fa, policy, selected):
        run(["vsearch", "--sortbysize", str(derep_fa),
             "--output", str(derep2_fa), "--minsize", "2"])

    if policy_should_recompute(f"ASV:unoise:{asvs_fa}", asvs_fa, policy, selected):
        run(["vsearch", "--cluster_unoise", str(derep2_fa),
             "--minsize", str(args.unoise_minsize), "--unoise_alpha", str(args.unoise_alpha),
             "--centroids", str(asvs_fa)])

    if policy_should_recompute(f"ASV:otutabout:{asv_table}", asv_table, policy, selected):
        run(["vsearch", "--usearch_global", str(all_fasta), "--db", str(asvs_fa),
             "--id", str(args.identity), "--otutabout", str(asv_table)])

    if policy_should_recompute(f"ASV:blast:{blast_out}", blast_out, policy, selected):
        run(["blastn", "-query", str(asvs_fa), "-db", str(silva_db),
             "-num_threads", str(args.threads), "-outfmt", "6", "-max_target_seqs", "1",
             "-out", str(blast_out)])

    asv_tax = parse_blast_taxonomy(blast_out, taxa_tsv)
    bad_asvs = asvs_to_exclude(asv_tax, blast_out)

    if policy_should_recompute(f"ASV:table_filtered:{asv_table_filt_path}", asv_table_filt_path, policy, selected):
        tab = load_asv_table(asv_table)
        tab_f = filter_asv_table(tab, bad_asvs, args.postfilter_min_sample_sum, asv_table_filt_path)

        if runs_excl:
            tab_f = drop_samples_from_asv_table(tab_f, runs_excl, root / "excluded_samples.txt")
            tab_f.to_csv(asv_table_filt_path, sep="\t")
            log(f"[asv] wrote {asv_table_filt_path}")
    else:
        tab_f = load_asv_table(asv_table_filt_path)

    compute_most_abundant(tab_f, asv_tax, metadata_in, metadata_out, policy, selected)

    ranks = ["ASV", "kingdom", "phylum", "class", "order", "family", "genus"]
    for rank in ranks:
        out_dir = root / f"saved_matrices_{rank}"
        out_dir.mkdir(exist_ok=True)

        rel_csv = out_dir / f"rel_abundance_{rank}.csv"
        bc_csv = out_dir / f"bc_matrix_{rank}.csv"
        coord_csv = out_dir / f"pcoa_coordinates_{rank}.csv"
        eig_csv = out_dir / f"eigenvalues_{rank}.csv"

        target_id = f"RANK:{rank}:{out_dir}"
        if not policy_should_recompute(target_id, coord_csv, policy, selected) and \
           all(is_nonempty(p) for p in [rel_csv, bc_csv, coord_csv, eig_csv]):
            log(f"[{rank}] skip (exists)")
            continue

        if rank == "ASV":
            rel = tab_f.div(tab_f.sum(axis=0), axis=1).fillna(0)
        else:
            rel = aggregate_by_rank(tab_f, asv_tax, rank)

        zero_samps = rel.columns[rel.sum(axis=0) == 0].tolist()
        if zero_samps:
            rel = rel.drop(columns=zero_samps)

        if rel.shape[1] < 2:
            log(f"[{rank}] skip (<2 samples)")
            continue

        rel.to_csv(rel_csv)
        log(f"[{rank}] wrote {rel_csv}")

        bc = compute_bc_matrix(rel)
        bc.to_csv(bc_csv)
        log(f"[{rank}] wrote {bc_csv}")

        coords, eig = pcoa(bc, n_components=10)
        coords.to_csv(coord_csv)
        eig.to_csv(eig_csv)
        log(f"[{rank}] wrote {coord_csv} and {eig_csv}")

    log("[done] ASV + base ranks complete")


# =============================
# Entry point
# =============================

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else (root / "unified.log")
    log_fp = open(log_path, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fp)
    sys.stderr = Tee(sys.__stderr__, log_fp)

    selected = load_selected_targets(Path(args.redo_list))
    policy = args.recalc

    log("=" * 80)
    log(f"[start] {datetime.now().isoformat(timespec='seconds')}")
    log(f"[root]  {root}")
    log(f"[policy] {policy}")
    log(f"[redo-list] {Path(args.redo_list).resolve()}")
    log(f"[metadata] {Path(args.metadata).resolve()}")
    log(f"[exclude] {parse_exclusions(args.exclude)}")
    log("=" * 80)

    if args.make_redo_list:
        write_redo_list_template(root, Path(args.redo_list).resolve())
        log("[done] redo list template created")
        return

    if args.run_asv:
        if not args.all_fasta or not args.silva_db:
            raise SystemExit("--all-fasta and --silva-db are required for --run-asv")
        run_asv_pipeline(args, policy, selected)

    ranks = ["ASV", "kingdom", "phylum", "class", "order", "family", "genus"]
    meta_for_subsets = Path(args.metadata_out).resolve() if args.metadata_out else Path(args.metadata).resolve()

    if args.make_niche_subsets:
        run_niche_subsets(meta_for_subsets, ranks=ranks, niche_col=args.niche_col, policy=policy, selected=selected)

    if args.make_species_subsets:
        if not args.species_col:
            raise SystemExit("--species-col is required for --make-species-subsets")
        run_species_subsets(meta_for_subsets, ranks=ranks, species_col=args.species_col, policy=policy, selected=selected)

    if args.postprocess:
        merge_map = parse_merge_rules(args.anosim_merge)

        corr_taxa_list = None
        if args.corr_mode == "fixed":
            if args.corr_taxa_file:
                p = Path(args.corr_taxa_file)
                corr_taxa_list = [l.strip() for l in p.read_text().splitlines()
                                  if l.strip() and not l.strip().startswith("#")]
            elif args.corr_taxa:
                corr_taxa_list = [x.strip() for x in args.corr_taxa.split(",") if x.strip()]
            else:
                raise SystemExit("Fixed correlation mode requires --corr-taxa or --corr-taxa-file")

        postprocess_everywhere(
            root=root,
            metadata_path=meta_for_subsets,
            do_shannon=args.do_shannon,
            do_anosim=args.do_anosim,
            do_corr=args.do_corr,
            anosim_factor=args.anosim_factor,
            merge_map=merge_map,
            anosim_pairs=args.anosim_pairs,
            anosim_perms=args.anosim_perms,
            corr_rank=args.corr_rank,
            corr_method=args.corr_method,
            corr_mode=args.corr_mode,
            corr_topn=args.corr_topn,
            corr_taxa_list=corr_taxa_list,
            policy=policy,
            selected=selected
        )

    log("[done]")


if __name__ == "__main__":
    main()

