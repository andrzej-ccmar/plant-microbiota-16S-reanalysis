#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code 1 (unified): From SRA accession(s) to final FASTA (per-sample)
==================================================================

For each accession (SRR/ERR/DRR/CRR etc.), this script generates a final FASTA that:
  - contains up to FINAL_READS_PER_SAMPLE reads (default: 1000)
  - is quality-filtered (vsearch fastq_filter; default maxEE=1.0)
  - has primers trimmed (user-configurable primer pair)
  - is forced into a consistent orientation (auto-orient by primer detection)
  - is length-filtered (median ±10%, optional strict window)
  - has reads removed if they BLAST-match mitochondria / chloroplast / host DBs above a threshold

NEW (Jan 2026):
---------------
- Optional FASTQ truncation BEFORE merging/filtering:
    Keep only first FASTQ_HEAD_LINES lines (default: 200000) after fastq-dump.
  (FASTQ = 4 lines/read; 200000 lines = 50,000 reads.)

External tools expected on PATH
-------------------------------
- prefetch, fastq-dump (SRA Toolkit)
- flash (optional; used if paired-end exists)
- vsearch
- blastn

CONFIGURATION (edit here first)
-------------------------------
"""

# =========================
# === USER CONFIG (EDIT) ===
# =========================

# Primer pair (IUPAC ambiguity allowed; anything not A/T/C/G is treated as "any base")
PRIMER_FWD = "GTGXCXGCMGCCGCGGTAA"   # e.g. 515F
PRIMER_REV = "GACTACHVGGGTWTCTAAT"   # e.g. 806R (note: NOT reverse-complement; we handle that)

# FASTQ truncation BEFORE merging/filtering (set 0 to disable)
FASTQ_HEAD_LINES = 200000            # 200000 lines = 50000 reads

# FLASH overlap parameters (only used when paired-end reads exist)
FLASH_MIN_OVERLAP = 50
FLASH_MAX_OVERLAP = 400

# Quality filtering (vsearch)
VSEARCH_MAXEE = 1.0

# BLAST identity threshold for filtering contaminant reads
BLAST_IDENTITY_THRESHOLD = 95.0

# Final number of reads per sample (FASTA records)
FINAL_READS_PER_SAMPLE = 1000

# Pre-BLAST cap (to avoid blasting huge intermediate sets; we blast at most this many reads)
PREBLAST_CAP_READS = 5000

# Length filtering
APPLY_MEDIAN_PLUS_MINUS_10PCT = True
APPLY_STRICT_LENGTH_WINDOW = False
STRICT_MIN_LEN = 252
STRICT_MAX_LEN = 254

# Randomness (used for downsampling)
RANDOM_SEED = 13

# =========================
# === END USER CONFIG =====
# =========================


import argparse
import re
import sys
import shutil
import random
import subprocess as sp
from pathlib import Path
from typing import List, Tuple, Optional, Set

try:
    from Bio import SeqIO
except Exception as e:
    sys.exit("Biopython is required (from Bio import SeqIO). Install or load it on the cluster.\nError: %s" % e)


# -----------------------------
# Helpers: logging + subprocess
# -----------------------------
def log(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def run(cmd: List[str], cwd: Optional[Path] = None):
    log("[RUN] " + " ".join(cmd))
    sp.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def which_or_die(tool: str):
    if shutil.which(tool) is None:
        sys.exit(f"Required tool not found on PATH: {tool}")

def ensure_tools():
    for t in ["prefetch", "fastq-dump", "vsearch", "blastn"]:
        which_or_die(t)
    # flash is optional: we only require it if paired-end reads exist
    if shutil.which("flash") is None:
        log("[WARN] 'flash' not found on PATH. Paired-end runs will fall back to single-end processing.")


# -----------------------------
# Optional: truncate FASTQ
# -----------------------------
def truncate_fastq_head(path: Path, n_lines: int):
    """
    Overwrite FASTQ with only the first n_lines lines, using an atomic replace.
    (Keeps file permissions when possible.)
    """
    if n_lines <= 0:
        return
    if not path.exists() or path.stat().st_size == 0:
        return

    tmp = path.with_suffix(path.suffix + ".headtmp")

    # Use head via bash for speed and simplicity
    run(["bash", "-lc", f'head -n {int(n_lines)} "{path}" > "{tmp}"'])

    # Preserve permissions
    try:
        shutil.copymode(str(path), str(tmp))
    except Exception:
        pass

    tmp.replace(path)


# -----------------------------
# Primer utilities (IUPAC-light)
# -----------------------------
def primer_to_regex(primer: str) -> str:
    """Convert primer into regex; non-ATCG bases become [ATCG]."""
    out = []
    for b in primer.upper():
        out.append(b if b in "ATCG" else "[ATCG]")
    return "".join(out)

def revcomp(seq: str) -> str:
    comp = {"A":"T","T":"A","C":"G","G":"C",
            "a":"t","t":"a","c":"g","g":"c"}
    return "".join(comp.get(b, b) for b in reversed(seq))

def trim_by_primers(seq: str, fwd: str, rev: str) -> Tuple[str, bool, bool]:
    """
    Trim sequence by locating:
      - forward primer (fwd) and trimming up to its end
      - reverse primer *reverse complement* and trimming from its start
    Returns (trimmed_seq, fwd_found, rev_found)
    """
    import re
    fwd_pat = re.compile(primer_to_regex(fwd))
    rev_rc = revcomp(rev)
    rev_pat = re.compile(primer_to_regex(rev_rc))

    m1 = fwd_pat.search(seq)
    if m1:
        start = m1.end()
        fwd_found = True
    else:
        start = 0
        fwd_found = False

    m2 = rev_pat.search(seq, pos=start)
    if m2:
        end = m2.start()
        rev_found = True
    else:
        end = len(seq)
        rev_found = False

    return seq[start:end], fwd_found, rev_found

def auto_orient_and_trim(seq: str, fwd: str, rev: str) -> Optional[str]:
    """
    Try trimming in the given orientation; if forward primer is not found,
    try reverse-complementing the whole sequence and trimming again.
    """
    trimmed, f_ok, _ = trim_by_primers(seq, fwd, rev)
    if f_ok and trimmed:
        return trimmed

    seq_rc = revcomp(seq)
    trimmed2, f_ok2, _ = trim_by_primers(seq_rc, fwd, rev)
    if f_ok2 and trimmed2:
        return trimmed2

    # Best-effort fallback
    if trimmed:
        return trimmed
    if trimmed2:
        return trimmed2
    return None


# -----------------------------
# BLAST-based filtering helpers
# -----------------------------
def blast_hits_to_remove(query_fa: Path, db: Path, out_txt: Path,
                         pident_threshold: float, threads: int) -> Set[str]:
    """
    Run blastn and return set of query IDs with pident >= threshold.
    Uses outfmt 6: qseqid sseqid pident ...
    """
    cmd = [
        "blastn",
        "-query", str(query_fa),
        "-db", str(db),
        "-num_threads", str(threads),
        "-outfmt", "6",
        "-max_target_seqs", "1",
        "-out", str(out_txt),
    ]
    run(cmd)

    to_remove: Set[str] = set()
    if not out_txt.exists() or out_txt.stat().st_size == 0:
        return to_remove

    with out_txt.open("r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid = parts[0]
            try:
                pident = float(parts[2])
            except Exception:
                continue
            if pident >= pident_threshold:
                to_remove.add(qid)
    return to_remove


# -----------------------------
# FASTA utilities
# -----------------------------
def write_fasta(records: List[Tuple[str, str]], out_fa: Path):
    """Write single-line FASTA."""
    with out_fa.open("w") as out:
        for rid, seq in records:
            out.write(f">{rid}\n{seq}\n")

def downsample(records: List[Tuple[str, str]], n: int, seed: int) -> List[Tuple[str, str]]:
    if len(records) <= n:
        return records
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    keep = sorted(idx[:n])
    return [records[i] for i in keep]


# -----------------------------
# Main per-accession pipeline
# -----------------------------
ACC_RE = re.compile(r"^[A-Z]{3}\d+$")  # SRR/ERR/DRR/CRR etc.

def infer_accessions(args) -> List[str]:
    accs: List[str] = []
    if args.accession:
        accs.extend(args.accession)
    if args.accessions_file:
        for line in Path(args.accessions_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            accs.append(line.split()[0])
    # de-dup preserve order
    seen = set()
    out = []
    for a in accs:
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out

def find_fastq_pair(workdir: Path, acc: str) -> Tuple[Optional[Path], Optional[Path]]:
    r1 = workdir / f"{acc}_1.fastq"
    r2 = workdir / f"{acc}_2.fastq"
    if r1.exists() and r1.stat().st_size > 0:
        if r2.exists() and r2.stat().st_size > 0:
            return r1, r2
        return r1, None
    se = workdir / f"{acc}.fastq"
    if se.exists() and se.stat().st_size > 0:
        return se, None
    return None, None

def vsearch_filter_fastq(in_fq: Path, out_fq: Path):
    run(["vsearch", "--fastq_filter", str(in_fq), "--fastq_maxee", str(VSEARCH_MAXEE), "--fastqout", str(out_fq)])

def fastq_to_fasta(in_fq: Path, out_fa: Path):
    with in_fq.open("r") as ih, out_fa.open("w") as oh:
        for rec in SeqIO.parse(ih, "fastq"):
            oh.write(f">{rec.id}\n{str(rec.seq)}\n")

def load_fasta_records(in_fa: Path) -> List[Tuple[str, str]]:
    records = []
    with in_fa.open("r") as ih:
        for rec in SeqIO.parse(ih, "fasta"):
            records.append((rec.id, str(rec.seq)))
    return records

def length_filter_records(records: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if not records:
        return records
    seq_lens = [len(s) for _, s in records]
    if APPLY_MEDIAN_PLUS_MINUS_10PCT:
        import statistics
        med = statistics.median(seq_lens)
        lo = med * 0.9
        hi = med * 1.1
        records = [(i,s) for i,s in records if lo <= len(s) <= hi]
    if APPLY_STRICT_LENGTH_WINDOW:
        records = [(i,s) for i,s in records if STRICT_MIN_LEN <= len(s) <= STRICT_MAX_LEN]
    return records

def process_accession(acc: str, args) -> Optional[Path]:
    if not ACC_RE.match(acc):
        log(f"[WARN] Accession '{acc}' doesn't look like SRR/ERR/DRR/CRR; continuing anyway.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    workdir = Path(args.workdir).resolve() / f"tmp_{acc}"
    if args.clean and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    final_out = outdir / f"{acc}_final.fasta"
    if final_out.exists() and final_out.stat().st_size > 0 and args.recalc == "missing":
        log(f"[SKIP] {acc}: final exists -> {final_out}")
        return final_out
    if args.recalc == "none" and final_out.exists() and final_out.stat().st_size > 0:
        log(f"[SKIP] {acc}: recalc=none and final exists -> {final_out}")
        return final_out

    log(f"\n=== {acc} ===")
    log(f"[workdir] {workdir}")

    run(["prefetch", acc], cwd=workdir)
    run(["fastq-dump", "--split-files", acc], cwd=workdir)

    r1, r2 = find_fastq_pair(workdir, acc)
    if r1 is None:
        log(f"[FAIL] {acc}: could not locate fastq outputs after fastq-dump")
        return None

    if FASTQ_HEAD_LINES and FASTQ_HEAD_LINES > 0:
        if FASTQ_HEAD_LINES % 4 != 0:
            log(f"[WARN] FASTQ_HEAD_LINES={FASTQ_HEAD_LINES} not divisible by 4; FASTQ may be malformed.")
        log(f"[fastq] truncating to first {FASTQ_HEAD_LINES} lines")
        truncate_fastq_head(r1, FASTQ_HEAD_LINES)
        if r2 is not None:
            truncate_fastq_head(r2, FASTQ_HEAD_LINES)

    merged_fq = workdir / f"{acc}_merged.fastq"
    if r2 is not None and shutil.which("flash") is not None:
        log("[merge] paired-end detected -> FLASH")
        run(["flash", "-m", str(FLASH_MIN_OVERLAP), "-M", str(FLASH_MAX_OVERLAP), "-O", "-o", acc, str(r1), str(r2)], cwd=workdir)
        ext = workdir / f"{acc}.extendedFrags.fastq"
        if ext.exists() and ext.stat().st_size > 0:
            ext.rename(merged_fq)
        else:
            log("[merge] FLASH produced no extendedFrags; falling back to single-end")
            merged_fq = r1
    else:
        log("[merge] single-end (or FLASH unavailable) -> using R1")
        merged_fq = r1

    filt_fq = workdir / f"{acc}_filtered.fastq"
    vsearch_filter_fastq(merged_fq, filt_fq)
    if not filt_fq.exists() or filt_fq.stat().st_size == 0:
        log(f"[FAIL] {acc}: vsearch produced empty filtered fastq")
        return None

    raw_fa = workdir / f"{acc}_filtered.fasta"
    fastq_to_fasta(filt_fq, raw_fa)

    raw_records = load_fasta_records(raw_fa)
    trimmed: List[Tuple[str, str]] = []
    for rid, seq in raw_records:
        t = auto_orient_and_trim(seq, PRIMER_FWD, PRIMER_REV)
        if t:
            trimmed.append((rid, t))
    if not trimmed:
        log(f"[FAIL] {acc}: no reads left after primer trimming/orientation")
        return None

    trimmed = length_filter_records(trimmed)
    if not trimmed:
        log(f"[FAIL] {acc}: no reads left after length filtering")
        return None

    preblast = downsample(trimmed, PREBLAST_CAP_READS, seed=RANDOM_SEED)
    preblast_fa = workdir / f"{acc}_preblast.fasta"
    write_fasta(preblast, preblast_fa)

    to_remove: Set[str] = set()

    def maybe_filter(db_path: Optional[str], tag: str):
        nonlocal to_remove
        if not db_path:
            return
        db = Path(db_path)
        if not db.exists():
            log(f"[WARN] {acc}: {tag} DB not found: {db} (skipping)")
            return
        out_txt = workdir / f"{acc}_{tag}.blast6"
        rm = blast_hits_to_remove(preblast_fa, db, out_txt, BLAST_IDENTITY_THRESHOLD, threads=args.threads)
        log(f"[blast:{tag}] flagged {len(rm)} reads (pident>={BLAST_IDENTITY_THRESHOLD})")
        to_remove |= rm

    maybe_filter(args.mito_db, "mito")
    maybe_filter(args.chloro_db, "chloro")
    maybe_filter(args.host_db, "host")

    if to_remove:
        trimmed = [(rid, seq) for rid, seq in trimmed if rid not in to_remove]
    if not trimmed:
        log(f"[FAIL] {acc}: all reads removed by contaminant filtering")
        return None

    final_records = downsample(trimmed, FINAL_READS_PER_SAMPLE, seed=RANDOM_SEED)
    write_fasta(final_records, final_out)
    log(f"[OK] {acc}: wrote {len(final_records)} reads -> {final_out}")

    if args.clean:
        shutil.rmtree(workdir, ignore_errors=True)
    return final_out


def build_runs_from_sratable(glob_pat: str, out_path: Path):
    import glob
    files = sorted(glob.glob(glob_pat))
    if not files:
        raise ValueError(f"No files matched: {glob_pat}")

    runs = set()
    for fn in files:
        with open(fn, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i == 1:
                    continue
                first = line.split(",", 1)[0].strip().strip('"')
                if first and first.lower() != "run":
                    runs.add(first)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(runs)) + "\n")
    log(f"[runs] wrote {len(runs)} accessions -> {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Unified pipeline: SRA accession(s) -> final FASTA (trimmed; filtered; oriented; mito/chloro/host removed)."
    )
    ap.add_argument("--accession", action="append", default=[], help="Accession to process (repeatable).")
    ap.add_argument("--accessions-file", default=None, help="Text file with one accession per line.")
    ap.add_argument("--outdir", default="final_fastas", help="Output directory for *_final.fasta files.")
    ap.add_argument("--workdir", default="work", help="Working directory for temp files (tmp_<acc>/).")
    ap.add_argument("--threads", type=int, default=8, help="Threads for BLAST.")
    ap.add_argument("--mito-db", dest="mito_db", default=None, help="BLAST DB basename/path for mitochondria filtering (optional).")
    ap.add_argument("--chloro-db", dest="chloro_db", default=None, help="BLAST DB basename/path for chloroplast filtering (optional).")
    ap.add_argument("--host-db", dest="host_db", default=None, help="BLAST DB basename/path for host filtering (optional).")
    ap.add_argument("--recalc", choices=["missing", "all", "none"], default="missing",
                    help="missing=only build missing finals; all=recompute and overwrite; none=skip if final exists.")
    ap.add_argument("--clean", action="store_true", help="Delete per-accession tmp folder after success.")

    ap.add_argument("--sra-table-glob", dest="sra_table_glob", default=None,
                    help="Glob for SraRunTable CSVs (accession must be column 1). Builds runs list automatically.")
    ap.add_argument("--runs-out", default="runs.txt",
                    help="Where to write runs list when using --sra-table-glob (default: runs.txt)")

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_tools()

    if args.sra_table_glob:
        runs_out = Path(args.runs_out).resolve()
        build_runs_from_sratable(args.sra_table_glob, runs_out)
        args.accessions_file = str(runs_out)

    accs = infer_accessions(args)
    if not accs:
        sys.exit("No accessions provided. Use --accession or --accessions-file (or --sra-table-glob).")

    ok = 0
    fail = 0
    for acc in accs:
        try:
            out = process_accession(acc, args)
            if out:
                ok += 1
            else:
                fail += 1
        except sp.CalledProcessError as e:
            fail += 1
            log(f"[FAIL] {acc}: external command failed: {e}")
        except Exception as e:
            fail += 1
            log(f"[FAIL] {acc}: {e}")

    log(f"\nDone. Success={ok} Fail={fail}")
    if fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
