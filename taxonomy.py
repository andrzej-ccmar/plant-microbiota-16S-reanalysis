#!/usr/bin/env python3
"""
Taxonomy Smart Bars — inline plots + optional export + double x-axis

Run:
  python taxonomy_smart9_fixed.py
"""

import os
import sys
import glob
import hashlib
import json
import datetime
from typing import Optional
import numpy as np

def _bray_curtis_distance(u, v=None):
    """
    Bray–Curtis distance.

    Supports:
      - 1D vs 1D: _bray_curtis_distance(u, v) -> float
      - 2D matrix (rows = samples, cols = features): _bray_curtis_distance(M) -> NxN distance matrix

    Notes:
      - If a row sums to zero, its distances are defined as 0.0 to any other all-zero row,
        and 1.0 to any non-zero row (consistent with Bray–Curtis limits).
    """
    U = np.asarray(u, dtype=float)

    # Pairwise distances for a whole matrix
    if v is None and U.ndim == 2:
        n = U.shape[0]
        # Precompute row sums to speed up denominator
        row_sums = U.sum(axis=1)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            ui = U[i]
            si = row_sums[i]
            for j in range(i + 1, n):
                vj = U[j]
                sj = row_sums[j]
                den = si + sj
                if den == 0:
                    d = 0.0
                else:
                    num = np.sum(np.abs(ui - vj))
                    d = num / den
                D[i, j] = d
                D[j, i] = d
        return D

    # Single pair distance
    if v is None:
        raise TypeError("_bray_curtis_distance(u, v): missing required argument 'v' for 1D inputs")

    V = np.asarray(v, dtype=float)
    num = np.sum(np.abs(U - V))
    den = np.sum(U + V)
    if den == 0:
        return 0.0
    return float(num / den)
import pandas as pd

# --- Metadata table (lazy-loaded) ---
metadata = pd.DataFrame()
_METADATA_SRC = None

# Mapping from sample/run id -> metadata dict (built from `metadata`)
run2meta = {}

def _rebuild_run2meta():
    """(Re)build `run2meta` from the global `metadata` dataframe."""
    global run2meta, metadata
    try:
        if metadata is None or getattr(metadata, "empty", True):
            run2meta = {}
            return
        d = metadata.to_dict(orient="index")
        # Ensure keys are strings (sample IDs)
        run2meta = {str(k): v for k, v in d.items()}
    except Exception:
        run2meta = {}

bbox_to_anchor = None


# --- Matplotlib setup (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import panel as pn
# Older Panel versions may not recognize the 'matplotlib' extension name.
# Matplotlib panes still work without explicitly enabling it.
pn.extension()  # no global sizing_mode here

# --- ANOVA letters cache for plot annotation ---
# key: (taxon, group_factor) -> {group_value: letter}
ANOVA_LETTERS = {}


# ==============================
# Constants & metadata discovery
# ==============================
BASE_OUTPUT_PREFIX = "./saved_matrices"
LEVELS = ["ASV", "kingdom", "phylum", "class", "order", "family", "genus"]

# UI sentinel values
NICHE_ALL = "(All)"
PLANT_ALL = "(All)"


# ==============================
# Metadata (fixed path)
# ==============================
# Per your requirement, the app will ALWAYS load metadata from this file and will NOT search elsewhere.
METADATA_FIXED_PATH = "/usr/local/storage/ebi-data/metadata_onlyhere/metadata_17March_with_taxa.csv"

def _discover_metadata_file() -> str | None:
    # Use ONLY the fixed metadata file.
    p = METADATA_FIXED_PATH
    return p if (p and os.path.isfile(p)) else None


def _read_metadata_file(path: str) -> pd.DataFrame:
    # Try to auto-detect delimiter; fall back to comma / tab.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path)

    if df.empty:
        return pd.DataFrame()

    # Choose an index column
    idx_col = None
    for cand in ["sample", "sample_id", "SampleID", "run", "Run", "accession", "Accession", "id", "ID"]:
        if cand in df.columns:
            idx_col = cand
            break
    if idx_col is None:
        idx_col = df.columns[0]

    df[idx_col] = df[idx_col].astype(str).str.strip()
    df = df.set_index(idx_col, drop=True)

    # Drop duplicated sample IDs (keep first)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]

    # Ensure all factor columns are strings where sensible (keep numeric as-is)
    # (No-op for most operations; helps with MultiSelect labels.)
    return df


def _autoload_metadata_if_available():
    global metadata, _METADATA_SRC, run2meta
    if metadata is not None and not metadata.empty:
        return
    p = _discover_metadata_file()
    if not p:
        return
    try:
        md = _read_metadata_file(p)
        if md is not None and not md.empty:
            metadata = md
            _METADATA_SRC = p
            _rebuild_run2meta()
    except Exception:
        # Keep empty metadata on failure
        return


def _ensure_metadata_for_samples(sample_ids):
    """Ensure `metadata` exists and contains rows for the given sample_ids."""
    global metadata, run2meta
    sample_ids = [str(s) for s in sample_ids]
    if metadata is None or metadata.empty:
        metadata = pd.DataFrame(index=sample_ids)
        _rebuild_run2meta()
        return
    # Add missing sample rows (with NaNs)
    missing = [s for s in sample_ids if s not in metadata.index]
    if missing:
        add = pd.DataFrame(index=missing)
        metadata = pd.concat([metadata, add], axis=0)
    _rebuild_run2meta()


def _refresh_metadata_dependent_widgets():
    """Update widget option lists that depend on metadata columns."""
    cols = list(metadata.columns) if metadata is not None else []
    opts = ["(None)"] + cols

    for wname in ["group_factor", "outer_group_factor", "pcoa_color_by", "pcoa_shape_by"]:
        w = globals().get(wname)
        if w is None:
            continue
        old = getattr(w, "value", "(None)")
        w.options = opts
        w.value = old if old in w.options else "(None)"

    # ANOVA study column selector (optional)
    w = globals().get("anova_study_col")
    if w is not None:
        old = getattr(w, "value", "(None)")
        w.options = opts
        w.value = old if old in w.options else "(None)"


# Try to load metadata immediately (if present)
_autoload_metadata_if_available()

# ==============================
# Optional taxonomy-to-higher-rank mapping (ASV_blast.txt)
# ==============================
# This enables filtering taxa lists (ASV/genus/family/...) by a higher rank (e.g. keep only Proteobacteria).
# Expected ASV_blast format (tab-delimited):
#   ASV_1;size=... <TAB> <accession>_<Kingdom;Phylum;Class;Order;Family;Genus;Species> <TAB> ...
#
# The app will try these locations (first existing wins):
ASV_BLAST_CANDIDATES = [
    "./ASV_blast.txt",
    "./ASV_blast.tsv",

]

TAX_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Cache (lazy-loaded)
_TAX_FILTER_MAP = None  # dict[level][filter_rank][taxon] -> set(values)
_TAX_FILTER_SRC = None  # path used
_PENDING_TAX_FILTER_VALUES = []  # used when loading presets before data is loaded


def _normalize_asv_id(x: str) -> str:
    """Return 'ASV_123' from 'ASV_123;size=...' etc."""
    s = str(x).strip()
    if not s:
        return s
    return s.split(";", 1)[0].strip()


def _find_asv_blast_file() -> str | None:
    for p in ASV_BLAST_CANDIDATES:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def _parse_taxonomy_path(raw: str) -> list[str]:
    """Extract the semicolon taxonomy path from 'acc_Bacteria;...;Species'."""
    s = str(raw).strip()
    if not s:
        return []
    # Many blast outputs use 'accession_tax1;tax2;...'. Take the part after the first underscore.
    if "_" in s:
        s = s.split("_", 1)[1]
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    return parts


def _load_tax_filter_map() -> tuple[dict, str | None]:
    """Build mapping dict[level][filter_rank][taxon] -> set(values)."""
    from collections import defaultdict

    path = _find_asv_blast_file()
    if not path:
        return {}, None

    # nested default dicts
    m = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # hierarchy order (top -> bottom)
    rank_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    idx = {r: i for i, r in enumerate(rank_order)}

    # streaming parse (fast + memory safe)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue

            asv_raw = parts[0]
            tax_raw = parts[1]

            asv = _normalize_asv_id(asv_raw)
            tax_parts = _parse_taxonomy_path(tax_raw)

            # pad to expected length
            vals = {r: None for r in rank_order}
            for i, r in enumerate(rank_order):
                if i < len(tax_parts):
                    vals[r] = tax_parts[i]

            # Build current-level to higher-rank mappings
            # Current level = ASV (special)
            for fr in rank_order:
                if vals.get(fr):
                    m["ASV"][fr][asv].add(vals[fr])

            # Current levels among normal ranks
            for cur in rank_order:
                cur_val = vals.get(cur)
                if not cur_val:
                    continue
                # map cur -> any higher rank above cur
                for fr in rank_order:
                    if idx[fr] < idx[cur] and vals.get(fr):
                        m[cur][fr][cur_val].add(vals[fr])

    # cast to plain dicts
    def freeze(d):
        if isinstance(d, dict):
            return {k: freeze(v) for k, v in d.items()}
        return d

    return freeze(m), path


def _tax_filter_allowed_ranks(current_level: str) -> list[str]:
    """Return higher ranks that can filter the current level."""
    order = ["kingdom", "phylum", "class", "order", "family", "genus", "species", "ASV"]
    if current_level not in order:
        return ["phylum", "class", "order", "family", "genus"]
    cur_i = order.index(current_level)
    return [r for r in order[:cur_i] if r != "species"]  # hide species by default (often messy)


def _ensure_tax_filter_map_loaded() -> bool:
    global _TAX_FILTER_MAP, _TAX_FILTER_SRC
    if _TAX_FILTER_MAP is not None:
        return bool(_TAX_FILTER_MAP)
    try:
        _TAX_FILTER_MAP, _TAX_FILTER_SRC = _load_tax_filter_map()
    except Exception:
        _TAX_FILTER_MAP, _TAX_FILTER_SRC = {}, None
    return bool(_TAX_FILTER_MAP)


def _taxon_belongs_to_filter(level: str, taxon: str, filter_rank: str, keep_values: set[str]) -> bool:
    """True if taxon (at 'level') belongs to any of keep_values at filter_rank."""
    if not keep_values:
        return True
    if taxon == "Others":
        return True
    if not _ensure_tax_filter_map_loaded():
        return True

    lvl = str(level)
    fr = str(filter_rank)
    t = str(taxon)

    if lvl == "ASV":
        t = _normalize_asv_id(t)

    try:
        vals = _TAX_FILTER_MAP.get(lvl, {}).get(fr, {}).get(t, None)
    except Exception:
        vals = None
    if not vals:
        return False
    return any(v in keep_values for v in vals)


def _apply_tax_filter_to_index(level: str, taxa_index: list[str]) -> list[str]:
    """Filter a list of taxa labels for the current level."""
    if not tax_filter_enable.value:
        return taxa_index
    keep = set(map(str, tax_filter_values.value or []))
    if not keep:
        return taxa_index

    fr = str(tax_filter_rank.value)
    out = [t for t in taxa_index if _taxon_belongs_to_filter(level, t, fr, keep)]
    return out


def _apply_tax_filter_to_rel(level: str, rel_df: pd.DataFrame) -> pd.DataFrame:
    """Return rel_df filtered by taxonomy filter (rows)."""
    if rel_df is None or rel_df.empty:
        return rel_df
    idx = list(map(str, rel_df.index))
    filt = _apply_tax_filter_to_index(level, idx)
    if not filt:
        return rel_df
    try:
        return rel_df.loc[filt]
    except Exception:
        return rel_df


def _update_tax_filter_rank_options(event=None):
    # Adjust available filter ranks based on current selected level
    ranks = _tax_filter_allowed_ranks(str(level_w.value))
    if not ranks:
        ranks = ["phylum"]
    tax_filter_rank.options = ranks
    if tax_filter_rank.value not in ranks:
        tax_filter_rank.value = ranks[0]


def _update_tax_filter_values_options(event=None):
    # Populate the 'keep_values' list from taxa present in current REL subset
    if not data_loaded.value:
        tax_filter_values.options = []
        tax_filter_note.object = ""
        return
    if not _ensure_REL_loaded():
        return
    if REL is None or REL.empty:
        tax_filter_values.options = []
        tax_filter_note.object = ""
        return

    if not _ensure_tax_filter_map_loaded():
        tax_filter_values.options = []
        tax_filter_note.object = (
            "⚠️ **ASV_blast.txt not found** (taxonomy filter disabled). "
            "Place it in the working dir or in one of the configured candidate paths."
        )
        return

    lvl = str(level_w.value)
    fr = str(tax_filter_rank.value)

    present = set()
    for t in map(str, REL.index):
        if t == "Others":
            continue
        key = _normalize_asv_id(t) if lvl == "ASV" else t
        try:
            vals = _TAX_FILTER_MAP.get(lvl, {}).get(fr, {}).get(key, None)
        except Exception:
            vals = None
        if vals:
            for v in vals:
                if v and v != "nan":
                    present.add(str(v))

    opts = sorted(present)
    tax_filter_values.options = opts

    # Apply any pending preset values first, then sanitize selection vs options
    global _PENDING_TAX_FILTER_VALUES
    desired = list(_PENDING_TAX_FILTER_VALUES) if _PENDING_TAX_FILTER_VALUES else list(tax_filter_values.value or [])
    cur = [v for v in desired if v in opts]
    try:
        tax_filter_values.value = cur
    except Exception:
        # MultiChoice can be strict; set only valid values
        tax_filter_values.value = [v for v in cur if v in opts]
    _PENDING_TAX_FILTER_VALUES = []

    tax_filter_note.object = f"Using taxonomy mapping from: `{_TAX_FILTER_SRC}`" if _TAX_FILTER_SRC else ""



def _refresh_manual_taxa_options(event=None):
    """Rebuild the manual_taxa widget options from REL, with optional taxonomy filter applied.

    Sorting modes:
      - Alphabetical
      - Total abundance (current selection): sums abundance across samples in CURRENT_REL_FILE / current REL
      - Total abundance (all data): sums abundance across samples in the *root* rel_abundance_<level>.csv
        (or best-available 'all data' file for that level). Uses chunked reading + caching.
    """
    if REL is None:
        return

    lvl = str(level_w.value)
    base = list(map(str, REL.index))
    opts = _apply_tax_filter_to_index(lvl, base)
    if not opts:
        opts = base

    # --- sort manual taxa list options (independent of plot Y-order) ---
    sort_mode = str(getattr(manual_taxa_list_sort, "value", "Alphabetical"))

    def _sort_alpha(items):
        return sorted(items, key=lambda x: str(x).lower())

    def _totals_from_current(items):
        # REL values are relative abundances per sample; sum across samples for a "total in current selection"
        try:
            sub = REL.loc[items]
        except Exception:
            sub = REL.reindex(items)
        s = sub.sum(axis=1, numeric_only=True)
        return {str(k): float(v) for k, v in s.items() if pd.notna(v)}

    def _best_all_data_rel_file(level: str) -> str | None:
        root = _level_root(level)
        # Prefer the root-level rel file if present
        cand = [
            os.path.join(root, f"rel_abundance_{level}.csv"),
            os.path.join(root, f"saved_matrices_{level}", f"rel_abundance_{level}.csv"),
        ]
        for c in cand:
            if os.path.exists(c):
                return c
        # Fallback to resolver (may return a directory with a rel file)
        try:
            d = _resolve_dir_for_rel(level, None, None)
            c = os.path.join(d, f"rel_abundance_{level}.csv")
            if os.path.exists(c):
                return c
        except Exception:
            pass
        return None

    def _taxa_hash(items):
        h = hashlib.sha1()
        for t in sorted(map(str, items)):
            h.update(t.encode("utf-8", "ignore"))
            h.update(b"\n")
        return h.hexdigest()

    def _totals_from_all_data(items):
        rel_file = _best_all_data_rel_file(lvl)
        if not rel_file:
            return {}

        try:
            mtime = os.path.getmtime(rel_file)
        except Exception:
            mtime = 0.0

        th = _taxa_hash(items)
        key = (rel_file, mtime, th)
        if key in _GLOBAL_TOTALS_CACHE:
            return _GLOBAL_TOTALS_CACHE[key]

        # Chunked reading to avoid loading huge files (ASV can be very large)
        totals = {}
        taxa_set = set(map(str, items))
        try:
            status_spinner.value = True
            status_text.object = f"Sorting manual taxa list by total abundance (all data): reading `{rel_file}` …"
            for chunk in pd.read_csv(rel_file, index_col=0, chunksize=5000, low_memory=False):
                # chunk index are taxa
                idx = chunk.index.astype(str)
                keep_mask = idx.isin(taxa_set)
                if not keep_mask.any():
                    continue
                sub = chunk.loc[keep_mask]
                s = sub.sum(axis=1, numeric_only=True)
                for k, v in s.items():
                    if pd.isna(v):
                        continue
                    totals[str(k)] = float(v)
        except Exception:
            totals = {}
        finally:
            status_spinner.value = False
            # Don't overwrite status_text if the app is using it for other messages
            if status_text.object.startswith("Sorting manual taxa list by total abundance (all data):"):
                status_text.object = ""

        # Cache (only if non-empty)
        if totals:
            _GLOBAL_TOTALS_CACHE[key] = totals
        return totals

    if sort_mode == "Alphabetical":
        opts = _sort_alpha(opts)
    elif sort_mode == "Total abundance (current selection)":
        tot = _totals_from_current(opts)
        opts = sorted(opts, key=lambda t: tot.get(str(t), -1e30), reverse=True)
    elif sort_mode == "Total abundance (all data)":
        tot = _totals_from_all_data(opts)
        if tot:
            opts = sorted(opts, key=lambda t: tot.get(str(t), -1e30), reverse=True)
        else:
            # fallback
            opts = _sort_alpha(opts)
    else:
        opts = _sort_alpha(opts)

    manual_taxa.options = opts

    # keep current selection if possible
    cur = [t for t in (manual_taxa.value or []) if t in opts]
    if list(manual_taxa.value or []) != cur:
        manual_taxa.value = cur


def _level_root(level: str) -> str:
    return f"{BASE_OUTPUT_PREFIX}_{level}"


def _discover_subfolders(level: str):
    root = _level_root(level)
    niches, plants = set(), set()
    if not os.path.isdir(root):
        return [], []
    for entry in os.scandir(root):
        if entry.is_dir():
            name = entry.name
            if name.startswith("plant_"):
                plants.add(name.replace("plant_", "", 1))
            else:
                niches.add(name)

    for niche_dir in glob.glob(os.path.join(root, "*/")):
        nname = os.path.basename(os.path.normpath(niche_dir))
        if nname and not nname.startswith("plant_"):
            niches.add(nname)
        for pdir in glob.glob(os.path.join(niche_dir, "plant_*")):
            plants.add(os.path.basename(pdir).replace("plant_", "", 1))

    for pdir in glob.glob(os.path.join(root, "plant_*")):
        plants.add(os.path.basename(pdir).replace("plant_", "", 1))
        for nd in glob.glob(os.path.join(pdir, "*/")):
            nname = os.path.basename(os.path.normpath(nd))
            if nname and not nname.startswith("plant_"):
                niches.add(nname)

    return sorted(niches), sorted(plants)


def _resolve_dir_for_rel(level: str, niche: str | None, plant: str | None):
    root = _level_root(level)

    def ok(d):
        return os.path.isdir(d) and os.path.exists(os.path.join(d, f"rel_abundance_{level}.csv"))

    cand = []
    if niche and plant:
        cand += [
            os.path.join(root, niche, f"plant_{plant}"),
            os.path.join(root, f"plant_{plant}", niche),
        ]
    if plant:
        cand.append(os.path.join(root, f"plant_{plant}"))
    if niche:
        cand.append(os.path.join(root, niche))
    cand.append(root)
    for c in cand:
        if ok(c):
            return c
    return root


def _sel_or_none(label, all_label):
    return None if label == all_label else label


# ==============================
# Widgets & status
# ==============================
level_w = pn.widgets.Select(name="Taxonomy Level", options=LEVELS, value="phylum")
niche_w = pn.widgets.Select(name="Niche", options=[NICHE_ALL], value=NICHE_ALL)
plant_w = pn.widgets.Select(name="Plant species", options=[PLANT_ALL], value=PLANT_ALL)

load_btn = pn.widgets.Button(name="Load data", button_type="primary")
plot_btn = pn.widgets.Button(name="Plot", button_type="primary")
export_btn = pn.widgets.Button(name="Export PNG", button_type="warning")
# --- Correlation widgets (for 2-taxa correlation) ---
corr_btn = pn.widgets.Button(name="Run correlation", button_type="success")
corr_result = pn.pane.Markdown("", sizing_mode="stretch_width")
show_corr_on_plot = pn.widgets.Checkbox(name="Show correlation on plot", value=False)

# --- ANOVA / post-hoc letters widgets ---
anova_btn = pn.widgets.Button(name="Run ANOVA / test", button_type="success")
anova_taxa = pn.widgets.MultiChoice(name="Taxa for ANOVA (if >1)", options=[], value=[])
anova_replicates = pn.widgets.Select(
    name="Replicates for ANOVA",
    options=["Runs (samples)", "Studies (study means)"],
    value="Studies (study means)",
)
anova_study_col = pn.widgets.Select(name="Study column", options=["(None)"], value="(None)")
anova_alpha = pn.widgets.FloatInput(name="Alpha", value=0.05, step=0.01, start=0.001, end=0.2)
anova_result = pn.pane.Markdown("", sizing_mode="stretch_width")
# Toggle display of post-hoc letters on the plot
show_anova_letters = pn.widgets.Checkbox(name="Show post-hoc letters on plot", value=True)

# ==============================
# PCoA widgets
# ==============================
pcoa_btn = pn.widgets.Button(name="Run PCoA", button_type="success")
pcoa_status = pn.pane.Markdown("", sizing_mode="stretch_width")

pcoa_transform = pn.widgets.Select(
    name="Transform (standardize + sqrt)",
    options=["Proportions → sqrt (Hellinger-like)"],
    value="Proportions → sqrt (Hellinger-like)",
)

pcoa_color_by = pn.widgets.Select(
    name="Color by (metadata)",
    options=["(None)"] + list(metadata.columns),
    value="(None)",
)
pcoa_color_by.visible = False  # driven by taxonomy selection

pcoa_shape_by = pn.widgets.Select(
    name="Shape by (metadata)",
    options=["(None)"] + list(metadata.columns),
    value="(None)",
)
pcoa_shape_by.visible = False  # driven by taxonomy selection

pcoa_point_size = pn.widgets.IntSlider(
    name="Point size", start=10, end=200, value=60, step=5
)

pcoa_legend_text = pn.widgets.IntSlider(
    name="Legend text size (pt)", start=6, end=28, value=10, step=1
)

pcoa_legend_markerscale = pn.widgets.FloatSlider(
    name="Legend icon scale", start=0.4, end=3.0, value=1.0, step=0.1
)


# ==============================
# Diversity / differences (Step 5)
# ==============================
div_btn = pn.widgets.Button(name="Run diversity summary", button_type="success")
div_status = pn.pane.Markdown("", sizing_mode="stretch_width")
div_summary_md = pn.pane.Markdown("", sizing_mode="stretch_width")
div_table_groups = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width", height=260)
div_table_pairs = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width", height=320)

# ==============================
# SIMPER (Step 6)
# ==============================
simper_factor = pn.widgets.Select(
    name="SIMPER factor (metadata)",
    options=["(None)"] + list(metadata.columns),
    value="(None)",
)

# NEW: Final condition pool (allows excluding conditions not to be included in A or B)
simper_condition_pool = pn.widgets.MultiChoice(
    name="Conditions to consider (remove those you don't want in A or B)",
    options=[],
    value=[],
)

simper_groupA = pn.widgets.MultiChoice(
    name="Group A (select condition(s))",
    options=[],
    value=[],
)

simper_groupB = pn.widgets.MultiChoice(
    name="Group B (select condition(s))",
    options=[],
    value=[],
)

simper_top_n = pn.widgets.IntInput(
    name="Top taxa to show", value=30, start=5, end=500
)

simper_save_csv = pn.widgets.Checkbox(
    name="Save SIMPER CSV to disk", value=True
)

simper_run_btn = pn.widgets.Button(
    name="Run SIMPER", button_type="primary"
)

simper_status = pn.pane.Markdown("", sizing_mode="stretch_width")
simper_table = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width", height=520)
simper_csv_note = pn.pane.Markdown("", sizing_mode="stretch_width")

# ==============================
# ANOSIM (Step 7)
# ==============================
anosim_btn = pn.widgets.Button(name="Run ANOSIM", button_type="primary")
anosim_perms = pn.widgets.IntInput(name="Permutations", value=199, start=0, end=9999, step=1)
anosim_seed = pn.widgets.IntInput(name="Random seed", value=1, start=0, end=999999)
anosim_save_csv = pn.widgets.Checkbox(name="Save ANOSIM CSV to disk", value=True)

anosim_status = pn.pane.Markdown("", sizing_mode="stretch_width")
anosim_result_md = pn.pane.Markdown("", sizing_mode="stretch_width")
anosim_groups_df = pn.pane.DataFrame(pd.DataFrame(), sizing_mode="stretch_width", height=260)
anosim_csv_note = pn.pane.Markdown("", sizing_mode="stretch_width")



pcoa_alpha = pn.widgets.FloatSlider(
    name="Point alpha", start=0.1, end=1.0, value=0.9, step=0.05
)

pcoa_default_color = pn.widgets.ColorPicker(
    name="Default point color (when no color-by)",
    value="#1f77b4",
)

pcoa_show_labels = pn.widgets.Checkbox(
    name="Show sample IDs on plot", value=False
)
# --- Plot style widgets ---
remove_spines = pn.widgets.Checkbox(name="Remove top/right plot border", value=True)
# --- NEW: Lite mode ---------------------------------------------------------
lite_mode = pn.widgets.Checkbox(
    name="Lite mode (ASV: keep top 100 taxa)",
    value=False,
)
# ---------------------------------------------------------------------------

data_loaded = pn.widgets.Checkbox(value=False, visible=False)
status_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)
status_text = pn.pane.Markdown("", sizing_mode="stretch_width")
path_info = pn.pane.Markdown("", sizing_mode="stretch_width")

plot_mode = pn.widgets.RadioButtonGroup(
    name="Plot mode",
    options=["Per-sample stacked", "Per-group average"],
    value="Per-sample stacked",
)

subset_mode = pn.widgets.RadioButtonGroup(
    name="Taxa selection",
    options=["Top-N", "Manual subset"],
    value="Top-N",
)

TopN = pn.widgets.IntInput(name="Top N taxa", value=10, start=1, end=50)

# NEW: scope for Top-N
topN_scope = pn.widgets.RadioButtonGroup(
    name="Top-N scope",
    options=["Global", "Filtered selection"],
    value="Global",
)

manual_taxa = pn.widgets.MultiSelect(name="Select taxa (manual)", options=[], size=12)

manual_taxa_list_sort = pn.widgets.Select(
    name="Manual taxa list sort",
    options=["Alphabetical", "Total abundance (current selection)", "Total abundance (all data)"],
    value="Alphabetical",
)

# --- NEW: taxonomy-based filter to reduce taxa list (uses ASV_blast.txt) ---
tax_filter_enable = pn.widgets.Checkbox(
    name="Filter taxa by higher taxonomy (ASV_blast.txt)", value=False
)
tax_filter_rank = pn.widgets.Select(
    name="Filter rank", options=["phylum", "class", "order", "family", "genus"], value="phylum"
)
tax_filter_values = pn.widgets.MultiChoice(
    name="Keep only taxa belonging to", options=[], value=[]
)
tax_filter_note = pn.pane.Markdown("", sizing_mode="stretch_width")
# --------------------------------------------------------------------------
include_others = pn.widgets.Checkbox(name="Add 'Others' bin (Top-N)", value=True)

group_factor = pn.widgets.Select(
    name="Group / x-axis factor (averages)",
    options=["(None)"] + list(metadata.columns),
    value="(None)",
)
outer_group_factor = pn.widgets.Select(
    name="Outer group (top x-axis)",
    options=["(None)"] + list(metadata.columns),
    value="(None)",
)
group_gap = pn.widgets.IntSlider(
    name="Gap between outer groups (bars)", start=0, end=20, value=4
)

group_bin_other = pn.widgets.MultiSelect(
    name="Bin these groups into 'Other'",
    options=[],
    size=8,
)

# --- group filter widgets (based on group_factor / x-axis groups) ---
group_filter_mode = pn.widgets.RadioButtonGroup(
    name="Group filter mode",
    options=["All", "Include only", "Exclude"],
    value="All",
)
group_filter_select = pn.widgets.MultiSelect(
    name="Groups to include/exclude",
    options=[],
    size=8,
)
# Which factor drives X-grouping / manual X domain
x_group_source = pn.widgets.RadioButtonGroup(
    name="X-group source",
    options=["Samples", "Group / x-axis factor", "Outer group"],
    value="Group / x-axis factor",
)

# Optional second filter on the outer group factor
outer_filter_mode = pn.widgets.RadioButtonGroup(
    name="Outer group filter mode",
    options=["All", "Include only", "Exclude"],
    value="All",
)
outer_filter_select = pn.widgets.MultiSelect(
    name="Outer groups to include/exclude",
    options=[],
    size=8,
)

x_order_mode = pn.widgets.RadioButtonGroup(
    name="X-axis order", options=["Auto", "Manual"], value="Auto"
)

# NEW: how groups (not samples) are ordered when X-axis order = Auto
x_group_order_mode = pn.widgets.RadioButtonGroup(
    name="X group order (Auto)",
    options=["Data order", "Alphabetical"],
    value="Data order",
)

manual_x_available = pn.widgets.MultiSelect(
    name="Available groups/samples", options=[], size=10
)
manual_x_selected = pn.widgets.MultiSelect(
    name="Manual X order", options=[], size=10
)
btn_x_add = pn.widgets.Button(name="▶ Add", button_type="primary")
btn_x_remove = pn.widgets.Button(name="◀ Remove", button_type="default")
btn_x_up = pn.widgets.Button(name="↑ Up", button_type="success")
btn_x_down = pn.widgets.Button(name="↓ Down", button_type="success")
btn_x_top = pn.widgets.Button(name="⇞ Top", button_type="warning")
btn_x_bottom = pn.widgets.Button(name="⇟ Bottom", button_type="warning")
btn_x_clear = pn.widgets.Button(name="Clear", button_type="danger")

taxa_order_mode = pn.widgets.RadioButtonGroup(
    name="Taxa order (Y)",
    options=["Alphabetical", "Abundance", "Manual"],
    value="Alphabetical",
)
manual_taxa_available = pn.widgets.MultiSelect(
    name="Available taxa", options=[], size=12
)
manual_taxa_selected = pn.widgets.MultiSelect(
    name="Manual taxa order", options=[], size=12
)
btn_t_add = pn.widgets.Button(name="▶ Add", button_type="primary")
btn_t_remove = pn.widgets.Button(name="◀ Remove", button_type="default")
btn_t_up = pn.widgets.Button(name="↑ Up", button_type="success")
btn_t_down = pn.widgets.Button(name="↓ Down", button_type="success")
btn_t_top = pn.widgets.Button(name="⇞ Top", button_type="warning")
btn_t_bottom = pn.widgets.Button(name="⇟ Bottom", button_type="warning")
btn_t_clear = pn.widgets.Button(name="Clear", button_type="danger")

sort_taxon = pn.widgets.Select(
    name="Sort by taxon (samples)", options=["(None)"], value="(None)"
)
sort_scope = pn.widgets.RadioButtonGroup(
    name="Sort scope", options=["Global", "Within groups"], value="Global"
)
sort_desc = pn.widgets.Checkbox(name="Descending", value=True)

# --- NEW: n and SE annotations (Per-group average) -------------------------
show_n_per_bar = pn.widgets.Checkbox(name="Show n (number of samples) per bar", value=False)
n_display_mode = pn.widgets.Select(
    name="n display",
    options=["Append to label", "Above bars"],
    value="Append to label",
)
n_count_basis = pn.widgets.Select(
    name="n counts",
    options=["Run (samples)", "source_folder (studies)"],
    value="Run (samples)",
)
n_show_both = pn.widgets.Checkbox(name="Also show studies count (n samples, studies)", value=False)
show_se = pn.widgets.Checkbox(
    name="Show SE (only when 1 taxon selected, average mode)",
    value=False,
)
# ---------------------------------------------------------------------------

show_legend = pn.widgets.Checkbox(name="Show Legend", value=True)
legend_text_size = pn.widgets.IntSlider(
    name="Legend Text Size (pt)", start=6, end=28, step=1, value=10
)

legend_font_family = pn.widgets.Select(
    name="Legend Font Family",
    options=["DejaVu Sans", "Arial", "sans-serif", "serif", "monospace"],
    value="DejaVu Sans",
)

# NEW: legend style preset
legend_style_preset = pn.widgets.Select(
    name="Legend preset",
    options=["Custom", "Compact", "Standard", "Presentation", "Poster"],
    value="Standard",
)

x_label_rotation = pn.widgets.IntSlider(
    name="X-label rotation (°)", start=0, end=90, step=5, value=45
)
outer_x_label_rotation = pn.widgets.IntSlider(
    name="Outer X-label rotation (°)", start=0, end=90, step=5, value=0
)
x_label_fontsize = pn.widgets.IntSlider(
    name="X-label font size", start=6, end=24, step=1, value=9
)
y_label_fontsize = pn.widgets.IntSlider(
    name="Y-label font size", start=6, end=24, step=1, value=10
)

# NEW: abundance-axis limit controls (Y for normal bars, X when flip_axes is ON)
y_axis_mode = pn.widgets.Select(
    name="Abundance axis limit",
    options=["Auto", "Manual"],
    value="Auto",
)

y_axis_max = pn.widgets.FloatInput(
    name="Max abundance (%)",
    value=100.0,
    step=1.0,
)

def _sync_y_axis_widgets(event=None):
    try:
        y_axis_max.disabled = (y_axis_mode.value != "Manual")
    except Exception:
        pass

_sync_y_axis_widgets()
y_axis_mode.param.watch(_sync_y_axis_widgets, "value")

figure_width = pn.widgets.FloatSlider(
    name="Figure width (inches)", start=4, end=24, step=0.5, value=14
)
figure_height = pn.widgets.FloatSlider(
    name="Figure height (inches)", start=3, end=16, step=0.5, value=6
)

plot_orientation = pn.widgets.RadioButtonGroup(
    name="Plot orientation",
    options=["Horizontal / wide", "Vertical / tall"],
    value="Horizontal / wide",
)

flip_axes = pn.widgets.Checkbox(
    name="Flip axes (X ↔ Y, horizontal bars)",
    value=False,
)

# show sample IDs on X (per-sample mode)
show_sample_labels = pn.widgets.Checkbox(
    name="Show sample IDs on X (per-sample mode)", value=False
)

# NEW: group boundary controls
show_group_boundaries = pn.widgets.Checkbox(
    name="Show group boundaries between groups", value=True
)

group_boundary_width = pn.widgets.IntSlider(
    name="Group boundary line width",
    start=0, end=10, step=1, value=2,
)

group_boundary_color = pn.widgets.ColorPicker(
    name="Group boundary color",
    value="#000000",  # black
)

# NEW: which factor to use for boundaries
group_boundary_source = pn.widgets.RadioButtonGroup(
    name="Boundary grouping",
    options=["Group / x-axis factor", "Outer group"],
    value="Group / x-axis factor",
)


apply_style_btn = pn.widgets.Button(
    name="Apply style (no replot)", button_type="success"
)

hide_menus = pn.widgets.Toggle(
    name="Hide all menus", value=False, button_type="primary"
)
export_mode = pn.widgets.Toggle(
    name="Export / Clean canvas", value=False, button_type="danger"
)
export_size = pn.widgets.IntSlider(
    name="Export square size (px)", start=600, end=2000, step=50, value=1200
)

# --- Preset save/load widgets ----------------------------------
preset_name_input = pn.widgets.TextInput(
    name="Preset name", placeholder="e.g. andrzej_phylum_leaf", value="default"
)
preset_select = pn.widgets.Select(
    name="Available presets",
    options=[],
    value=None,
)
save_preset_btn = pn.widgets.Button(name="Save preset", button_type="success")
load_preset_btn = pn.widgets.Button(name="Load preset", button_type="primary")
preset_status = pn.pane.Markdown("", sizing_mode="stretch_width")
# ---------------------------------------------------------------

# ==============================
# State & cache
# ==============================
CURRENT_DIR = ""
CURRENT_REL_FILE = ""
REL = None

# Cache for total-abundance sorting (manual taxa list)
# key: (rel_file, mtime, taxa_hash) -> dict[taxon, total]
_GLOBAL_TOTALS_CACHE = {}

CACHE_DIR = "./cache_tax_bars"
os.makedirs(CACHE_DIR, exist_ok=True)
SETTINGS_DIR = "./ui_presets"
os.makedirs(SETTINGS_DIR, exist_ok=True)

last_png_path = {"path": None}
_last_fig = {"fig": None, "ax": None}

# --- caches for on-plot annotations ---
ANOVA_LETTERS = {}  # (taxon, group_factor) -> {group_value: letters}
PLOT_STATE = {"corr_annot": None}   # persistent correlation annotation text


# ==============================
# PCoA state (derived from current taxonomy selection / filters)
# ==============================
# Holds the last per-sample taxa matrix used as input for ordination (taxa x samples),
# plus helpful labels for plotting.
_LAST_PCOA_INPUT = {"matrix": None, "samples": None}
manual_x_sequence: list[str] = []
_manual_x_domain: list[str] = []

manual_taxa_sequence: list[str] = []
_taxa_domain: list[str] = []


# ==============================
# Stats for n / SE (average mode)
# ==============================
_LAST_GROUP_TO_SAMPLES: dict[str, list[str]] = {}
_LAST_PREAVG_MATRIX: Optional[pd.DataFrame] = None
# =============================================================================


# ==============================
# Legend preset behavior
# ==============================
def _apply_legend_preset(event):
    preset = legend_style_preset.value
    if preset == "Compact":
        legend_text_size.value = 8
        legend_font_family.value = "DejaVu Sans"
    elif preset == "Standard":
        legend_text_size.value = 10
        legend_font_family.value = "DejaVu Sans"
    elif preset == "Presentation":
        legend_text_size.value = 14
        legend_font_family.value = "Arial"
    elif preset == "Poster":
        legend_text_size.value = 18
        legend_font_family.value = "Arial"
    # "Custom" → do nothing; user tweaks manually


legend_style_preset.param.watch(_apply_legend_preset, "value")


# ==============================
# Wiring: data choices
# ==============================
def _update_data_source_choices(event=None):
    level = level_w.value
    niches, plants = _discover_subfolders(level)
    niche_w.options = [NICHE_ALL] + niches
    plant_w.options = [PLANT_ALL] + plants
    if niche_w.value not in niche_w.options:
        niche_w.value = NICHE_ALL
    if plant_w.value not in plant_w.options:
        plant_w.value = PLANT_ALL
    _update_path_info()
    try:
        _update_tax_filter_rank_options()
    except Exception:
        pass


def _update_path_info(event=None):
    global CURRENT_DIR, CURRENT_REL_FILE, REL
    level = level_w.value
    niche = _sel_or_none(niche_w.value, NICHE_ALL)
    plant = _sel_or_none(plant_w.value, PLANT_ALL)
    d = _resolve_dir_for_rel(level, niche, plant)
    rel_file = os.path.join(d, f"rel_abundance_{level}.csv")
    CURRENT_DIR = d
    CURRENT_REL_FILE = rel_file if os.path.exists(rel_file) else ""
    REL = None
    data_loaded.value = False
    status_text.object = ""
    status_spinner.value = False
    subset = [f"niche={niche or 'ALL'}", f"plant={plant or 'ALL'}"]
    if CURRENT_REL_FILE:
        info = f"Using: {CURRENT_REL_FILE} • ({', '.join(subset)})"
        if level_w.value == "ASV" and lite_mode.value:
            info += " • **Lite mode ON (top 100 taxa + Others)**"
        path_info.object = info
    else:
        path_info.object = f"⚠️ rel_abundance file not found • ({', '.join(subset)})"


level_w.param.watch(_update_data_source_choices, "value")
niche_w.param.watch(_update_path_info, "value")
plant_w.param.watch(_update_path_info, "value")

# if lite_mode is toggled, force reload / reset
lite_mode.param.watch(lambda e: _update_path_info(), "value")

_update_data_source_choices()


def _do_load(event=None):
    data_loaded.value = True
    status_spinner.value = True
    status_text.object = "**Loading rel_abundance…**"
    try:
        _ensure_REL_loaded()  # also refreshes taxa options (and taxonomy filter widgets)
    except Exception as e:
        status_spinner.value = False
        data_loaded.value = False
        status_text.object = f"⚠️ Error loading rel_abundance: {e}"
        return

    label = os.path.basename(CURRENT_REL_FILE) if CURRENT_REL_FILE else "subset"
    suffix = " (lite: top 100 + Others)" if level_w.value == "ASV" and lite_mode.value else ""
    status_text.object = f"**Data loaded:** {label}{suffix} — choose options and **Plot**."
    status_spinner.value = False

    try:
        _refresh_group_outer_filters()
    except Exception:
        pass


def _on_plot(event=None):
    status_spinner.value = True
    status_text.object = "**Plotting…**"


load_btn.on_click(_do_load)
plot_btn.on_click(_on_plot)

# Keep group/outer filter widgets populated immediately when user changes factors
try:
    group_factor.param.watch(_refresh_group_outer_filters, "value")
    outer_group_factor.param.watch(_refresh_group_outer_filters, "value")
    plot_mode.param.watch(_refresh_group_outer_filters, "value")
    group_bin_other.param.watch(_refresh_group_outer_filters, "value")
    data_loaded.param.watch(_refresh_group_outer_filters, "value")
except Exception:
    pass



# ==============================
# Wiring: taxonomy filter widgets
# ==============================
def _on_tax_filter_change(event=None):
    # Only meaningful once data is loaded
    if not data_loaded.value:
        return
    try:
        _ensure_REL_loaded()
        _update_tax_filter_values_options()
        _refresh_manual_taxa_options()
    except Exception as e:
        try:
            tax_filter_note.object = f"⚠️ Taxonomy filter error: {e}"
        except Exception:
            pass

tax_filter_enable.param.watch(_on_tax_filter_change, "value")
tax_filter_rank.param.watch(_on_tax_filter_change, "value")
tax_filter_values.param.watch(_on_tax_filter_change, "value")
manual_taxa_list_sort.param.watch(lambda e: _refresh_manual_taxa_options(), "value")




def _toggle_anova_letters_replot(event=None):
    # If data is already loaded, let users hide/show cached letters without re-running ANOVA
    if data_loaded.value:
        try:
            plot_btn.clicks = int(plot_btn.clicks) + 1
        except Exception:
            pass


show_anova_letters.param.watch(_toggle_anova_letters_replot, "value")
remove_spines.param.watch(_toggle_anova_letters_replot, "value")


# ==============================
# IO helpers
# ==============================


def _format_n_label(n_samples: int, n_studies: int) -> str:
    """Format combined counts as: n=<samples>, studies=<studies>."""
    try:
        ns = int(n_samples)
    except Exception:
        ns = 0
    try:
        nd = int(n_studies)
    except Exception:
        nd = 0
    return f"n={ns}, studies={nd}"

def _safe_read_rel(rel_file: str) -> pd.DataFrame:
    df = pd.read_csv(rel_file)
    if df.shape[1] < 2:
        raise ValueError("rel_abundance file has no sample columns")

    df = df.copy()
    df.rename(columns={df.columns[0]: "taxon"}, inplace=True)
    df.set_index("taxon", inplace=True)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mx = float(df.to_numpy().max()) if not df.empty else 0.0
    if mx <= 1.001:
        df = df * 100.0

    df = df.astype("float32")

    if metadata is not None and not metadata.empty:
        valid_samples = [c for c in df.columns if c in metadata.index]
        if valid_samples:
            df = df[valid_samples]

    return df


# --- NEW: lite reader for ASV level ----------------------------------------
def _safe_read_rel_lite(rel_file: str, max_taxa: int = 100) -> pd.DataFrame:
    """
    Chunked 'lite' reader for huge ASV tables.

    1) First pass: compute total abundance per taxon across ALL samples,
       and global max value (to detect % vs [0–1]).
    2) Select top max_taxa taxa.
    3) Second pass: build matrix with those taxa + one 'Others' row that
       aggregates ALL remaining taxa (including tails beyond top 100).
    """
    taxa_totals: dict[str, float] = {}
    global_max = 0.0

    # First pass
    for chunk in pd.read_csv(rel_file, chunksize=2000):
        if chunk.shape[1] < 2:
            continue
        chunk = chunk.copy()
        first_col = chunk.columns[0]
        chunk.rename(columns={first_col: "taxon"}, inplace=True)

        vals = chunk.iloc[:, 1:]
        vals = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        if not vals.empty:
            mx = float(vals.to_numpy().max())
            if mx > global_max:
                global_max = mx

        sums = vals.sum(axis=1)
        for taxon, s in zip(chunk["taxon"], sums):
            t = str(taxon)
            taxa_totals[t] = taxa_totals.get(t, 0.0) + float(s)

    if not taxa_totals:
        raise ValueError("rel_abundance file has no data")

    totals_series = pd.Series(taxa_totals)
    top_taxa = totals_series.nlargest(max_taxa).index.tolist()

    # Second pass
    top_df_list = []
    others_sum = None

    for chunk in pd.read_csv(rel_file, chunksize=2000):
        if chunk.shape[1] < 2:
            continue
        chunk = chunk.copy()
        first_col = chunk.columns[0]
        chunk.rename(columns={first_col: "taxon"}, inplace=True)

        vals = chunk.iloc[:, 1:]
        vals = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        top_mask = chunk["taxon"].astype(str).isin(top_taxa)
        top_chunk = vals[top_mask]
        top_idx = chunk.loc[top_mask, "taxon"].astype(str)

        if not top_chunk.empty:
            top_chunk.index = top_idx
            top_df_list.append(top_chunk)

        other_chunk = vals[~top_mask]
        if not other_chunk.empty:
            s = other_chunk.sum(axis=0)
            if others_sum is None:
                others_sum = s
            else:
                others_sum = others_sum.add(s, fill_value=0.0)

    if top_df_list:
        top_df = pd.concat(top_df_list, axis=0)
        top_df = top_df.groupby(level=0).sum()
    else:
        top_df = pd.DataFrame()

    if others_sum is not None:
        others_df = pd.DataFrame([others_sum], index=["Others"])
        rel = pd.concat([top_df, others_df], axis=0)
    else:
        rel = top_df

    if global_max <= 1.001:
        rel = rel * 100.0

    rel = rel.astype("float32")

    if metadata is not None and not metadata.empty:
        valid_samples = [c for c in rel.columns if c in metadata.index]
        if valid_samples:
            rel = rel[valid_samples]

    return rel
# ---------------------------------------------------------------------------


def _ensure_REL_loaded() -> bool:
    global REL
    if not data_loaded.value:
        return False
    if REL is None:
        if not CURRENT_REL_FILE:
            raise SystemExit("Selected subset has no rel_abundance file.")
        # Use lite reader only for ASV level when toggle is ON
        if level_w.value == "ASV" and lite_mode.value:
            REL = _safe_read_rel_lite(CURRENT_REL_FILE, max_taxa=100)
            try:
                _ensure_metadata_for_samples(REL.columns)
                _refresh_metadata_dependent_widgets()
            except Exception:
                pass
        else:
            REL = _safe_read_rel(CURRENT_REL_FILE)
        try:
            _ensure_metadata_for_samples(REL.columns)
            _refresh_metadata_dependent_widgets()
        except Exception:
            pass
        # Update taxonomy-filter controls (if enabled) and populate taxa list
        try:
            _update_tax_filter_rank_options()
            _update_tax_filter_values_options()
        except Exception:
            pass
        _refresh_manual_taxa_options()
    return True



# ==============================
# Refresh group/outer filter widget options (no-plot)
# ==============================
def _unique_in_order(items):
    """Return unique stringified items preserving first-seen order."""
    out = []
    seen = set()
    for x in items:
        s = str(x) if x is not None else "NA"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _refresh_group_outer_filters(event=None):
    """Keep group/outer filter widgets populated as soon as factors change.

    Previously these options were only populated inside `render_plot()`, so they stayed empty
    until the user clicked Plot. This refresh is light-weight: it uses current REL columns
    (sample IDs) + metadata, without building the aggregated matrix.
    """
    try:
        if not data_loaded.value:
            group_bin_other.options = []
            group_filter_select.options = []
            outer_filter_select.options = []
            return
        if not _ensure_REL_loaded():
            return
        if REL is None or REL.empty:
            group_bin_other.options = []
            group_filter_select.options = []
            outer_filter_select.options = []
            return

        samples = [str(c) for c in REL.columns if (metadata is not None and not metadata.empty and c in metadata.index)]
        if not samples:
            # If metadata is empty/missing rows, ensure it exists for these samples
            try:
                _ensure_metadata_for_samples(list(REL.columns))
                samples = [str(c) for c in REL.columns if c in metadata.index]
            except Exception:
                samples = [str(c) for c in REL.columns]

        # ---- Primary group factor (group_factor) ----
        gf = str(group_factor.value)
        if gf != "(None)" and (metadata is not None) and (gf in metadata.columns) and samples:
            try:
                grp_labels = metadata.loc[samples, gf].astype(str).tolist()
            except Exception:
                grp_labels = []
            orig_labels = _unique_in_order(grp_labels)
        else:
            orig_labels = []

        # Bin-to-Other selector uses *original* labels (excluding 'Other' itself)
        group_bin_other.options = sorted({lab for lab in orig_labels if str(lab) != "Other"})
        # Sanitize current selection (MultiSelect can hold stale values)
        group_bin_other.value = [v for v in (group_bin_other.value or []) if v in group_bin_other.options]

        bin_set = set(map(str, group_bin_other.value or []))

        # Group filter options refer to group_factor *after* binning
        if gf != "(None)" and (metadata is not None) and (gf in metadata.columns) and samples:
            gf_labels_all, seen = [], set()
            for s in samples:
                lbl = _safe_group_label(s, gf, bin_set)
                if lbl not in seen:
                    seen.add(lbl)
                    gf_labels_all.append(lbl)
        else:
            # No grouping factor -> filter by sample IDs (rare but consistent)
            gf_labels_all = [str(s) for s in samples]

        group_filter_select.options = gf_labels_all
        group_filter_select.value = [v for v in (group_filter_select.value or []) if v in group_filter_select.options]

        # ---- Outer group factor (outer_group_factor) ----
        of = str(outer_group_factor.value)
        if (
            plot_mode.value == "Per-sample stacked"
            and of != "(None)"
            and (metadata is not None)
            and (of in metadata.columns)
            and samples
        ):
            try:
                of_labels = metadata.loc[samples, of].astype(str).tolist()
            except Exception:
                of_labels = []
            outer_labels_all = _unique_in_order(of_labels)
        else:
            outer_labels_all = []

        outer_filter_select.options = outer_labels_all
        outer_filter_select.value = [v for v in (outer_filter_select.value or []) if v in outer_filter_select.options]

    except Exception:
        # Never crash the app because of widget refresh
        return


# ==============================
# Manual list editor helpers (X)
# ==============================
def _refresh_manual_x_lists():
    chosen = [x for x in manual_x_sequence if x in _manual_x_domain]
    avail = [x for x in _manual_x_domain if x not in chosen]
    manual_x_selected.options = chosen
    manual_x_available.options = avail


def _set_manual_x_domain(domain):
    global _manual_x_domain, manual_x_sequence
    _manual_x_domain = list(dict.fromkeys(map(str, domain)))
    manual_x_sequence = [x for x in manual_x_sequence if x in _manual_x_domain]
    _refresh_manual_x_lists()


def _x_add():
    sel = set(manual_x_available.value)
    if not sel:
        return
    for x in _manual_x_domain:
        if x in sel and x not in manual_x_sequence:
            manual_x_sequence.append(x)
    _refresh_manual_x_lists()


def _x_remove():
    sel = set(manual_x_selected.value)
    if not sel:
        return
    manual_x_sequence[:] = [x for x in manual_x_sequence if x not in sel]
    _refresh_manual_x_lists()


def _x_up():
    sel = list(manual_x_selected.value)
    if not sel:
        return
    for s in sel:
        i = manual_x_sequence.index(s)
        if i > 0 and manual_x_sequence[i - 1] not in sel:
            manual_x_sequence[i - 1], manual_x_sequence[i] = (
                manual_x_sequence[i],
                manual_x_sequence[i - 1],
            )
    _refresh_manual_x_lists()


def _x_down():
    sel = list(reversed(list(manual_x_selected.value)))
    if not sel:
        return
    for s in sel:
        i = manual_x_sequence.index(s)
        if i < len(manual_x_sequence) - 1 and manual_x_sequence[i + 1] not in sel:
            manual_x_sequence[i + 1], manual_x_sequence[i] = (
                manual_x_sequence[i],
                manual_x_sequence[i + 1],
            )
    _refresh_manual_x_lists()


def _x_top():
    sel = list(manual_x_selected.value)
    rest = [x for x in manual_x_sequence if x not in sel]
    manual_x_sequence[:] = sel + rest
    _refresh_manual_x_lists()


def _x_bottom():
    sel = list(manual_x_selected.value)
    rest = [x for x in manual_x_sequence if x not in sel]
    manual_x_sequence[:] = rest + sel
    _refresh_manual_x_lists()


def _x_clear():
    manual_x_sequence.clear()
    _refresh_manual_x_lists()


btn_x_add.on_click(lambda e: _x_add())
btn_x_remove.on_click(lambda e: _x_remove())
btn_x_up.on_click(lambda e: _x_up())
btn_x_down.on_click(lambda e: _x_down())
btn_x_top.on_click(lambda e: _x_top())
btn_x_bottom.on_click(lambda e: _x_bottom())
btn_x_clear.on_click(lambda e: _x_clear())


# ==============================
# Manual list editor helpers (Taxa / Y)
# ==============================
def _refresh_taxa_lists():
    chosen = [t for t in manual_taxa_sequence if t in _taxa_domain]
    avail = [t for t in _taxa_domain if t not in chosen]
    manual_taxa_selected.options = chosen
    manual_taxa_available.options = avail


def _set_taxa_domain(domain):
    global _taxa_domain, manual_taxa_sequence
    _taxa_domain = list(dict.fromkeys(map(str, domain)))
    manual_taxa_sequence = [t for t in manual_taxa_sequence if t in _taxa_domain]
    _refresh_taxa_lists()


def _t_add():
    sel = set(manual_taxa_available.value)
    if not sel:
        return
    for t in _taxa_domain:
        if t in sel and t not in manual_taxa_sequence:
            manual_taxa_sequence.append(t)
    _refresh_taxa_lists()


def _t_remove():
    sel = set(manual_taxa_selected.value)
    if not sel:
        return
    manual_taxa_sequence[:] = [t for t in manual_taxa_sequence if t not in sel]
    _refresh_taxa_lists()


def _t_up():
    sel = list(manual_taxa_selected.value)
    if not sel:
        return
    for s in sel:
        i = manual_taxa_sequence.index(s)
        if i > 0 and manual_taxa_sequence[i - 1] not in sel:
            manual_taxa_sequence[i - 1], manual_taxa_sequence[i] = (
                manual_taxa_sequence[i],
                manual_taxa_sequence[i - 1],
            )
    _refresh_taxa_lists()


def _t_down():
    sel = list(reversed(list(manual_taxa_selected.value)))
    if not sel:
        return
    for s in sel:
        i = manual_taxa_sequence.index(s)
        if i < len(manual_taxa_sequence) - 1 and manual_taxa_sequence[i + 1] not in sel:
            manual_taxa_sequence[i + 1], manual_taxa_sequence[i] = (
                manual_taxa_sequence[i],
                manual_taxa_sequence[i + 1],
            )
    _refresh_taxa_lists()


def _t_top():
    sel = list(manual_taxa_selected.value)
    rest = [t for t in manual_taxa_sequence if t not in sel]
    manual_taxa_sequence[:] = sel + rest
    _refresh_taxa_lists()


def _t_bottom():
    sel = list(manual_taxa_selected.value)
    rest = [t for t in manual_taxa_sequence if t not in sel]
    manual_taxa_sequence[:] = rest + sel
    _refresh_taxa_lists()


def _t_clear():
    manual_taxa_sequence.clear()
    _refresh_taxa_lists()


btn_t_add.on_click(lambda e: _t_add())
btn_t_remove.on_click(lambda e: _t_remove())
btn_t_up.on_click(lambda e: _t_up())
btn_t_down.on_click(lambda e: _t_down())
btn_t_top.on_click(lambda e: _t_top())
btn_t_bottom.on_click(lambda e: _t_bottom())
btn_t_clear.on_click(lambda e: _t_clear())


# ==============================
# Group label helpers
# ==============================
def apply_group_binning(sample, factor):
    """Apply taxonomy binning (group_bin_other) consistently for PCoA."""
    if factor == "(None)" or factor not in metadata.columns:
        return None
    try:
        val = str(metadata.at[sample, factor])
    except Exception:
        return None

    if factor == group_factor.value:
        try:
            bin_set = set(map(str, group_bin_other.value))
            if val in bin_set:
                return "Other"
        except Exception:
            pass

    return val


def _safe_group_label(sample, factor_name, bin_set):
    if sample in metadata.index and factor_name in metadata.columns:
        lab = str(metadata.at[sample, factor_name])
    else:
        lab = "NA"
    if lab in bin_set:
        lab = "Other"
    return lab


# ==============================
# Build matrices & ordering
# ==============================
def _order_taxa(A: pd.DataFrame, legend_cats: list[str]) -> list[str]:
    if taxa_order_mode.value == "Manual":
        sel = [t for t in manual_taxa_sequence if t in A.index]
        remaining = [t for t in legend_cats if t not in sel]
        if "Others" in remaining:
            remaining = [t for t in remaining if t != "Others"] + ["Others"]
        return sel + remaining
    elif taxa_order_mode.value == "Alphabetical":
        ordered = sorted([t for t in legend_cats if t in A.index])
        if "Others" in ordered:
            ordered.remove("Others")
            ordered.append("Others")
        return ordered
    else:
        sums = A.sum(axis=1).to_dict()
        ordered = sorted(legend_cats, key=lambda t: (-sums.get(t, 0.0), str(t)))
        if "Others" in ordered:
            ordered.remove("Others")
            ordered.append("Others")
        return ordered


# NEW: columns used to compute Top-N taxa
def _columns_for_topN(rel: pd.DataFrame) -> list[str]:
    cols = list(rel.columns)
    # Global: all columns
    if topN_scope.value != "Filtered selection":
        return cols

    gf = group_factor.value
    if gf == "(None)" or gf not in metadata.columns:
        return cols

    bin_set = set(group_bin_other.value)
    selected_groups = set(map(str, group_filter_select.value))
    mode = group_filter_mode.value

    # If filter mode is "All" or no groups selected, effectively global
    if mode == "All" or not selected_groups:
        return cols

    keep_cols = []
    for c in cols:
        if c not in metadata.index:
            continue
        lab = _safe_group_label(c, gf, bin_set)
        if mode == "Include only":
            if lab in selected_groups:
                keep_cols.append(c)
        else:  # Exclude
            if lab not in selected_groups:
                keep_cols.append(c)

    # Fallback: if everything got filtered away, just use all
    return keep_cols or cols


def build_aggregated_matrix():
    if not _ensure_REL_loaded():
        return pd.DataFrame(), []
    rel = REL.copy()
    rel.index = rel.index.astype(str)

    # Optional: filter taxa by higher taxonomy (ASV_blast.txt)
    rel = _apply_tax_filter_to_rel(str(level_w.value), rel)

    # Optional: filter taxa by higher taxonomy (ASV_blast.txt)
    rel = _apply_tax_filter_to_rel(str(level_w.value), rel)

    if subset_mode.value == "Top-N":
        cols_for_topn = _columns_for_topN(rel)
        totals = rel[cols_for_topn].sum(axis=1)
        top = totals.nlargest(max(1, int(TopN.value))).index.tolist()
        A = rel.loc[top].copy()
        if include_others.value and len(rel.index) > len(top):
            others = rel.drop(index=top).sum(axis=0)
            A.loc["Others"] = others
    else:
        sel = [str(t) for t in manual_taxa.value if str(t) in rel.index]
        if not sel:
            sel = rel.sum(axis=1).nlargest(10).index.tolist()
        A = rel.loc[sel].copy()

    if plot_mode.value == "Per-group average" and group_factor.value != "(None)":
        gf = group_factor.value
        valid = [s for s in A.columns if s in metadata.index]
        if valid:
            # Save pre-aggregation (taxa x samples) so we can compute n and SE later
            global _LAST_PREAVG_MATRIX, _LAST_GROUP_TO_SAMPLES
            _LAST_PREAVG_MATRIX = A[valid].copy()
            meta_sub = metadata.loc[valid, [gf]].astype(str)
            _LAST_GROUP_TO_SAMPLES = {
                str(g): [str(s) for s in list(idx)]
                for g, idx in meta_sub.groupby(gf).groups.items()
            }

            groups = {g: A[list(idx)].mean(axis=1) for g, idx in meta_sub.groupby(gf).groups.items()}
            A = pd.DataFrame(groups)
        else:
            _LAST_PREAVG_MATRIX = None
            _LAST_GROUP_TO_SAMPLES = {}

    legend_cats = [str(t) for t in A.index]
    legend_cats = _order_taxa(A, legend_cats)
    return A, legend_cats


# ==============================
# X positions & grouping helpers
# ==============================
def _positions_preserve_order_with_outer_gap(
    ordered_cols, outer_factor=None, gap=0, bin_set=None
):
    if not outer_factor or outer_factor == "(None)":
        return list(range(len(ordered_cols))), []
    bin_set = bin_set or set()

    def lab(c):
        if c in metadata.index and outer_factor in metadata.columns:
            lbl = str(metadata.at[c, outer_factor])
        else:
            lbl = "NA"
        if lbl in bin_set:
            lbl = "Other"
        return lbl

    positions, spans, pos = [], [], 0
    if not ordered_cols:
        return [], []
    prev = lab(ordered_cols[0])
    start = 0
    for i, c in enumerate(ordered_cols):
        cur = lab(c)
        if i > 0 and cur != prev:
            spans.append((prev, positions[start], positions[-1]))
            pos += int(gap)
            start = i
        positions.append(pos)
        pos += 1
        prev = cur
    if positions:
        spans.append((lab(ordered_cols[-1]), positions[start], positions[-1]))
    return positions, spans


def _expand_groups_to_samples(A_cols, group_name_order, group_factor_name, bin_set=None):
    cols = list(A_cols)
    bin_set = bin_set or set()
    if not group_factor_name or group_factor_name == "(None)":
        return cols
    sample2grp = {c: _safe_group_label(c, group_factor_name, bin_set) for c in cols}
    ordered, seen = [], set()
    for g in group_name_order:
        for c in cols:
            if c not in seen and sample2grp.get(c, "NA") == str(g):
                ordered.append(c)
                seen.add(c)
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _auto_grouped_sample_order(
    A_cols,
    group_factor_name,
    bin_set=None,
    alphabetical: bool = False,
):
    cols = list(A_cols)
    bin_set = bin_set or set()
    if not group_factor_name or group_factor_name == "(None)":
        return cols

    seen, order = set(), []
    for c in cols:
        lab = _safe_group_label(c, group_factor_name, bin_set)
        if lab not in seen:
            seen.add(lab)
            order.append(lab)

    if alphabetical:
        order = sorted(order, key=str)

    return _expand_groups_to_samples(cols, order, group_factor_name, bin_set=bin_set)



def _compute_group_centers_for_labels(
    ordered_cols, pos, label_factor: str, bin_set=None
):
    if (
        not label_factor
        or label_factor == "(None)"
        or label_factor not in metadata.columns
    ):
        return None, None
    bin_set = bin_set or set()
    labs = [_safe_group_label(c, label_factor, bin_set) for c in ordered_cols]
    centers, labels, i = [], [], 0
    while i < len(ordered_cols):
        j = i
        lab = labs[i]
        while j + 1 < len(ordered_cols) and labs[j + 1] == lab:
            j += 1
        block = pos[i : j + 1]
        centers.append((block[0] + block[-1]) / 2.0)
        labels.append(lab)
        i = j + 1
    return centers, labels


def _draw_group_boundaries(ax, ordered_cols, pos, label_factor: str, bin_set=None):
    # New: allow turning boundaries off or making them invisible
    if (
        not show_group_boundaries.value
        or group_boundary_width.value <= 0
        or not group_boundary_color.value
    ):
        return

    if (
        not label_factor
        or label_factor == "(None)"
        or label_factor not in metadata.columns
    ):
        return
    if not ordered_cols:
        return

    bin_set = bin_set or set()
    labs = [_safe_group_label(c, label_factor, bin_set) for c in ordered_cols]

    boundaries = []
    prev_lab = labs[0]
    for i in range(1, len(ordered_cols)):
        cur_lab = labs[i]
        if cur_lab != prev_lab:
            boundary = (pos[i - 1] + pos[i]) / 2.0
            boundaries.append(boundary)
            prev_lab = cur_lab

    for b in boundaries:
        ax.axvline(
            b,
            color=group_boundary_color.value,
            linewidth=float(group_boundary_width.value),
            alpha=0.8,
        )


# ==============================
# Plotters
# ==============================
def _get_taxon_colors(legend_cats):
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = []
    if prop_cycle is not None:
        cycle_colors = prop_cycle.by_key().get("color", [])

    colors = {}
    non_other = [t for t in legend_cats if t != "Others"]
    for i, t in enumerate(non_other):
        if cycle_colors:
            colors[t] = cycle_colors[i % len(cycle_colors)]
        else:
            colors[t] = None
    colors["Others"] = "0.7"
    return colors


def _apply_common_style(ax):
    ax.tick_params(axis="y", labelsize=int(y_label_fontsize.value))
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(int(legend_text_size.value))
            t.set_fontfamily(str(legend_font_family.value))


# --- NEW: abundance-axis limit helpers -------------------------------------
def _manual_abundance_max():
    """Return user-chosen max (%) if enabled, else None."""
    if getattr(y_axis_mode, "value", "Auto") != "Manual":
        return None
    try:
        v = float(y_axis_max.value)
    except Exception:
        return None
    try:
        import numpy as _np
        if not _np.isfinite(v):
            return None
    except Exception:
        pass
    return v if v > 0 else None


def _apply_abundance_axis_limits(ax):
    """Apply manual axis limits to the abundance axis (Y normally, X when flipped)."""
    vmax = _manual_abundance_max()
    if vmax is None:
        return
    if flip_axes.value:
        ax.set_xlim(0, vmax)
    else:
        ax.set_ylim(0, vmax)


def _abundance_axis_top(ax):
    """Top value of the abundance axis (after limits are applied)."""
    return ax.get_xlim()[1] if flip_axes.value else ax.get_ylim()[1]
# ---------------------------------------------------------------------------


# --- NEW: n labels for Per-sample stacked (grouped ticks) -------------------
def _compute_n_maps_for_blocks(ordered_cols, label_factor, bin_set):
    """
    For per-sample stacked plots: ordered_cols are samples (Run IDs).
    We compute n per consecutive label block (group), so it matches the grouped x-ticks.
    Returns:
      block_centers_idxspace, block_labels, n_samples_per_block, n_studies_per_block
    """
    if not label_factor or label_factor == "(None)" or label_factor not in metadata.columns:
        return None, None, None, None

    labs = [_safe_group_label(c, label_factor, bin_set) for c in ordered_cols]

    block_centers = []
    block_labels = []
    n_samples = []
    n_studies = []

    i = 0
    while i < len(ordered_cols):
        j = i
        lab = labs[i]
        while j + 1 < len(ordered_cols) and labs[j + 1] == lab:
            j += 1

        block_samples = [str(s) for s in ordered_cols[i : j + 1]]
        block_labels.append(lab)
        block_centers.append((i + j) / 2.0)

        n_samples.append(len(block_samples))

        sf = []
        for r in block_samples:
            sfv = run2meta.get(r, {}).get("source_folder", None)
            if sfv is not None and str(sfv).strip() != "":
                sf.append(str(sfv))
        n_studies.append(len(set(sf)))

        i = j + 1

    return block_centers, block_labels, n_samples, n_studies
# ---------------------------------------------------------------------------
def plot_stacked_per_sample(A: pd.DataFrame, legend_cats: list):
    if A.empty:
        return pn.pane.Markdown("⚠️ No data to plot.")
    cols = list(A.columns)
    bin_set = set(group_bin_other.value)

    # apply group filter (per-sample mode) – always based on group_factor
    gf = group_factor.value
    filter_mode = group_filter_mode.value
    selected_groups = set(map(str, group_filter_select.value))
    if gf != "(None)" and filter_mode != "All" and selected_groups:
        keep_cols = []
        for c in cols:
            lab = _safe_group_label(c, gf, bin_set)
            if filter_mode == "Include only":
                if lab in selected_groups:
                    keep_cols.append(c)
            else:  # Exclude
                if lab not in selected_groups:
                    keep_cols.append(c)
        cols = keep_cols
        A = A[cols]
        if not cols:
            return pn.pane.Markdown("⚠️ No samples left after group filter.")
            
    # apply outer-group filter (second factor), if enabled
    of = outer_group_factor.value
    of_mode = outer_filter_mode.value
    of_sel = set(map(str, outer_filter_select.value))
    if of != "(None)" and of in metadata.columns and of_mode != "All" and of_sel:
        keep_cols2 = []
        for c in cols:
            if c not in metadata.index:
                continue
            lab2 = str(metadata.at[c, of])
            if of_mode == "Include only":
                if lab2 in of_sel:
                    keep_cols2.append(c)
            else:  # Exclude
                if lab2 not in of_sel:
                    keep_cols2.append(c)
        cols = keep_cols2
        A = A[cols]
        if not cols:
            return pn.pane.Markdown("⚠️ No samples left after outer-group filter.")


    # Decide which factor controls grouping for X ordering
    src = x_group_source.value
    of = outer_group_factor.value

    if x_order_mode.value == "Manual" and manual_x_sequence:
        if src == "Outer group" and of != "(None)":
            # Manual sequence = list of outer-group conditions
            ordered_cols = _expand_groups_to_samples(
                cols,
                manual_x_sequence,
                of,
                bin_set=set(),  # no binning on outer groups
            )
        elif src == "Group / x-axis factor" and gf != "(None)":
            # Manual sequence = list of group-factor conditions
            ordered_cols = _expand_groups_to_samples(
                cols,
                manual_x_sequence,
                gf,
                bin_set=bin_set,
            )
        else:
            # Manual sequence = list of sample IDs
            sel = [c for c in manual_x_sequence if c in cols]
            ordered_cols = sel + [c for c in cols if c not in sel]

    else:
        # Auto mode: group according to the chosen X-group source
        if src == "Outer group" and of != "(None)":
            # group by outer group
            ordered_cols = _auto_grouped_sample_order(
                cols, of, bin_set=set()
            )
        elif src == "Group / x-axis factor" and gf != "(None)":
            # group by group_factor (default behaviour)
            ordered_cols = _auto_grouped_sample_order(
                cols, gf, bin_set=bin_set
            )
        else:
            # raw sample order
            ordered_cols = cols



    if sort_taxon.value != "(None)" and sort_taxon.value in A.index:
        tax = sort_taxon.value
        vals = A.loc[tax].reindex(ordered_cols).fillna(0.0)
        if sort_scope.value == "Global":
            ordered_cols = sorted(
                ordered_cols, key=lambda c: vals[c], reverse=bool(sort_desc.value)
            )
        else:
            if gf != "(None)":
                def lab(x):
                    return _safe_group_label(x, gf, bin_set)

                blocks, cur = [], [ordered_cols[0]]
                for c in ordered_cols[1:]:
                    if lab(c) == lab(cur[-1]):
                        cur.append(c)
                    else:
                        blocks.append(cur)
                        cur = [c]
                blocks.append(cur)
                ordered_cols = sum(
                    [
                        sorted(
                            b,
                            key=lambda c: vals[c],
                            reverse=bool(sort_desc.value),
                        )
                        for b in blocks
                    ],
                    [],
                )
            else:
                ordered_cols = sorted(
                    ordered_cols, key=lambda c: vals[c], reverse=bool(sort_desc.value)
                )

    pos, spans_top = _positions_preserve_order_with_outer_gap(
        ordered_cols,
        outer_factor=(
            outer_group_factor.value
            if outer_group_factor.value != "(None)"
            else None
        ),
        gap=int(group_gap.value),
        bin_set=bin_set,
    )

    A = A.loc[legend_cats][ordered_cols]
    n = len(ordered_cols)
    if plot_orientation.value == "Horizontal / wide":
        figsize = (figure_width.value, figure_height.value)
    else:
        figsize = (figure_height.value, figure_width.value)
    fig, ax = plt.subplots(figsize=figsize)

    colors = _get_taxon_colors(legend_cats)

    if flip_axes.value:
        # Horizontal bars: samples/groups on Y, abundance on X
        y = np.array(pos, dtype=float)
        left = np.zeros(n)
        for t in [tt for tt in legend_cats if tt in A.index]:
            vals = A.loc[t].to_numpy()
            ax.barh(
                y,
                vals,
                left=left,
                label=t,
                height=1.0,
                color=colors.get(t),
            )
            left += vals
    else:
        # Standard vertical stacked bars
        bottom = np.zeros(n)
        x = np.array(pos, dtype=float)
        for t in [tt for tt in legend_cats if tt in A.index]:
            vals = A.loc[t].to_numpy()
            ax.bar(
                x,
                vals,
                bottom=bottom,
                label=t,
                width=1.0,
                color=colors.get(t),
            )
            bottom += vals

    if flip_axes.value:
        ax.set_xlabel("Relative abundance (%)")
    else:
        ax.set_ylabel("Relative abundance (%)")

    # Apply optional manual limit for the abundance axis
    _apply_abundance_axis_limits(ax)

    rot = int(x_label_rotation.value)

    if not flip_axes.value:
        # sample names under each sample (Run IDs) if requested
        if show_sample_labels.value:
            ax.set_xticks(x)
            ax.set_xticklabels(
                [str(c) for c in ordered_cols],
                rotation=rot,
                ha=("right" if rot else "center"),
                fontsize=int(x_label_fontsize.value),
            )
            centers = None
            group_labels = None
        else:
            centers, group_labels = _compute_group_centers_for_labels(
                ordered_cols,
                pos,
                gf if gf != "(None)" else None,
                bin_set=bin_set,
            )
            if centers is not None:
                # ---- NEW: n labels for grouped ticks in Per-sample stacked ----
                bc, bl, ns_list, nd_list = _compute_n_maps_for_blocks(
                    ordered_cols,
                    gf if gf != "(None)" else None,
                    bin_set=bin_set,
                )
                # map index-space centers to real x positions (pos may include gaps)
                if bc is not None:
                    mapped_centers = []
                    for cidx in bc:
                        lo = int(np.floor(cidx))
                        hi = int(np.ceil(cidx))
                        lo = max(0, min(lo, len(pos) - 1))
                        hi = max(0, min(hi, len(pos) - 1))
                        mapped_centers.append((pos[lo] + pos[hi]) / 2.0)
                    centers = mapped_centers

                tick_labels = list(group_labels)

                if show_n_per_bar.value and (bc is not None):
                    n_main = nd_list if n_count_basis.value == "source_folder (studies)" else ns_list
                    if n_display_mode.value == "Append to label":
                        new_labels = []
                        for lab, ns, nd, nn in zip(bl, ns_list, nd_list, n_main):
                            if n_show_both.value:
                                new_labels.append(f"{lab} ({_format_n_label(ns, nd)})")
                            else:
                                new_labels.append(f"{lab} (n={nn})")
                        tick_labels = new_labels

                ax.set_xticks(centers)
                ax.set_xticklabels(
                    tick_labels,
                    rotation=rot,
                    ha=("right" if rot else "center"),
                    fontsize=int(x_label_fontsize.value),
                )

                if show_n_per_bar.value and (bc is not None) and (n_display_mode.value == "Above bars"):
                    n_main = nd_list if n_count_basis.value == "source_folder (studies)" else ns_list
                    y_text = _abundance_axis_top(ax) * 1.01
                    for cx, ns, nd, nn in zip(centers, ns_list, nd_list, n_main):
                        txt = _format_n_label(ns, nd) if n_show_both.value else f"n={nn}"
                        ax.text(
                            float(cx),
                            float(y_text),
                            txt,
                            ha="center",
                            va="bottom",
                            fontsize=max(6, int(x_label_fontsize.value) - 1),
                            clip_on=False,
                        )
                # -------------------------------------------------------------
            else:
                if n > 120:
                    step = max(1, n // 120)
                    ticks = x[::step]
                    labels = [str(c) for c in ordered_cols[::step]]
                else:
                    ticks = x
                    labels = [str(c) for c in ordered_cols]
                ax.set_xticks(ticks)
                ax.set_xticklabels(
                    labels,
                    rotation=rot,
                    ha=("right" if rot else "center"),
                    fontsize=int(x_label_fontsize.value),
                )
    else:
        # Flipped: samples/groups on Y axis
        y = np.array(pos, dtype=float)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [str(c) for c in ordered_cols],
            rotation=0,
            ha="right",
            fontsize=int(x_label_fontsize.value),
        )
        centers = None
        group_labels = None

    # Draw group boundaries according to selected source
    boundary_factor = None
    boundary_bin = None

    if group_boundary_source.value == "Group / x-axis factor":
        # Use the primary X factor (group_factor), with binning to "Other"
        boundary_factor = gf
        boundary_bin = bin_set
    elif group_boundary_source.value == "Outer group":
        # Use the secondary X factor (outer_group_factor), no binning
        boundary_factor = outer_group_factor.value
        boundary_bin = None

    if boundary_factor and boundary_factor != "(None)" and not flip_axes.value:
        _draw_group_boundaries(
            ax,
            ordered_cols,
            pos,
            label_factor=boundary_factor,
            bin_set=boundary_bin,
        )

    if spans_top:
        y_top = ax.get_ylim()[1]
        rot_top = int(outer_x_label_rotation.value)
        ha = "center"
        va = "bottom"
        rotation_mode = "default"
        if rot_top != 0:
            ha = "right"
            rotation_mode = "anchor"

        for (olab, xs, xe) in spans_top:
            ax.text(
                (xs + xe) / 2.0,
                y_top * 1.02,
                str(olab),
                ha=ha,
                va=va,
                fontsize=int(x_label_fontsize.value),
                rotation=rot_top,
                rotation_mode=rotation_mode,
            )
        ax.margins(y=0.15)

    if show_legend.value:
        h, l = ax.get_legend_handles_labels()
        pos = legend_position.value

        if pos == "outside right":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.,
            )
        elif pos == "outside left":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(-0.02, 1),
                loc="upper right",
                borderaxespad=0.,
            )
        elif pos == "above":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(0.5, 1.15),
                loc="upper center",
                borderaxespad=0.,
                ncol=2,
            )
        elif pos == "below":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(0.5, -0.15),
                loc="lower center",
                borderaxespad=0.,
                ncol=2,
            )
        else:
            ax.legend(
                h[::-1], l[::-1],
                loc=pos,
                frameon=True,
            )

    _apply_common_style(ax)
    fig.tight_layout()
    _last_fig["fig"] = fig
    _last_fig["ax"] = ax
    return pn.pane.Matplotlib(fig, tight=True)


def plot_group_average(A: pd.DataFrame, legend_cats: list):
    # map taxon -> list of matplotlib Rectangle patches (used for letter placement)
    bar_patches_by_taxon = {}
    if A.empty:
        return pn.pane.Markdown("⚠️ No data to plot.")
    # Fetch the last pre-avg matrix and sample membership per group (computed in build_aggregated_matrix)
    # These are used for n and SE annotations in average mode.
    global _LAST_PREAVG_MATRIX, _LAST_GROUP_TO_SAMPLES
    bin_set = set(group_bin_other.value)

    # Bin selected groups into "Other" at column level
    if group_factor.value != "(None)":
        other_cols = [g for g in A.columns if g in bin_set]
        keep_cols = [g for g in A.columns if g not in bin_set]
        if other_cols:
            other_series = A[other_cols].mean(axis=1)
            A = pd.concat(
                [A[keep_cols], other_series.rename("Other")],
                axis=1,
            )
    cols = list(A.columns)

    # apply group filter at group level (per-group average)
    filter_mode = group_filter_mode.value
    selected_groups = set(map(str, group_filter_select.value))
    if filter_mode != "All" and selected_groups:
        if filter_mode == "Include only":
            cols = [c for c in cols if c in selected_groups]
        else:  # Exclude
            cols = [c for c in cols if c not in selected_groups]
        A = A[cols]
        if not cols:
            return pn.pane.Markdown("⚠️ No groups left after group filter.")

    # Manual X ordering (groups)
    if x_order_mode.value == "Manual" and manual_x_sequence:
        sel = [c for c in manual_x_sequence if c in cols]
        cols = sel + [c for c in cols if c not in sel]

    # Optional sort by taxon, BUT ONLY in Auto mode (robust string matching)
    if x_order_mode.value != "Manual" and sort_taxon.value != "(None)":
        tax = str(sort_taxon.value)
        # map stringified taxa to actual index labels
        _idx_match = None
        for _lab in A.index:
            if str(_lab) == tax:
                _idx_match = _lab
                break
        if _idx_match is not None:
            vals = A.loc[_idx_match].reindex(cols).fillna(0.0)
            cols = sorted(cols, key=lambda c: float(vals.get(c, 0.0)), reverse=bool(sort_desc.value))

    # --- Compute n and SE maps (only meaningful in average mode) -------------
    n_samples_map: dict[str, int] = {}
    n_studies_map: dict[str, int] = {}
    n_map: dict[str, int] = {}  # active map (depends on UI choice)
    se_map_taxon: dict[str, dict[str, float]] = {}

    # Build group->samples mapping, respecting binning to "Other"
    group_to_samples: dict[str, list[str]] = {}
    if isinstance(_LAST_GROUP_TO_SAMPLES, dict):
        for g, samp in _LAST_GROUP_TO_SAMPLES.items():
            if g in bin_set:
                group_to_samples.setdefault("Other", []).extend(list(samp))
            else:
                group_to_samples.setdefault(str(g), []).extend(list(samp))

    # n is computed from the sample membership (after binning)
    for g, samp in group_to_samples.items():
        uniq_runs = list(dict.fromkeys(map(str, samp)))

        # samples
        n_samples_map[str(g)] = len(uniq_runs)

        # studies (unique source_folder in metadata)
        sf = []
        for r in uniq_runs:
            sfv = run2meta.get(r, {}).get("source_folder", None)
            if sfv is not None and str(sfv).strip() != "":
                sf.append(str(sfv))
        n_studies_map[str(g)] = len(set(sf))

    # active n map (what "n=" refers to)
    if n_count_basis.value == "source_folder (studies)":
        n_map = dict(n_studies_map)
    else:
        n_map = dict(n_samples_map)

    # SE per-taxon (segment) per group, computed from the pre-avg sample matrix
    if show_se.value and _LAST_PREAVG_MATRIX is not None:
        pre_idx_map = {str(ix): ix for ix in _LAST_PREAVG_MATRIX.index}
        taxa_for_se = [str(t) for t in A.index if str(t) in pre_idx_map]
        for t in taxa_for_se:
            se_map_taxon.setdefault(t, {})
        for g, samp in group_to_samples.items():
            ss = [s for s in samp if s in _LAST_PREAVG_MATRIX.columns]
            if len(ss) <= 1:
                for t in taxa_for_se:
                    se_map_taxon[t][str(g)] = 0.0
                continue
            for t in taxa_for_se:
                ix = pre_idx_map[t]
                v = _LAST_PREAVG_MATRIX.loc[ix, ss].astype(float).to_numpy()
                se_map_taxon[t][str(g)] = float(np.nanstd(v, ddof=1) / np.sqrt(len(v)))
    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------

    A = A.loc[legend_cats][cols]
    colors = _get_taxon_colors(legend_cats)
    x = np.arange(len(cols), dtype=float)

    if plot_orientation.value == "Horizontal / wide":
        figsize = (figure_width.value, figure_height.value)
    else:
        figsize = (figure_height.value, figure_width.value)
    fig, ax = plt.subplots(figsize=figsize)
    max_annot_y = None  # track highest letter position to auto-expand ylim safely

    # Optional: remove top/right plot border for a cleaner look (helps when annotations approach the top)
    if remove_spines.value:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if flip_axes.value:
        # Horizontal bars: samples/groups on Y, abundance on X
        y = x.copy()
        left = np.zeros(len(cols))
        for t in [tt for tt in legend_cats if tt in A.index]:
            vals = A.loc[t].to_numpy()
            ax.barh(
                y,
                vals,
                left=left,
                label=t,
                height=0.8,
                color=colors.get(t),
            )
            if show_se.value and se_map_taxon:
                tkey = str(t)
                for i, c in enumerate(cols):
                    se = float(se_map_taxon.get(tkey, {}).get(str(c), 0.0))
                    if se > 0:
                        x_center = float(left[i] + vals[i])  # top of segment
                        ax.errorbar(
                            x_center,
                            float(y[i]),
                            xerr=se,
                            fmt="none",
                            ecolor="black",
                            capsize=2,
                            linewidth=1,
                        )
            left += vals
    else:
        # Standard vertical stacked bars
        bottom = np.zeros(len(cols))
        for t in [tt for tt in legend_cats if tt in A.index]:
            vals = A.loc[t].to_numpy()
            cont = ax.bar(
                x,
                vals,
                bottom=bottom,
                label=t,
                width=0.8,
                color=colors.get(t),
            )
            bar_patches_by_taxon[str(t)] = list(cont.patches)
            if show_se.value and se_map_taxon:
                tkey = str(t)
                # draw SE as errorbars centered on each stacked segment
                for i, c in enumerate(cols):
                    se = float(se_map_taxon.get(tkey, {}).get(str(c), 0.0))
                    if se > 0:
                        y_center = float(bottom[i] + vals[i])  # top of segment
                        ax.errorbar(
                            float(x[i]),
                            y_center,
                            yerr=se,
                            fmt="none",
                            capsize=2,
                            linewidth=1,
                        )
            bottom += vals

    # --- annotate ANOVA posthoc letters on bars (if available) -------------
    gf = str(group_factor.value)
    # Apply optional manual limit for the abundance axis
    _apply_abundance_axis_limits(ax)

    if show_anova_letters.value and gf != "(None)" and ANOVA_LETTERS and bar_patches_by_taxon:
        multiple_taxa_in_plot = len(bar_patches_by_taxon) > 1
        for taxon_name, patches in bar_patches_by_taxon.items():
            letters_map = ANOVA_LETTERS.get((str(taxon_name), gf))
            if not letters_map:
                continue
            for i, c in enumerate(cols):
                if i >= len(patches):
                    continue
                letter = letters_map.get(str(c), "")
                if not letter:
                    continue
                rect = patches[i]
                # Place inside segment if multiple taxa; otherwise just above bar
                if multiple_taxa_in_plot:
                    y = rect.get_y() + rect.get_height() / 2.0
                    va = "center"
                else:
                    # Put letters above the *top of the segment + its SE* to avoid overlapping the error bar
                    y_top = rect.get_y() + rect.get_height()
                    se_here = 0.0
                    try:
                        se_here = float(se_map_taxon.get(str(taxon_name), {}).get(str(c), 0.0))
                    except Exception:
                        se_here = 0.0

                    # Offset relative to current y-range (more stable than using y itself)
                    y0, y1 = ax.get_ylim()
                    yr = max(1e-9, float(y1 - y0))
                    offset = 0.015 * yr  # tweak if you want more/less spacing

                    y = float(y_top) + float(se_here) + float(offset)
                    va = "bottom"
                    try:
                        max_annot_y = y if (max_annot_y is None or y > max_annot_y) else max_annot_y
                    except Exception:
                        pass
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    y,
                    str(letter),
                    ha="center",
                    va=va,
                    fontsize=11,
                    fontweight="normal",
                    color="black",
                    clip_on=False,
                )

    # Ensure post-hoc letters don't get clipped at the top
    if max_annot_y is not None and y_axis_mode.value != "Manual":
        y0, y1 = ax.get_ylim()
        # Add headroom if the highest annotation is near/above the top
        if max_annot_y >= y1 - 1e-12:
            ax.set_ylim(y0, max_annot_y * 1.05)
        elif max_annot_y > 0.94 * y1:
            ax.set_ylim(y0, y1 * 1.06)

    # --- annotate correlation (persistent) ---------------------------------
    if show_corr_on_plot.value and PLOT_STATE["corr_annot"]:
        ax.text(
            0.02, 0.98, str(PLOT_STATE["corr_annot"]),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12,
            color="red",
        )

    # --- n / SE annotations --------------------------------------------------
    if show_n_per_bar.value and n_display_mode.value == "Above bars":
        for i, c in enumerate(cols):
            nn = n_map.get(str(c), 0)
            nn_samp = n_samples_map.get(str(c), 0)
            nn_stud = n_studies_map.get(str(c), 0)
            if flip_axes.value:
                total = float(A[c].sum())
                ax.text(
                    total + 0.5,
                    x[i],
                    (_format_n_label(nn_samp, nn_stud) if n_show_both.value else f"n={nn}"),
                    va="center",
                    ha="left",
                    fontsize=max(6, int(x_label_fontsize.value) - 1),
                )
            else:
                total = float(A[c].sum())
                ax.text(
                    x[i],
                    total + 0.5,
                    (_format_n_label(nn_samp, nn_stud) if n_show_both.value else f"n={nn}"),
                    va="bottom",
                    ha="center",
                    fontsize=max(6, int(x_label_fontsize.value) - 1),
                )

    # ------------------------------------------------------------------------




    if flip_axes.value:
        ax.set_xlabel("Average relative abundance (%)")
        ax.set_yticks(x)
        ax.set_yticklabels(
            [(
                f"{c} (" + (_format_n_label(n_samples_map.get(str(c), 0), n_studies_map.get(str(c), 0)) if n_show_both.value else f"n={n_map.get(str(c), 0)}") + ")"
                if (show_n_per_bar.value and n_display_mode.value == 'Append to label')
                else str(c)
            ) for c in list(A.columns)],
            rotation=0,
            ha="right",
            fontsize=int(x_label_fontsize.value),
        )
    else:
        ax.set_ylabel("Average relative abundance (%)")
        rot = int(x_label_rotation.value)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [(
                f"{c} (" + (_format_n_label(n_samples_map.get(str(c), 0), n_studies_map.get(str(c), 0)) if n_show_both.value else f"n={n_map.get(str(c), 0)}") + ")"
                if (show_n_per_bar.value and n_display_mode.value == 'Append to label')
                else str(c)
            ) for c in list(A.columns)],
            rotation=rot,
            ha=("right" if rot else "center"),
            fontsize=int(x_label_fontsize.value),
    )

    if show_legend.value:
        h, l = ax.get_legend_handles_labels()
        pos = legend_position.value

        if pos == "outside right":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0.,
            )
        elif pos == "outside left":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(-0.02, 1),
                loc="upper right",
                borderaxespad=0.,
            )
        elif pos == "above":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(0.5, 1.15),
                loc="upper center",
                borderaxespad=0.,
                ncol=2,
            )
        elif pos == "below":
            ax.legend(
                h[::-1], l[::-1],
                bbox_to_anchor=(0.5, -0.15),
                loc="lower center",
                borderaxespad=0.,
                ncol=2,
            )
        else:
            ax.legend(
                h[::-1], l[::-1],
                loc=pos,
                frameon=True,
            )

    _apply_common_style(ax)
    fig.tight_layout()
    _last_fig["fig"] = fig
    _last_fig["ax"] = ax
    return pn.pane.Matplotlib(fig, tight=True)



# ==============================
# PCoA helpers
# ==============================
def _build_pcoa_input_matrix() -> pd.DataFrame:
    """
    Build a per-sample taxa matrix (taxa x samples) from the currently selected
    taxonomy subset settings (Top-N / Manual), applying sample filters.

    Steps:
      1) Subset taxa exactly like the taxonomy plot selection
      2) Apply per-sample filters (group + outer filters)
      3) Standardise to proportions per sample and square-root transform
      4) Cache to _LAST_PCOA_INPUT
    """
    if not _ensure_REL_loaded():
        return pd.DataFrame()

    rel = REL.copy()
    rel.index = rel.index.astype(str)

    # Subset taxa exactly like the taxonomy plot selection
    if subset_mode.value == "Top-N":
        cols_for_topn = _columns_for_topN(rel)
        totals = rel[cols_for_topn].sum(axis=1)
        top = totals.nlargest(max(1, int(TopN.value))).index.tolist()
        M = rel.loc[top].copy()
        if include_others.value and len(rel.index) > len(top):
            others = rel.drop(index=top).sum(axis=0)
            M.loc["Others"] = others
    else:
        sel = [str(t) for t in manual_taxa.value if str(t) in rel.index]
        if not sel:
            sel = rel.sum(axis=1).nlargest(10).index.tolist()
        M = rel.loc[sel].copy()

    # Apply per-sample filters
    cols = list(M.columns)
    bin_set = set(group_bin_other.value)

    gf = group_factor.value
    filter_mode = group_filter_mode.value
    selected_groups = set(map(str, group_filter_select.value))
    if gf != "(None)" and filter_mode != "All" and selected_groups:
        keep_cols = []
        for c in cols:
            lab = _safe_group_label(c, gf, bin_set)
            if filter_mode == "Include only":
                if lab in selected_groups:
                    keep_cols.append(c)
            else:
                if lab not in selected_groups:
                    keep_cols.append(c)
        cols = keep_cols

    of = outer_group_factor.value
    of_mode = outer_filter_mode.value
    of_sel = set(map(str, outer_filter_select.value))
    if of != "(None)" and of in metadata.columns and of_mode != "All" and of_sel:
        keep_cols2 = []
        for c in cols:
            if c not in metadata.index:
                continue
            lab2 = str(metadata.at[c, of])
            if of_mode == "Include only":
                if lab2 in of_sel:
                    keep_cols2.append(c)
            else:
                if lab2 not in of_sel:
                    keep_cols2.append(c)
        cols = keep_cols2

    cols = [c for c in cols if c in metadata.index]
    if not cols:
        return pd.DataFrame()

    M = M[cols].copy()

    # Standardise to proportions per sample, then sqrt
    col_sums = M.sum(axis=0).replace(0.0, np.nan)
    P = (M / col_sums).fillna(0.0)
    P = np.sqrt(P.astype(float))

    _LAST_PCOA_INPUT["matrix"] = P
    _LAST_PCOA_INPUT["samples"] = list(P.columns)
    return P


def _bray_curtis_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Bray–Curtis dissimilarity for sample-by-feature matrix X.
    Returns an (n x n) symmetric matrix.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    try:
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(X, metric="braycurtis")).astype(float)
    except Exception:
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            xi = X[i]
            for j in range(i + 1, n):
                xj = X[j]
                num = np.abs(xi - xj).sum()
                den = (xi + xj).sum()
                bc = float(num / den) if den != 0 else 0.0
                D[i, j] = bc
                D[j, i] = bc
        return D


def _pcoa_from_distance(D: np.ndarray):
    """
    Classic metric PCoA (classical MDS) from a distance matrix D.
    Returns coords (n x k) and eigenvalues (n,).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n < 2:
        return np.zeros((n, 1)), np.array([0.0])

    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * (J @ D2 @ J)

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-12
    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]
    coords = eigvecs_pos * np.sqrt(eigvals_pos)
    return coords, eigvals



def _run_diversity_summary(event=None):
    """Step 5: compute alpha (Shannon), within-group dispersion, and between-group centroid distances,
    using the SAME selections (taxa subset, Others, binning, sample filters) as the taxonomy plot/PCoA.
    """
    if group_factor.value == "(None)" or group_factor.value not in metadata.columns:
        div_status.object = "⚠️ Set **Group / x-axis factor** first (used as categories)."
        return

    div_status.object = "⏳ Computing diversity summary…"
    try:
        M = _build_pcoa_input_matrix()  # taxa x samples, sqrt(proportions)
        if M is None or M.empty:
            div_status.object = "⚠️ No data (after filters)."
            return

        samples = list(M.columns)
        gf = str(group_factor.value)

        # Group labels with taxonomy binning into 'Other'
        g_labels = []
        for s in samples:
            g = apply_group_binning(s, gf)
            g_labels.append(str(g) if g is not None else "NA")

        # Reconstruct proportions (because M is sqrt(P))
        P = (M.astype(float) ** 2)
        # ensure columns sum to 1 (numerical safety)
        col_sums = P.sum(axis=0).replace(0.0, np.nan)
        P = (P / col_sums).fillna(0.0)

        # Alpha: Shannon per sample
        arr = P.to_numpy().T  # samples x taxa
        # avoid log(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            shannon = -(arr * np.log(arr)).sum(axis=1)
            shannon = np.nan_to_num(shannon, nan=0.0, posinf=0.0, neginf=0.0)

        # Bray–Curtis + PCoA coords (reuse existing implementation)
        # IMPORTANT: M is taxa x samples; Bray–Curtis must be computed BETWEEN SAMPLES
        D = _bray_curtis_distance(M.T)  # samples x taxa (sqrt-proportions)
        coords, eigvals = _pcoa_from_distance(D)
        if coords is None or coords.size == 0:
            div_status.object = "⚠️ Could not compute PCoA coordinates."
            return

        # Use only positive-eigenvalue axes (stable)
        try:
            pos_idx = [i for i,v in enumerate(eigvals) if float(v) > 0]
            k = min(10, len(pos_idx)) if pos_idx else min(2, coords.shape[1])
            use = pos_idx[:k] if pos_idx else list(range(k))
        except Exception:
            use = list(range(min(2, coords.shape[1])))

        C = coords[:, use]

        # Within-group dispersion: mean distance-to-centroid in PCoA space
        import pandas as pd
        df = pd.DataFrame({
            "sample": samples,
            "group": g_labels,
            "shannon": shannon,
        })

        disp_rows = []
        centroids = {}
        for g, sub in df.groupby("group"):
            idx = sub.index.to_numpy()
            pts = C[idx, :]
            if pts.shape[0] == 0:
                continue
            cen = pts.mean(axis=0)
            centroids[g] = cen
            dists = np.sqrt(((pts - cen) ** 2).sum(axis=1))
            disp_rows.append({
                "group": g,
                "n_samples": int(len(idx)),
                "shannon_mean": float(sub["shannon"].mean()),
                "shannon_median": float(sub["shannon"].median()),
                "dispersion_mean": float(dists.mean()),
                "dispersion_median": float(np.median(dists)),
            })

        groups_df = pd.DataFrame(disp_rows).sort_values("group").reset_index(drop=True)

        # Between-group centroid distances (Euclidean in chosen PCoA space)
        gnames = list(centroids.keys())
        mat = np.zeros((len(gnames), len(gnames)), dtype=float)
        for i, a in enumerate(gnames):
            for j in range(i+1, len(gnames)):
                b = gnames[j]
                da = centroids[a]; db = centroids[b]
                dist = float(np.sqrt(((da - db) ** 2).sum()))
                mat[i, j] = dist
                mat[j, i] = dist
        pairs_df = pd.DataFrame(mat, index=gnames, columns=gnames)

        # Rankings
        if not groups_df.empty:
            g_min_alpha = groups_df.sort_values("shannon_mean").iloc[0]
            g_max_alpha = groups_df.sort_values("shannon_mean").iloc[-1]
            g_min_disp = groups_df.sort_values("dispersion_mean").iloc[0]
            g_max_disp = groups_df.sort_values("dispersion_mean").iloc[-1]
        else:
            g_min_alpha = g_max_alpha = g_min_disp = g_max_disp = None

        # Most similar/different pairs
        best_pair = worst_pair = None
        if len(gnames) >= 2:
            tri = []
            for i in range(len(gnames)):
                for j in range(i+1, len(gnames)):
                    tri.append((gnames[i], gnames[j], mat[i, j]))
            tri_sorted = sorted(tri, key=lambda x: x[2])
            best_pair = tri_sorted[0]
            worst_pair = tri_sorted[-1]

        lines = []
        lines.append(f"**Samples used:** {len(samples)} • **Groups:** {len(gnames)} • **Taxa (selected):** {M.shape[0]}")
        if g_min_alpha is not None:
            lines.append(f"- **Lowest mean Shannon (alpha):** {g_min_alpha['group']} (mean={g_min_alpha['shannon_mean']:.3f})")
            lines.append(f"- **Highest mean Shannon (alpha):** {g_max_alpha['group']} (mean={g_max_alpha['shannon_mean']:.3f})")
            lines.append(f"- **Most internally consistent (lowest dispersion):** {g_min_disp['group']} (mean={g_min_disp['dispersion_mean']:.3f})")
            lines.append(f"- **Most heterogeneous (highest dispersion):** {g_max_disp['group']} (mean={g_max_disp['dispersion_mean']:.3f})")
        if best_pair is not None:
            lines.append(f"- **Most similar groups (centroid distance):** {best_pair[0]} vs {best_pair[1]} (d={best_pair[2]:.3f})")
            lines.append(f"- **Most different groups (centroid distance):** {worst_pair[0]} vs {worst_pair[1]} (d={worst_pair[2]:.3f})")

        div_summary_md.object = "\n".join(lines)
        div_table_groups.object = groups_df
        div_table_pairs.object = pairs_df
        div_status.object = "✓ Diversity summary computed."
        try:
            plot_area.active = 2  # switch to Diversity tab
        except Exception:
            pass
    except Exception as e:
        import traceback
        div_status.object = f"⚠️ Error: {e}"
        div_summary_md.object = "```\n" + traceback.format_exc() + "\n```"

div_btn.on_click(_run_diversity_summary)

# ==============================
# SIMPER wiring + computation
# ==============================
def _simper_clean_list(vals):
    return [str(v) for v in (vals or []) if str(v).strip() != ""]

def _simper_get_samples_and_labels(factor: str):
    """Return (samples, labels) for the current data subset (REL columns), mapped via metadata."""
    if (factor == "(None)") or (factor not in metadata.columns):
        return [], []
    try:
        _ensure_REL_loaded()
    except Exception:
        return [], []

    if REL is None or REL.empty:
        return [], []

    samples = [s for s in REL.columns if s in metadata.index]
    # Apply the SAME binning used in taxonomy plots when factor == group_factor
    try:
        bin_set = set(map(str, group_bin_other.value)) if str(factor) == str(group_factor.value) else set()
    except Exception:
        bin_set = set()

    labels = []
    for s in samples:
        labels.append(_safe_group_label(s, factor, bin_set))
    return samples, labels


def _simper_sync_widgets(event=None):
    """Populate condition pool, then keep A/B options consistent and disjoint."""
    if not data_loaded.value:
        simper_condition_pool.options = []
        simper_condition_pool.value = []
        simper_groupA.options = []
        simper_groupA.value = []
        simper_groupB.options = []
        simper_groupB.value = []
        return

    factor = str(simper_factor.value)
    samples, labels = _simper_get_samples_and_labels(factor)
    uniq = _unique_in_order(labels)

    # Update pool options
    simper_condition_pool.options = uniq

    # If pool has no explicit value yet, default to "all"
    pool_val = _simper_clean_list(simper_condition_pool.value)
    if not pool_val:
        pool_val = list(uniq)
        simper_condition_pool.value = pool_val
    else:
        pool_val = [v for v in pool_val if v in uniq]
        simper_condition_pool.value = pool_val

    # Enforce A/B within pool and disjoint
    A = _simper_clean_list(simper_groupA.value)
    B = _simper_clean_list(simper_groupB.value)

    A = [v for v in A if v in pool_val]
    B = [v for v in B if v in pool_val and v not in A]

    simper_groupA.options = list(pool_val)
    simper_groupB.options = [v for v in pool_val if v not in A]

    simper_groupA.value = A
    simper_groupB.value = B


def _simper_on_A_change(event=None):
    # Rebuild B options to exclude A, drop overlaps
    _simper_sync_widgets()

def _simper_on_B_change(event=None):
    # Just ensure disjointness (drop overlaps from B)
    _simper_sync_widgets()

def _simper_on_pool_change(event=None):
    _simper_sync_widgets()

def _simper_on_factor_change(event=None):
    # Reset A/B to avoid stale choices when factor changes
    simper_groupA.value = []
    simper_groupB.value = []
    _simper_sync_widgets()

# Watchers
try:
    simper_factor.param.watch(_simper_on_factor_change, "value")
    simper_condition_pool.param.watch(_simper_on_pool_change, "value")
    simper_groupA.param.watch(_simper_on_A_change, "value")
    simper_groupB.param.watch(_simper_on_B_change, "value")
    data_loaded.param.watch(lambda e: _simper_sync_widgets(), "value")
    group_bin_other.param.watch(lambda e: _simper_sync_widgets(), "value")
    group_factor.param.watch(lambda e: _simper_sync_widgets(), "value")
except Exception:
    pass


def _compute_simper(event=None):
    """
    SIMPER between Group A and Group B using the same taxa matrix used for PCoA:
    - build sqrt(proportions) matrix (taxa x samples)
    - reconstruct proportions and compute average Bray–Curtis taxon contributions across all A×B pairs
    """
    if not data_loaded.value:
        simper_status.object = "⚠️ Load data first."
        return

    factor = str(simper_factor.value)
    if (factor == "(None)") or (factor not in metadata.columns):
        simper_status.object = "⚠️ Choose a **SIMPER factor** (metadata column)."
        return

    pool = _simper_clean_list(simper_condition_pool.value)
    A_conds = _simper_clean_list(simper_groupA.value)
    B_conds = _simper_clean_list(simper_groupB.value)

    if not A_conds or not B_conds:
        simper_status.object = "⚠️ Select at least one condition for **Group A** and **Group B**."
        return

    # prevent overlap
    overlap = set(A_conds) & set(B_conds)
    if overlap:
        simper_status.object = f"⚠️ Group A and B overlap: {', '.join(sorted(overlap))}. Remove overlap."
        return

    simper_status.object = "⏳ Running SIMPER…"
    simper_table.object = pd.DataFrame()
    simper_csv_note.object = ""

    # NOTE: do not import pandas as pd inside this function after using `pd` above.
    # Doing so makes `pd` a local variable and triggers UnboundLocalError.
    import numpy as np
    import time

    t0 = time.time()

    try:
        # taxa x samples, sqrt(proportions)
        M = _build_pcoa_input_matrix()
        if M is None or M.empty:
            simper_status.object = "⚠️ No data (after filters)."
            return

        samples = list(M.columns)

        # labels with binning-to-Other if factor == group_factor
        try:
            bin_set = set(map(str, group_bin_other.value)) if str(factor) == str(group_factor.value) else set()
        except Exception:
            bin_set = set()

        labels = []
        for s in samples:
            labels.append(_safe_group_label(s, factor, bin_set))

        # restrict to pool
        keep = [i for i,(s,l) in enumerate(zip(samples, labels)) if str(l) in set(pool)]
        if not keep:
            simper_status.object = "⚠️ No samples left after the condition pool filter."
            return

        samples = [samples[i] for i in keep]
        labels = [str(labels[i]) for i in keep]

        # map samples to A/B sets
        A_samples = [s for s,l in zip(samples, labels) if l in set(A_conds)]
        B_samples = [s for s,l in zip(samples, labels) if l in set(B_conds)]

        if len(A_samples) < 2 or len(B_samples) < 2:
            simper_status.object = f"⚠️ Need at least 2 samples per side (A={len(A_samples)}, B={len(B_samples)})."
            return

        # Reconstruct proportions from sqrt(P)
        P = (M.astype(float) ** 2).reindex(columns=samples)
        col_sums = P.sum(axis=0).replace(0.0, np.nan)
        P = (P / col_sums).fillna(0.0) * 100.0  # now ~percent; sums ≈ 100

        # numeric arrays
        taxa = list(P.index.astype(str))
        XA = P[A_samples].to_numpy(dtype=np.float32)  # taxa x nA
        XB = P[B_samples].to_numpy(dtype=np.float32)  # taxa x nB

        nA = XA.shape[1]
        nB = XB.shape[1]

        # Denominator per pair (sumA_i + sumB_j). Usually constant (~200).
        sumA = XA.sum(axis=0).astype(np.float32)  # nA
        sumB = XB.sum(axis=0).astype(np.float32)  # nB

        use_const = False
        denom_const = None
        try:
            if np.allclose(sumA, float(sumA[0]), rtol=1e-3, atol=1e-2) and np.allclose(sumB, float(sumB[0]), rtol=1e-3, atol=1e-2):
                denom_const = float(sumA[0] + sumB[0])
                if denom_const > 0:
                    use_const = True
        except Exception:
            use_const = False

        if (not use_const):
            D = (sumA[:, None] + sumB[None, :]).astype(np.float32)
            D[D == 0] = np.nan  # avoid divide-by-zero

        # Chunked SIMPER to avoid huge memory
        chunk = 250
        avg_contrib = np.zeros((len(taxa),), dtype=np.float64)
        sd_contrib = np.zeros((len(taxa),), dtype=np.float64)

        for start in range(0, len(taxa), chunk):
            end = min(len(taxa), start + chunk)
            A_chunk = XA[start:end, :]  # m x nA
            B_chunk = XB[start:end, :]  # m x nB

            # m x nA x nB
            diff = np.abs(A_chunk[:, :, None] - B_chunk[:, None, :]).astype(np.float32)
            if use_const:
                contrib = diff / float(denom_const)
            else:
                contrib = diff / D[None, :, :]

            # average and sd over all pairs
            flat = contrib.reshape(contrib.shape[0], -1)
            avg_contrib[start:end] = np.nanmean(flat, axis=1)
            sd_contrib[start:end] = np.nanstd(flat, axis=1)

        mean_bc = float(np.nansum(avg_contrib))
        if mean_bc <= 0:
            simper_status.object = "⚠️ Mean between-group Bray–Curtis is 0 (nothing to explain)."
            return

        perc = (avg_contrib / mean_bc) * 100.0
        ratio = np.divide(avg_contrib, sd_contrib, out=np.full_like(avg_contrib, np.nan), where=sd_contrib > 0)

        # Mean abundances in each side
        meanA = XA.mean(axis=1).astype(np.float64)
        meanB = XB.mean(axis=1).astype(np.float64)

        out = pd.DataFrame({
            "taxon": taxa,
            "mean_A(%)": meanA,
            "mean_B(%)": meanB,
            "avg_contrib": avg_contrib,
            "sd_contrib": sd_contrib,
            "ratio": ratio,
            "contrib_%": perc,
        }).sort_values("contrib_%", ascending=False).reset_index(drop=True)

        out["cumulative_%"] = out["contrib_%"].cumsum()

        topn = int(simper_top_n.value) if simper_top_n.value else 30
        out_show = out.head(topn).copy()

        # Add context columns (handy when saving)
        ctx = {
            "level": str(level_w.value),
            "niche": str(niche_w.value),
            "plant": str(plant_w.value),
            "simper_factor": factor,
            "groupA": ";".join(A_conds),
            "groupB": ";".join(B_conds),
            "pool": ";".join(pool),
            "nA": int(nA),
            "nB": int(nB),
            "mean_bray_curtis": mean_bc,
        }
        for k, v in ctx.items():
            out_show.insert(0, k, v)

        simper_table.object = out_show

        dt = time.time() - t0
        simper_status.object = (
            f"✓ SIMPER computed in {dt:.1f}s • "
            f"A: {len(A_samples)} samples • B: {len(B_samples)} samples • "
            f"mean Bray–Curtis={mean_bc:.4f}"
        )

        # Save CSV (full results, not only topn)
        if bool(simper_save_csv.value):
            safe = lambda s: "".join(c for c in str(s) if c.isalnum() or c in ("-", "_")).strip("_")[:60]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(CACHE_DIR, "simper")
            os.makedirs(out_dir, exist_ok=True)
            fn = f"simper_{safe(level_w.value)}_{safe(factor)}_{safe('A')}_{safe('B')}_{ts}.csv"
            path = os.path.join(out_dir, fn)

            out_full = out.copy()
            # replicate context in full output too
            for k, v in reversed(list(ctx.items())):
                out_full.insert(0, k, v)

            out_full.to_csv(path, index=False)
            simper_csv_note.object = f"💾 Saved: `{path}`"

        # switch tab
        try:
            plot_area.active = 3  # SIMPER tab (after adding it)
        except Exception:
            pass

    except Exception as e:
        import traceback
        simper_status.object = f"⚠️ Error: {e}"
        simper_csv_note.object = "```\n" + traceback.format_exc() + "\n```"

simper_run_btn.on_click(_compute_simper)

# ==============================
# ANOSIM wiring + computation
# ==============================
def _rankdata_average(x: np.ndarray) -> np.ndarray:
    '''Average ranks for ties (like scipy.stats.rankdata(method='average')).'''
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # average rank in 1..n
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _anosim_stat_from_ranks(ranks: np.ndarray, iu: np.ndarray, ju: np.ndarray, codes: np.ndarray) -> float:
    '''Compute ANOSIM R from precomputed distance ranks for upper-triangle pairs.'''
    n = len(codes)
    m = len(ranks)
    if n < 3 or m < 3:
        return float("nan")

    within = (codes[iu] == codes[ju])
    n_within = int(within.sum())
    n_between = int(m - n_within)
    if n_within == 0 or n_between == 0:
        return float("nan")

    denom = 0.25 * n * (n - 1)
    total_sum = float(ranks.sum())
    sum_within = float(np.dot(ranks, within.astype(np.float32)))
    mean_within = sum_within / n_within
    mean_between = (total_sum - sum_within) / n_between
    return float((mean_between - mean_within) / denom)


def _compute_anosim(event=None):
    '''
    ANOSIM (Analysis of Similarities) based on the CURRENT selections:
      - same taxa subset as taxonomy plot (Top-N / Manual + Others)
      - same sample filters (group + outer filters)
      - same binning to 'Other' (group_bin_other)

    Uses Bray-Curtis on the sqrt(proportions) matrix (same as PCoA input).
    '''
    anosim_status.object = ""
    anosim_result_md.object = ""
    anosim_csv_note.object = ""
    anosim_groups_df.object = pd.DataFrame()

    if not data_loaded.value:
        anosim_status.object = "⚠️ Load data first."
        return

    gf = str(group_factor.value)
    if gf == "(None)" or gf not in metadata.columns:
        anosim_status.object = "⚠️ Set **Group / x-axis factor** first (this defines the ANOSIM groups)."
        return

    anosim_status.object = "⏳ Running ANOSIM…"

    try:
        M = _build_pcoa_input_matrix()  # taxa x samples, sqrt(proportions)
        if M is None or M.empty:
            anosim_status.object = "⚠️ No data (after filters)."
            return

        samples = list(M.columns)
        labels = []
        for s in samples:
            g = apply_group_binning(s, gf)
            labels.append(str(g) if g is not None else "NA")

        # Need at least 2 groups
        uniq = sorted(set(labels))
        if len(uniq) < 2:
            anosim_status.object = "⚠️ ANOSIM needs at least 2 groups (check binning/filters)."
            return

        # Group counts table
        g_counts = pd.Series(labels).value_counts(dropna=False).rename_axis("group").reset_index(name="n_samples")
        anosim_groups_df.object = g_counts

        X = M.T.to_numpy(dtype=float)  # samples x taxa
        D = _bray_curtis_distance_matrix(X)
        n = D.shape[0]
        if n < 3:
            anosim_status.object = "⚠️ Need at least 3 samples after filters."
            return

        iu, ju = np.triu_indices(n, k=1)
        dvec = D[iu, ju]

        # Ranks of distances
        try:
            from scipy.stats import rankdata  # type: ignore
            ranks = rankdata(dvec, method="average").astype(float)
        except Exception:
            ranks = _rankdata_average(dvec).astype(float)

        # Encode group labels as ints
        codes = pd.Categorical(labels).codes.astype(int)

        R_obs = _anosim_stat_from_ranks(ranks, iu, ju, codes)
        if not np.isfinite(R_obs):
            anosim_status.object = "⚠️ Could not compute ANOSIM R (e.g. only within or only between distances)."
            return

        perms = int(anosim_perms.value)
        seed = int(anosim_seed.value) if anosim_seed.value is not None else 1

        p_val = None
        valid = 0
        if perms > 0:
            rng = np.random.default_rng(seed)
            ge = 0
            m_pairs = len(ranks)
            denom = 0.25 * n * (n - 1)
            total_sum = float(ranks.sum())

            for _k in range(perms):
                perm_codes = rng.permutation(codes)
                within = (perm_codes[iu] == perm_codes[ju])
                n_within = int(within.sum())
                n_between = int(m_pairs - n_within)
                if n_within == 0 or n_between == 0:
                    continue
                sum_within = float(np.dot(ranks, within.astype(np.float32)))
                mean_within = sum_within / n_within
                mean_between = (total_sum - sum_within) / n_between
                Rp = float((mean_between - mean_within) / denom)
                valid += 1
                if Rp >= R_obs:
                    ge += 1

            if valid > 0:
                p_val = (ge + 1.0) / (valid + 1.0)
            else:
                p_val = float("nan")

        # Report
        meta_bits = [
            f"**Level:** `{level_w.value}`",
            f"**Subset:** niche={_sel_or_none(niche_w.value, NICHE_ALL) or 'ALL'} • plant={_sel_or_none(plant_w.value, PLANT_ALL) or 'ALL'}",
            f"**Groups:** `{gf}` • n_groups={len(uniq)} • n_samples={len(samples)}",
            f"**Permutations:** {perms if perms>0 else 0} (seed={seed})",
        ]
        res_bits = [
            f"**ANOSIM R:** `{R_obs:.4f}`",
            f"**p-value:** `{p_val:.4g}`" if p_val is not None else "**p-value:** *(not computed)*",
        ]
        if perms > 0:
            res_bits.append(f"**Valid permutations:** {valid}/{perms}")

        anosim_result_md.object = "\n\n".join(["\n".join(meta_bits), "\n".join(res_bits)])

        # Optional save
        if anosim_save_csv.value:
            try:
                out_dir = os.path.join(CACHE_DIR, "ANOSIM")
                os.makedirs(out_dir, exist_ok=True)
                tag = f"{level_w.value}_{gf}_{_sel_or_none(niche_w.value, NICHE_ALL) or 'ALL'}_{_sel_or_none(plant_w.value, PLANT_ALL) or 'ALL'}"
                tag = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in tag)[:120]
                out_csv = os.path.join(out_dir, f"anosim_{tag}.csv")
                out = pd.DataFrame([{
                    "level": str(level_w.value),
                    "niche": _sel_or_none(niche_w.value, NICHE_ALL) or "ALL",
                    "plant": _sel_or_none(plant_w.value, PLANT_ALL) or "ALL",
                    "group_factor": gf,
                    "n_samples": int(len(samples)),
                    "n_groups": int(len(uniq)),
                    "anosim_R": float(R_obs),
                    "p_value": float(p_val) if p_val is not None and np.isfinite(p_val) else np.nan,
                    "permutations": int(perms),
                    "valid_permutations": int(valid),
                    "seed": int(seed),
                }])
                out.to_csv(out_csv, index=False)
                anosim_csv_note.object = f"💾 Saved: `{out_csv}`"
            except Exception as e:
                anosim_csv_note.object = f"⚠️ Could not save ANOSIM CSV: {e}"

        anosim_status.object = "✅ ANOSIM complete."
        try:
            plot_area.active = 4  # Taxonomy, PCoA, Diversity, SIMPER, ANOSIM
        except Exception:
            pass

    except Exception as e:
        import io, traceback
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        anosim_status.object = f"⚠️ ANOSIM error: {e}"
        anosim_result_md.object = f"```pytb\n{buf.getvalue()}\n```"


anosim_btn.on_click(_compute_anosim)


def plot_pcoa_scatter(coords2: np.ndarray, samples: list[str], eigvals: np.ndarray):
    """
    Scatter plot of PCoA1 vs PCoA2 with optional color/shape by metadata.
    Legend size + icon scale controlled by widgets.
    """
    if coords2.shape[0] == 0:
        return pn.pane.Markdown("⚠️ No samples to plot (PCoA).")

    x = coords2[:, 0]
    y = coords2[:, 1] if coords2.shape[1] > 1 else np.zeros_like(x)

    # % explained from positive eigenvalues
    try:
        ev = np.array([v for v in eigvals if v > 0], dtype=float)
        denom = float(ev.sum()) if ev.size else 1.0
        p1 = 100.0 * float(ev[0] / denom) if ev.size >= 1 else 0.0
        p2 = 100.0 * float(ev[1] / denom) if ev.size >= 2 else 0.0
    except Exception:
        p1, p2 = 0.0, 0.0

    fig, ax = plt.subplots(figsize=(figure_width.value, figure_height.value))
    ax.set_xlabel(f"PCoA1 ({p1:.1f}%)")
    ax.set_ylabel(f"PCoA2 ({p2:.1f}%)")

    color_factor = group_factor.value  # follow taxonomy selection
    shape_factor = outer_group_factor.value  # follow taxonomy selection
    size = int(pcoa_point_size.value)
    alpha = float(pcoa_alpha.value)

    def _meta(sample, col):
        """Fetch metadata value for a sample, applying taxonomy binning to 'Other' when relevant."""
        if col == "(None)" or col not in metadata.columns:
            return None
        try:
            v = metadata.at[sample, col]
        except Exception:
            return None

        # Apply the SAME binning used in taxonomy plots (group_bin_other) when coloring by group_factor
        try:
            if str(col) == str(group_factor.value):
                bin_set = set(map(str, group_bin_other.value))
                sv = str(v)
                if sv in bin_set:
                    return "Other"
        except Exception:
            pass

        return v

    color_vals = [apply_group_binning(s, color_factor) for s in samples]
    shape_vals = [ _meta(s, shape_factor) for s in samples ]

    # Shapes
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H", "8"]
    shape_map = {}
    if shape_factor != "(None)" and shape_factor in metadata.columns:
        uniq = []
        for v in shape_vals:
            vv = str(v) if v is not None else "NA"
            if vv not in uniq:
                uniq.append(vv)
        for i, vv in enumerate(uniq):
            shape_map[vv] = markers[i % len(markers)]

    # Colors
    if color_factor == "(None)" or color_factor not in metadata.columns:
        ax.scatter(x, y, s=size, alpha=alpha, c=[pcoa_default_color.value]*len(samples), edgecolors="none")
    else:
        ser = pd.Series(color_vals)
        # numeric?
        is_numeric = False
        try:
            pd.to_numeric(ser, errors="raise")
            is_numeric = True
        except Exception:
            is_numeric = False

        if is_numeric:
            vals = pd.to_numeric(ser, errors="coerce").to_numpy(dtype=float)
            sc = ax.scatter(x, y, s=size, alpha=alpha, c=vals, edgecolors="none")
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label(str(color_factor))
        else:
            cats = []
            for v in color_vals:
                vv = str(v) if v is not None else "NA"
                if vv not in cats:
                    cats.append(vv)

            prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
            cycle_colors = prop_cycle.by_key().get("color", []) if prop_cycle else []
            col_map = {c: (cycle_colors[i % len(cycle_colors)] if cycle_colors else None) for i, c in enumerate(cats)}

            if shape_map:
                for ccat in cats:
                    for scat in shape_map.keys():
                        idx = [i for i,(cv,sv) in enumerate(zip(color_vals, shape_vals))
                               if (str(cv) if cv is not None else "NA")==ccat and (str(sv) if sv is not None else "NA")==scat]
                        if not idx:
                            continue
                        ax.scatter(
                            x[idx], y[idx],
                            s=size, alpha=alpha,
                            c=[col_map[ccat]]*len(idx),
                            marker=shape_map[scat],
                            label=f"{ccat} • {scat}" if len(shape_map)>1 else str(ccat),
                            edgecolors="none",
                        )
            else:
                for ccat in cats:
                    idx = [i for i,cv in enumerate(color_vals) if (str(cv) if cv is not None else "NA")==ccat]
                    ax.scatter(x[idx], y[idx], s=size, alpha=alpha, c=[col_map[ccat]]*len(idx), marker="o",
                               label=str(ccat), edgecolors="none")

            leg = ax.legend(
                loc="best",
                frameon=True,
                fontsize=int(pcoa_legend_text.value),
                markerscale=float(pcoa_legend_markerscale.value),
            )
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontfamily(str(legend_font_family.value))

    if pcoa_show_labels.value:
        for i, s in enumerate(samples):
            ax.text(x[i], y[i], str(s), fontsize=max(6, int(pcoa_legend_text.value)-2), ha="left", va="bottom")

    ax.axhline(0, linewidth=0.5, alpha=0.4)
    ax.axvline(0, linewidth=0.5, alpha=0.4)
    fig.tight_layout()

    # Save PCoA plot to cache (handy for debugging / exporting)
    try:
        pcoa_dir = os.path.join(CACHE_DIR, "PCoA")
        os.makedirs(pcoa_dir, exist_ok=True)
        # simple stable-ish name from current selection
        tag = f"{level_w.value}_{subset_mode.value}_{TopN.value}_{group_factor.value}_{outer_group_factor.value}"
        tag = "".join(c if c.isalnum() or c in ("-","_") else "_" for c in tag)[:120]
        out_png = os.path.join(pcoa_dir, f"pcoa_{tag}.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    except Exception:
        pass

    return pn.pane.Matplotlib(fig, tight=True)


def render_pcoa(_tick=0):
    if not data_loaded.value:
        return pn.pane.Markdown("👋 Load data first. Then click **Run PCoA**.")
    try:
        P = _build_pcoa_input_matrix()
        if P is None or P.empty:
            pcoa_status.object = "⚠️ PCoA: no samples after filters."
            return pn.pane.Markdown("⚠️ No samples to ordinate (check filters).")

        X = P.T.to_numpy(dtype=float)  # samples x taxa
        D = _bray_curtis_distance_matrix(X)
        coords, eigvals = _pcoa_from_distance(D)
        coords2 = coords[:, :2] if coords.shape[1] >= 2 else coords
        samples = list(P.columns)
        pcoa_status.object = f"✅ PCoA computed • samples={len(samples)} • taxa={P.shape[0]}"
        return plot_pcoa_scatter(coords2, samples, eigvals)
    except Exception as e:
        import io, traceback
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        pcoa_status.object = f"⚠️ PCoA error: {e}"
        return pn.pane.Markdown(f"```pytb\n{buf.getvalue()}\n```", height=300, sizing_mode="stretch_width")


pcoa_pane = pn.bind(render_pcoa, _tick=pcoa_btn.param.clicks)

# ==============================
# Render + export
# ==============================
def _cache_key(A: pd.DataFrame, legend_cats: list) -> str:
    parts = [
        f"level={level_w.value}",
        f"niche={_sel_or_none(niche_w.value, NICHE_ALL)}",
        f"plant={_sel_or_none(plant_w.value, PLANT_ALL)}",
        f"mode={'sample' if plot_mode.value=='Per-sample stacked' else 'avg'}",
        f"subset={subset_mode.value}",
        f"topN={TopN.value}",
        f"topNScope={topN_scope.value}",
        f"others={int(include_others.value)}",
        f"group={group_factor.value}",
        f"outer={outer_group_factor.value}",
        f"xord={x_order_mode.value}",
        f"taxord={taxa_order_mode.value}",
        f"sorttax={sort_taxon.value}",
        f"scope={sort_scope.value}",
        f"desc={int(sort_desc.value)}",
        f"gap={group_gap.value}",
        f"manX={';'.join(manual_x_sequence)}",
        f"manTaxa={';'.join(manual_taxa_sequence)}",
        f"bin={';'.join(sorted(group_bin_other.value))}",
        f"filterMode={group_filter_mode.value}",
        f"filterSel={';'.join(map(str, group_filter_select.value))}",
        f"lite={int(lite_mode.value)}",
        f"gbShow={int(show_group_boundaries.value)}",
        f"gbWidth={int(group_boundary_width.value)}",
        f"gbColor={group_boundary_color.value}",
        f"xGroupSrc={x_group_source.value}",
        f"outerFilterMode={outer_filter_mode.value}",
        f"outerFilterSel={';'.join(map(str, outer_filter_select.value))}",
    ]
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()


def _compute_correlation(event=None):
    """Compute Pearson correlation between exactly two taxa from the final matrix A.

    Behaviour:
      - If 'Manual subset' is active, uses the taxa selected there.
      - Otherwise, if the final matrix has exactly 2 taxa, uses those.
    """
    try:
        A, legend_cats = build_aggregated_matrix()
    except Exception as e:
        corr_result.object = f"⚠️ Error building matrix: {e}"
        return

    if A is None or A.empty:
        corr_result.object = "⚠️ No data available for correlation."
        PLOT_STATE["corr_annot"] = None
        return

    # Decide which taxa to correlate
    sel = []
    if subset_mode.value == "Manual subset":
        # Intersect user selection with actually present taxa
        sel = [str(t) for t in manual_taxa.value if str(t) in A.index]

    # If not in manual mode or selection is empty, but we only have 2 taxa, use those
    if (not sel) and (len(A.index) == 2):
        sel = list(A.index.astype(str))

    if len(sel) != 2:
        corr_result.object = "⚠️ Please ensure **exactly two taxa** are selected (or present) for correlation."
        PLOT_STATE["corr_annot"] = None
        return

    t1, t2 = sel[0], sel[1]

    try:
        x = A.loc[t1].astype(float).to_numpy()
        y = A.loc[t2].astype(float).to_numpy()
    except KeyError as e:
        corr_result.object = f"⚠️ Selected taxa not found in matrix: {e}"
        return

    if x.size < 3:
        corr_result.object = "⚠️ Not enough samples (need at least 3) for correlation."
        return

    # Compute Pearson r without external dependencies
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    x_dev = x - x_mean
    y_dev = y - y_mean
    num = float((x_dev * y_dev).sum())
    den = float(np.sqrt((x_dev * x_dev).sum() * (y_dev * y_dev).sum()))

    if den == 0.0:
        corr_result.object = "⚠️ Zero variance for at least one taxon; correlation undefined."
        PLOT_STATE["corr_annot"] = None
        return

    r = num / den
    PLOT_STATE["corr_annot"] = f"r = {r:.3f}"

    corr_result.object = (
        f"### Correlation between **{t1}** and **{t2}**\n"
        f"- **Pearson r = {r:.3f}**"
    )

    # Optionally annotate the current plot
    ax = _last_fig.get("ax")
    fig = _last_fig.get("fig")
    if show_corr_on_plot.value and ax is not None:
        # Clear any previous text in top-left by over-plotting; keep it simple
        ax.text(
            0.02,
            0.98,
            f"r = {r:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="red",
        )
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
def render_plot(_tick=0):
    if not data_loaded.value:
        return pn.pane.Markdown(
            "👋 Choose Level/Niche/Plant, (optionally enable **Lite mode** for ASV), "
            "click **Load data**, then **Plot**."
        )

    try:
        A, legend_cats = build_aggregated_matrix()
    finally:
        status_spinner.value = False

    try:
        _set_taxa_domain(list(A.index.astype(str)))

        gf = group_factor.value
        of = outer_group_factor.value

        # -------- group_bin_other + filter options based on group_factor --------
        if plot_mode.value == "Per-group average" and gf != "(None)":
            # columns already represent groups
            orig_labels = list(A.columns)
        elif gf != "(None)":
            # per-sample: derive labels from metadata
            samples = [s for s in A.columns if s in metadata.index]
            if samples:
                grp_labels = metadata.loc[samples, gf].astype(str)
                orig_labels = list(pd.Categorical(grp_labels).categories)
            else:
                orig_labels = []
        else:
            orig_labels = []

        group_bin_other.options = sorted({lab for lab in orig_labels if lab != "Other"})

        bin_set = set(group_bin_other.value)

        # group-factor labels after binning ("Other") → used for filter widget
        if gf != "(None)" and orig_labels:
            if plot_mode.value == "Per-group average":
                gf_labels_all, seen = [], set()
                for lab in orig_labels:
                    lbl = "Other" if lab in bin_set else lab
                    if lbl not in seen:
                        seen.add(lbl)
                        gf_labels_all.append(lbl)
            else:
                samples = [c for c in A.columns if c in metadata.index]
                gf_labels_all, seen = [], set()
                for s in samples:
                    lbl = _safe_group_label(s, gf, bin_set)
                    if lbl not in seen:
                        seen.add(lbl)
                        gf_labels_all.append(lbl)
        else:
            gf_labels_all = list(A.columns)

        # group filter choices always refer to group_factor (after binning)
        group_filter_select.options = gf_labels_all

        # -------- outer-group filter options based on outer_group_factor --------
        if (
            of != "(None)"
            and of in metadata.columns
            and plot_mode.value == "Per-sample stacked"
        ):
            samples = [c for c in A.columns if c in metadata.index]
            if samples:
                of_labels = metadata.loc[samples, of].astype(str)
                outer_labels_all = list(pd.Categorical(of_labels).categories)
            else:
                outer_labels_all = []
        else:
            outer_labels_all = []

        outer_filter_select.options = outer_labels_all




        # -------- X-manual domain based on chosen source --------
        src = x_group_source.value

        if plot_mode.value == "Per-group average":
            # In average mode, columns are groups; outer group doesn't apply.
            # "Samples" is not meaningful here → fall back to group labels.
            eff_labels_manual = list(A.columns)
        else:
            cols = [c for c in A.columns if c in metadata.index]

            if src == "Samples" or (gf == "(None)" and src == "Group / x-axis factor"):
                # Domain is directly sample IDs
                eff_labels_manual = [str(c) for c in cols]

            elif src == "Outer group" and of != "(None)" and of in metadata.columns:
                # Domain is levels of outer_group_factor in the current samples
                labs = metadata.loc[cols, of].astype(str)
                eff_labels_manual = list(pd.Categorical(labs).categories)

            elif src == "Group / x-axis factor" and gf != "(None)" and gf in metadata.columns:
                # Domain is group-factor labels (already binned)
                eff_labels_manual = gf_labels_all

            else:
                # Fallback: samples
                eff_labels_manual = [str(c) for c in cols]

        _set_manual_x_domain(eff_labels_manual)

        sort_taxon.options = ["(None)"] + list(A.index.astype(str))
        anova_taxa.options = list(A.index.astype(str))




    except Exception:
        pass

    if CURRENT_REL_FILE:
        suffix = " • lite (top 100 + Others)" if level_w.value == "ASV" and lite_mode.value else ""
        status_text.object = (
            f"**Plotted:** {os.path.basename(CURRENT_REL_FILE)} • columns={A.shape[1]}{suffix}"
        )
    else:
        status_text.object = f"**Plotted:** columns={A.shape[1]}"

    try:
        key = _cache_key(A, legend_cats)
        csv_path = os.path.join(CACHE_DIR, f"{key}.csv")
        png_path = os.path.join(CACHE_DIR, f"{key}.png")
        A.to_csv(csv_path)
        last_png_path["path"] = png_path
    except Exception:
        last_png_path["path"] = None

    try:
        if plot_mode.value == "Per-sample stacked":
            pane = plot_stacked_per_sample(A, legend_cats)
        else:
            pane = plot_group_average(A, legend_cats)
        return pane
    except Exception as e:
        import io, traceback

        buf = io.StringIO()
        traceback.print_exc(file=buf)
        status_text.object = f"**Plot error:** {e}"
        return pn.pane.Markdown(
            f"```pytb\n{buf.getvalue()}\n```",
            height=300,
            sizing_mode="stretch_width",
        )


def _export_png(event=None):
    fig = _last_fig.get("fig")
    if fig is None:
        status_text.object = "⚠️ No plot to export. Click **Plot** first."
        return
    try:
        A, legend_cats = build_aggregated_matrix()
        key = _cache_key(A, legend_cats)
        png_path = os.path.join(CACHE_DIR, f"{key}.png")

        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        last_png_path["path"] = png_path
        status_text.object = f"**Exported PNG:** {os.path.basename(png_path)}"
    except Exception as e:
        status_text.object = f"**Export failed:** {e}"


export_btn.on_click(_export_png)
corr_btn.on_click(_compute_correlation)
pcoa_btn.on_click(lambda e: None)  # triggers pcoa_pane via clicks

# ==============================
# ANOVA / tests (per-taxon) + compact letters
# ==============================
def _compact_letter_display(groups, reject_matrix):
    """Return dict group->letters from Tukey reject decisions."""
    letters = []
    assign = {g: "" for g in groups}

    def conflict(g, letter):
        for h in groups:
            if h == g:
                continue
            if letter not in assign[h]:
                continue
            i = groups.index(g); j = groups.index(h)
            a, b = (i, j) if i < j else (j, i)
            if reject_matrix.get((a, b), False):
                return True
        return False

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for g in groups:
        placed_any = False
        for letter in letters:
            if not conflict(g, letter):
                assign[g] += letter
                placed_any = True
        if not placed_any:
            letter = alphabet[len(letters)] if len(letters) < len(alphabet) else f"l{len(letters)+1}"
            letters.append(letter)
            assign[g] += letter

    return assign


def _compute_anova(event=None):
    gf = group_factor.value
    if gf == "(None)" or gf not in metadata.columns:
        anova_result.object = "⚠️ Please set **Group / x-axis factor** to your condition column first."
        return

    # Build per-sample matrix (ANOVA needs replicates)
    _pm = plot_mode.value
    try:
        if _pm != "Per-sample stacked":
            plot_mode.value = "Per-sample stacked"
        A, _ = build_aggregated_matrix()
    finally:
        try:
            plot_mode.value = _pm
        except Exception:
            pass

    if A is None or A.empty:
        anova_result.object = "⚠️ No data available for ANOVA."
        return

    taxa_all = list(A.index.astype(str))
    taxa_sel = list(anova_taxa.value) if anova_taxa.value else []
    if not taxa_sel:
        if len(taxa_all) == 1:
            taxa_sel = taxa_all
        else:
            anova_result.object = "⚠️ Select one or more taxa in **Taxa for ANOVA (if >1)**."
            return

    import pandas as pd
    samples = list(A.columns)

    alpha = float(anova_alpha.value) if anova_alpha.value else 0.05
    study_col = anova_study_col.value
    use_study = (anova_replicates.value.startswith("Studies")) and (study_col in metadata.columns) and (study_col != "(None)")

    # map sample -> group (+study)
    rows = []
    for s in samples:
        if s not in metadata.index:
            continue
        rows.append({
            "sample": s,
            "group": str(metadata.at[s, gf]),
            "study": str(metadata.at[s, study_col]) if use_study else None
        })
    meta_map = pd.DataFrame(rows)
    if meta_map.empty:
        anova_result.object = "⚠️ Could not map samples to metadata for the chosen grouping."
        return

    out = []
    for tax in taxa_sel:
        if tax not in A.index:
            continue
        vals = A.loc[tax].astype(float).rename("value").reset_index()
        vals.columns = ["sample", "value"]
        D = meta_map.merge(vals, on="sample", how="inner").dropna(subset=["group", "value"])

        if use_study:
            D = D.groupby(["study", "group"], as_index=False)["value"].mean()

        groups = sorted(D["group"].unique())
        if len(groups) < 2:
            out.append(f"### {tax}\n⚠️ Need at least 2 groups.\n")
            continue

        try:
            import numpy as np
            from scipy import stats
        except Exception as e:
            out.append(f"### {tax}\n⚠️ scipy missing for stats: {e}\n")
            continue

        arrs = [D.loc[D["group"] == g, "value"].values for g in groups]

        if len(groups) == 2:
            t, p = stats.ttest_ind(arrs[0], arrs[1], equal_var=False, nan_policy="omit")
            same = (p >= alpha)
            letters = {groups[0]: "a", groups[1]: ("a" if same else "b")}
            # cache letters for plot annotation
            try:
                ANOVA_LETTERS[(str(tax), str(gf))] = {str(k): str(v) for k, v in letters.items()}
            except Exception:
                pass
            tab = pd.DataFrame({
                "group": groups,
                "mean": [float(np.nanmean(a)) for a in arrs],
                "n": [int(len(a)) for a in arrs],
                "letter": [letters[g] for g in groups]
            })
            out.append(
                f"### {tax}\n*Test:* Welch t-test (alpha={alpha}) • p={p:.3g}\n\n"
                + tab.to_string(index=False) + "\n"
            )
            continue

        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols
            from statsmodels.stats.multicomp import MultiComparison
        except Exception as e:
            out.append(f"### {tax}\n⚠️ statsmodels missing for Tukey: {e}\n")
            continue

        tmp = D.copy()
        tmp["group"] = tmp["group"].astype(str)
        model = ols("value ~ C(group)", data=tmp).fit()
        anova_tbl = sm.stats.anova_lm(model, typ=2)
        p_anova = float(anova_tbl["PR(>F)"].iloc[0]) if "PR(>F)" in anova_tbl.columns else float("nan")

        mc = MultiComparison(tmp["value"], tmp["group"])
        tuk = mc.tukeyhsd(alpha=alpha)

        g_order = list(map(str, tuk.groupsunique))
        # reject matrix from Tukey table
        reject = {}
        idx_map = {g:i for i,g in enumerate(g_order)}
        for r in tuk.summary().data[1:]:
            g1, g2, *_rest, rej = r
            i, j = idx_map[str(g1)], idx_map[str(g2)]
            a, b = (i, j) if i < j else (j, i)
            reject[(a, b)] = bool(rej)

        letters = _compact_letter_display(g_order, reject)
        # cache letters for plot annotation
        try:
            ANOVA_LETTERS[(str(tax), str(gf))] = {str(k): str(v) for k, v in letters.items()}
        except Exception:
            pass

        import numpy as np
        tab = pd.DataFrame({
            "group": g_order,
            "mean": [float(np.nanmean(tmp.loc[tmp["group"]==g,"value"].values)) for g in g_order],
            "n": [int((tmp["group"]==g).sum()) for g in g_order],
            "letter": [letters.get(g,"") for g in g_order]
        })

        out.append(
            f"### {tax}\n*ANOVA:* p={p_anova:.3g} (alpha={alpha}) • *Post-hoc:* Tukey letters\n\n"
            + tab.to_string(index=False) + "\n"
        )

    anova_result.object = "\n\n".join(out) if out else "⚠️ Nothing to report."

anova_btn.on_click(_compute_anova)
plot_pane = pn.bind(render_plot, _tick=plot_btn.param.clicks)


# ==============================
# Live style (no replot)
# ==============================
def _apply_style_only(event=None):
    fig = _last_fig.get("fig")
    ax = _last_fig.get("ax")
    if fig is None or ax is None:
        status_text.object = "⚠️ No plot to style. Click Plot first."
        return
    for lab in ax.get_xticklabels():
        lab.set_fontsize(int(x_label_fontsize.value))
        lab.set_rotation(int(x_label_rotation.value))
        lab.set_ha("right" if int(x_label_rotation.value) else "center")
    ax.tick_params(axis="y", labelsize=int(y_label_fontsize.value))
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(int(legend_text_size.value))
            t.set_fontfamily(str(legend_font_family.value))
    try:
        _apply_abundance_axis_limits(ax)
    except Exception:
        pass

    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
    status_text.object = "✓ Style updated (no replot)."


apply_style_btn.on_click(_apply_style_only)


# ==============================
# Preset save/load helpers
# ==============================
def _preset_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    if not safe:
        safe = "default"
    return os.path.join(SETTINGS_DIR, f"taxonomy_smart_{safe}.json")


def _refresh_preset_list():
    """Scan SETTINGS_DIR and populate preset_select.options with preset names."""
    pattern = os.path.join(SETTINGS_DIR, "taxonomy_smart_*.json")
    files = sorted(glob.glob(pattern))
    names = []
    for f in files:
        base = os.path.basename(f)
        if base.startswith("taxonomy_smart_") and base.endswith(".json"):
            core = base[len("taxonomy_smart_") : -len(".json")]
            if core:
                names.append(core)
    if names:
        preset_select.options = names
        # keep current selection if possible
        if preset_select.value not in names:
            preset_select.value = names[0]
    else:
        preset_select.options = []
        preset_select.value = None


def _on_preset_select_change(event):
    """When user picks a preset from the dropdown, update the text input."""
    val = event.new
    if val:
        preset_name_input.value = val


preset_select.param.watch(_on_preset_select_change, "value")


def _save_preset(event=None):
    name = preset_name_input.value.strip() or "default"
    path = _preset_path(name)
    data = {
        "level": level_w.value,
        "niche": niche_w.value,
        "plant": plant_w.value,
        "lite_mode": bool(lite_mode.value),
        "plot_mode": plot_mode.value,
        "subset_mode": subset_mode.value,
        "TopN": int(TopN.value),
        "topN_scope": topN_scope.value,
        "manual_taxa": list(manual_taxa.value),
        "manual_taxa_list_sort": manual_taxa_list_sort.value,
        "include_others": bool(include_others.value),
        "tax_filter_enable": bool(tax_filter_enable.value),
        "tax_filter_rank": tax_filter_rank.value,
        "tax_filter_values": list(tax_filter_values.value),
        "group_factor": group_factor.value,
        "outer_group_factor": outer_group_factor.value,
        "group_gap": int(group_gap.value),
        "group_bin_other": list(group_bin_other.value),
        "group_filter_mode": group_filter_mode.value,
        "group_filter_select": list(group_filter_select.value),
        "x_order_mode": x_order_mode.value,
        "manual_x_sequence": list(manual_x_sequence),
        "taxa_order_mode": taxa_order_mode.value,
        "manual_taxa_sequence": list(manual_taxa_sequence),
        "sort_taxon": sort_taxon.value,
        "sort_scope": sort_scope.value,
        "sort_desc": bool(sort_desc.value),
        "show_n_per_bar": bool(show_n_per_bar.value),
        "n_display_mode": str(n_display_mode.value),
        "show_se": bool(show_se.value),
        "show_legend": bool(show_legend.value),
        "legend_text_size": int(legend_text_size.value),
        "legend_font_family": legend_font_family.value,
        "legend_style_preset": legend_style_preset.value,
        "x_label_rotation": int(x_label_rotation.value),
        "outer_x_label_rotation": int(outer_x_label_rotation.value),
        "x_label_fontsize": int(x_label_fontsize.value),
        "y_label_fontsize": int(y_label_fontsize.value),
        "y_axis_mode": str(y_axis_mode.value),
        "y_axis_max": float(y_axis_max.value),
        "show_sample_labels": bool(show_sample_labels.value),
        "show_group_boundaries": bool(show_group_boundaries.value),
        "group_boundary_width": int(group_boundary_width.value),
        "group_boundary_color": group_boundary_color.value,
        "group_boundary_source": group_boundary_source.value,
        # --- SIMPER ---
        "simper_factor": simper_factor.value,
        "simper_condition_pool": list(simper_condition_pool.value),
        "simper_groupA": list(simper_groupA.value),
        "simper_groupB": list(simper_groupB.value),
        "simper_top_n": int(simper_top_n.value),
        "simper_save_csv": bool(simper_save_csv.value),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        preset_status.object = f"✅ Preset **{name}** saved to {os.path.basename(path)}"
        _refresh_preset_list()
        preset_select.value = name
    except Exception as e:
        preset_status.object = f"⚠️ Error saving preset **{name}**: {e}"


def _load_preset(event=None):
    name = preset_name_input.value.strip() or "default"
    path = _preset_path(name)
    if not os.path.exists(path):
        preset_status.object = f"⚠️ Preset **{name}** not found in ./ui_presets."
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        preset_status.object = f"⚠️ Error reading preset **{name}**: {e}"
        return

    # Apply safely, checking options where relevant
    if "level" in data and data["level"] in level_w.options:
        level_w.value = data["level"]
    if "niche" in data and data["niche"] in niche_w.options:
        niche_w.value = data["niche"]
    if "plant" in data and data["plant"] in plant_w.options:
        plant_w.value = data["plant"]

    lite_mode.value = bool(data.get("lite_mode", lite_mode.value))

    if "plot_mode" in data and data["plot_mode"] in plot_mode.options:
        plot_mode.value = data["plot_mode"]
    if "subset_mode" in data and data["subset_mode"] in subset_mode.options:
        subset_mode.value = data["subset_mode"]
    if "TopN" in data:
        TopN.value = int(data["TopN"])
    if "topN_scope" in data and data["topN_scope"] in topN_scope.options:
        topN_scope.value = data["topN_scope"]

    if "manual_taxa" in data:
        # intersect with available options
        mt = [t for t in data["manual_taxa"] if t in manual_taxa.options]
        manual_taxa.value = mt

        if "manual_taxa_list_sort" in data and data["manual_taxa_list_sort"] in manual_taxa_list_sort.options:
            manual_taxa_list_sort.value = data["manual_taxa_list_sort"]

    include_others.value = bool(data.get("include_others", include_others.value))

    # New: taxonomy filter
    tax_filter_enable.value = bool(data.get("tax_filter_enable", tax_filter_enable.value))
    if "tax_filter_rank" in data:
        try:
            _update_tax_filter_rank_options()
            if data["tax_filter_rank"] in tax_filter_rank.options:
                tax_filter_rank.value = data["tax_filter_rank"]
        except Exception:
            pass
    if "tax_filter_values" in data:
        # Will be intersected with available options after data is loaded
        try:
            global _PENDING_TAX_FILTER_VALUES
            _PENDING_TAX_FILTER_VALUES = list(data["tax_filter_values"])
        except Exception:
            _PENDING_TAX_FILTER_VALUES = []

    if "group_factor" in data and data["group_factor"] in group_factor.options:
        group_factor.value = data["group_factor"]
    if "outer_group_factor" in data and data["outer_group_factor"] in outer_group_factor.options:
        outer_group_factor.value = data["outer_group_factor"]
    if "group_gap" in data:
        group_gap.value = int(data["group_gap"])

    if "group_bin_other" in data:
        gbo = [g for g in data["group_bin_other"] if g in group_bin_other.options]
        group_bin_other.value = gbo

    if "group_filter_mode" in data and data["group_filter_mode"] in group_filter_mode.options:
        group_filter_mode.value = data["group_filter_mode"]
    if "group_filter_select" in data:
        gfs = [g for g in data["group_filter_select"] if g in group_filter_select.options]
        group_filter_select.value = gfs

    if "x_order_mode" in data and data["x_order_mode"] in x_order_mode.options:
        x_order_mode.value = data["x_order_mode"]

    # sequences: keep even if domain not yet set; they will be filtered later
    global manual_x_sequence, manual_taxa_sequence
    manual_x_sequence = list(data.get("manual_x_sequence", manual_x_sequence))
    manual_taxa_sequence = list(data.get("manual_taxa_sequence", manual_taxa_sequence))
    _refresh_manual_x_lists()
    _refresh_taxa_lists()

    if "taxa_order_mode" in data and data["taxa_order_mode"] in taxa_order_mode.options:
        taxa_order_mode.value = data["taxa_order_mode"]

    if "sort_taxon" in data and data["sort_taxon"] in sort_taxon.options:
        sort_taxon.value = data["sort_taxon"]
    if "sort_scope" in data and data["sort_scope"] in sort_scope.options:
        sort_scope.value = data["sort_scope"]
    sort_desc.value = bool(data.get("sort_desc", sort_desc.value))
    show_n_per_bar.value = bool(data.get("show_n_per_bar", show_n_per_bar.value))
    if "n_display_mode" in data and data["n_display_mode"] in n_display_mode.options:
        n_display_mode.value = data["n_display_mode"]
    show_se.value = bool(data.get("show_se", show_se.value))

    show_legend.value = bool(data.get("show_legend", show_legend.value))
    if "legend_text_size" in data:
        legend_text_size.value = int(data["legend_text_size"])
    if "legend_font_family" in data and data["legend_font_family"] in legend_font_family.options:
        legend_font_family.value = data["legend_font_family"]
    if "legend_style_preset" in data and data["legend_style_preset"] in legend_style_preset.options:
        legend_style_preset.value = data["legend_style_preset"]

    if "x_label_rotation" in data:
        x_label_rotation.value = int(data["x_label_rotation"])
    if "outer_x_label_rotation" in data:
        outer_x_label_rotation.value = int(data["outer_x_label_rotation"])
    if "x_label_fontsize" in data:
        x_label_fontsize.value = int(data["x_label_fontsize"])
    if "y_label_fontsize" in data:
        y_label_fontsize.value = int(data["y_label_fontsize"])
    if "y_axis_mode" in data and data["y_axis_mode"] in y_axis_mode.options:
        y_axis_mode.value = data["y_axis_mode"]
    if "y_axis_max" in data:
        try:
            y_axis_max.value = float(data["y_axis_max"])
        except Exception:
            pass

    show_sample_labels.value = bool(data.get("show_sample_labels", show_sample_labels.value))

    # New: group boundary settings
    show_group_boundaries.value = bool(data.get("show_group_boundaries", show_group_boundaries.value))
    if "group_boundary_width" in data:
        group_boundary_width.value = int(data["group_boundary_width"])
    if "group_boundary_color" in data:
        group_boundary_color.value = data["group_boundary_color"]
    if "group_boundary_source" in data and data["group_boundary_source"] in group_boundary_source.options:
        group_boundary_source.value = data["group_boundary_source"]


    # --- SIMPER (apply if widgets exist) ---
    if "simper_factor" in data and data["simper_factor"] in simper_factor.options:
        simper_factor.value = data["simper_factor"]
    if "simper_condition_pool" in data:
        # pool options are populated dynamically; set value after sync
        try:
            simper_condition_pool.value = list(map(str, data["simper_condition_pool"]))
        except Exception:
            pass
    if "simper_groupA" in data:
        try:
            simper_groupA.value = list(map(str, data["simper_groupA"]))
        except Exception:
            pass
    if "simper_groupB" in data:
        try:
            simper_groupB.value = list(map(str, data["simper_groupB"]))
        except Exception:
            pass
    if "simper_top_n" in data:
        try:
            simper_top_n.value = int(data["simper_top_n"])
        except Exception:
            pass
    if "simper_save_csv" in data:
        simper_save_csv.value = bool(data["simper_save_csv"])

    try:
        _simper_sync_widgets()
    except Exception:
        pass
    preset_status.object = f"✅ Preset **{name}** loaded. Click **Load data** (if needed) and **Plot**."


save_preset_btn.on_click(_save_preset)
load_preset_btn.on_click(_load_preset)

# Initial scan for presets at startup
_refresh_preset_list()


# ==============================
# Layout
# ==============================
header = pn.Row(hide_menus, pn.Spacer(width=12), export_mode, export_size)

source_card = pn.Column(
    pn.Row(level_w, niche_w, plant_w),
    path_info,
    pn.Row(load_btn, lite_mode, pn.pane.Markdown("← click after choosing subset")),
    pn.Row(status_spinner, status_text),
)

manual_x_editor = pn.Column(
    pn.pane.Markdown("**Manual X order editor**"),
    pn.Row(
        manual_x_available,
        pn.Column(
            btn_x_add,
            btn_x_remove,
            btn_x_up,
            btn_x_down,
            btn_x_top,
            btn_x_bottom,
            btn_x_clear,
            sizing_mode="fixed",
        ),
        manual_x_selected,
    ),
)

manual_taxa_editor = pn.Column(
    pn.pane.Markdown("**Manual TAXA (Y) order editor**"),
    pn.Row(
        manual_taxa_available,
        pn.Column(
            btn_t_add,
            btn_t_remove,
            btn_t_up,
            btn_t_down,
            btn_t_top,
            btn_t_bottom,
            btn_t_clear,
            sizing_mode="fixed",
        ),
        manual_taxa_selected,
    ),
)


# --- Rebuilt: selector_card as collapsible sections (prevents widget overlap) ---
def _wrap_col(*objs):
    return pn.Column(*objs, sizing_mode="stretch_width")

selector_card = pn.Card(
    pn.Accordion(
        ("Taxa selection",
            _wrap_col(
                pn.Row(plot_mode, subset_mode, sizing_mode="stretch_width"),
                pn.Row(show_n_per_bar, n_count_basis, n_show_both, n_display_mode, sizing_mode="stretch_width"),
                pn.Row(show_se, sizing_mode="stretch_width"),
                pn.Row(TopN, include_others, topN_scope, sizing_mode="stretch_width"),
                pn.Row(tax_filter_enable, tax_filter_rank, sizing_mode="stretch_width"),
                pn.Row(tax_filter_values, sizing_mode="stretch_width"),
                tax_filter_note,
                pn.Row(manual_taxa_list_sort, sizing_mode="stretch_width"),
                manual_taxa,
            )
        ),
        ("Grouping & filters",
            _wrap_col(
                pn.Row(group_factor, sizing_mode="stretch_width"),
                pn.Row(outer_group_factor, group_gap, sizing_mode="stretch_width"),
                pn.Row(group_bin_other, sizing_mode="stretch_width"),
                pn.Row(group_filter_mode, sizing_mode="stretch_width"),
                group_filter_select,
                pn.Row(outer_filter_mode, sizing_mode="stretch_width"),
                outer_filter_select,
            )
        ),
        ("X-axis ordering",
            _wrap_col(
                pn.Row(x_order_mode, x_group_order_mode, x_group_source, sizing_mode="stretch_width"),
                manual_x_editor,
            )
        ),
        ("Y-axis ordering",
            _wrap_col(
                pn.Row(taxa_order_mode, sizing_mode="stretch_width"),
                manual_taxa_editor,
            )
        ),
        ("Sort",
            _wrap_col(
                pn.Row(sort_taxon, sort_scope, sort_desc, sizing_mode="stretch_width"),
            )
        ),
        active=[0],
    ),
    title="Taxa & Grouping",
    collapsed=False,
)

legend_position = pn.widgets.Select(
    name="Legend position",
    options=[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
        # Outside positions
        "outside right",
        "outside left",
        "above",
        "below",
    ],
    value="upper right",
)


legend_card = pn.Card(
    pn.Row(show_legend, legend_text_size, legend_font_family, legend_style_preset),
    pn.Row(legend_position),
    title="Legend & Colors",
    collapsed=False,
)

style_card = pn.Card(
    pn.pane.Markdown("### Style (instant, no replot)"),
    pn.Row(
        x_label_rotation,
        outer_x_label_rotation,
        x_label_fontsize,
        y_label_fontsize,
    ),
    pn.Row(y_axis_mode, y_axis_max),
    pn.Row(figure_width, figure_height, plot_orientation, flip_axes),
    pn.Row(show_sample_labels),
    pn.Row(
        show_group_boundaries,
        group_boundary_source,
        group_boundary_width,
        group_boundary_color,
    ),
    pn.Row(apply_style_btn),
    title="Live Style",
    collapsed=False,
)
presets_card = pn.Card(
    pn.pane.Markdown("### Settings presets"),
    pn.Row(preset_name_input),
    pn.Row(preset_select),
    pn.Row(save_preset_btn, load_preset_btn),
    preset_status,
    title="Presets",
    collapsed=False,
)


pcoa_card = pn.Card(
    pn.Accordion(
        ("Run / status",
            pn.Column(
                pn.Row(pcoa_btn),
                pcoa_status,
            )
        ),
        ("Aesthetics",
            pn.Column(
                pn.Row(pcoa_point_size, pcoa_alpha),
                pn.Row(pcoa_default_color, pcoa_show_labels),
            )
        ),
        ("Grouping (color / shape)",
            pn.Column(
                pn.pane.Markdown("Uses **Group / x-axis factor** for color and **Outer group** for shape (from the taxonomy menu)."),
            )
        ),
        ("Legend",
            pn.Column(
                pn.Row(pcoa_legend_text, pcoa_legend_markerscale),
            )
        ),
        active=[0],
    ),
    title="PCoA",
    collapsed=False,
)


diversity_card = pn.Card(
    pn.Accordion(
        ("Run / status", pn.Column(pn.Row(div_btn), div_status)),
        ("Summary", pn.Column(div_summary_md)),
        ("Per-group metrics", pn.Column(div_table_groups)),
        ("Between-group centroid distances", pn.Column(div_table_pairs)),
        active=[0],
    ),
    title="Diversity / differences",
    collapsed=False,
)

simper_card = pn.Card(
    pn.Accordion(
        ("Compare (A vs B)",
            pn.Column(
                pn.Row(simper_factor),
                simper_condition_pool,
                pn.Row(simper_groupA),
                pn.Row(simper_groupB),
                pn.Row(simper_top_n, simper_save_csv, simper_run_btn),
                simper_status,
                simper_csv_note,
            )
        ),
        ("Results (top-N)", pn.Column(simper_table)),
        active=[0],
    ),
    title="SIMPER",
    collapsed=False,
)



anosim_card = pn.Card(
    pn.Accordion(
        ("Run / status",
            pn.Column(
                pn.Row(anosim_btn),
                pn.Row(anosim_perms, anosim_seed, anosim_save_csv),
                anosim_status,
                anosim_csv_note,
            )
        ),
        ("Results", pn.Column(anosim_result_md)),
        ("Group sizes", pn.Column(anosim_groups_df)),
        active=[0],
    ),
    title="ANOSIM",
    collapsed=False,
)


menus_container = pn.Accordion(
    ("Data & actions",
        pn.Column(
            source_card,
            pn.Row(plot_btn, export_btn, corr_btn, anova_btn, pcoa_btn, anosim_btn),
            pn.Row(show_corr_on_plot),
            corr_result,
            sizing_mode="stretch_width",
        )
    ),
    ("Taxa & grouping", selector_card),
    ("Legend & colours", legend_card),
    ("Style", style_card),
    ("Presets", presets_card),
    ("Diversity / differences", diversity_card),
    ("SIMPER", simper_card),
    ("ANOSIM", anosim_card),
    ("PCoA", pcoa_card),
    ("ANOVA / post-hoc",
        pn.Column(
            anova_taxa,
            pn.Row(anova_replicates, anova_study_col, anova_alpha),
            anova_result,
            show_anova_letters,
            remove_spines,
            sizing_mode="stretch_width",
        )
    ),
    active=[0],
    sizing_mode="stretch_width",
)

div_pane = pn.Column(
    pn.pane.Markdown("### Diversity / differences"),
    div_status,
    pn.layout.Divider(),
    div_summary_md,
    pn.layout.Divider(),
    pn.pane.Markdown("#### Per-group metrics"),
    div_table_groups,
    pn.pane.Markdown("#### Between-group centroid distances"),
    div_table_pairs,
    sizing_mode="stretch_both",
)

simper_pane = pn.Column(
    pn.pane.Markdown("### SIMPER"),
    simper_status,
    pn.layout.Divider(),
    simper_csv_note,
    pn.layout.Divider(),
    simper_table,
    sizing_mode="stretch_both",
)



anosim_pane = pn.Column(
    pn.pane.Markdown("### ANOSIM"),
    anosim_status,
    pn.layout.Divider(),
    anosim_csv_note,
    pn.layout.Divider(),
    anosim_result_md,
    pn.layout.Divider(),
    anosim_groups_df,
    sizing_mode="stretch_both",
)


plot_area = pn.Tabs(
    ("Taxonomy plot", pn.panel(plot_pane, sizing_mode="stretch_both", min_height=650)),
    ("PCoA", pn.panel(pcoa_pane, sizing_mode="stretch_both", min_height=650)),
    ("Diversity", pn.panel(div_pane, sizing_mode="stretch_both", min_height=650)),
    ("SIMPER", pn.panel(simper_pane, sizing_mode="stretch_both", min_height=650)),
    ("ANOSIM", pn.panel(anosim_pane, sizing_mode="stretch_both", min_height=650)),
    sizing_mode="stretch_both",
)



def _toggle_menus(event=None):
    menus_container.visible = not (hide_menus.value or export_mode.value)


hide_menus.param.watch(lambda e: _toggle_menus(), "value")
export_mode.param.watch(lambda e: _toggle_menus(), "value")
_toggle_menus()

root = pn.Column(
    header,
    menus_container,
    pn.layout.Divider(),
    plot_area,
    sizing_mode="stretch_both",
)

if __name__ == "__main__":
    pn.serve(
        root,
        address="0.0.0.0",
        port=5012,
        allow_websocket_origin=["*"],
        show=True,
    )


