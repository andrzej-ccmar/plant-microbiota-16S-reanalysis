from pathlib import Path

#!/usr/bin/env python3
"""
shannon_viewer_panel_v4.py

Panel app to visualise precomputed Shannon indices from shannon_<level>.csv files, merged with metadata.

Under your V4/V7 tree, it understands structures like:

  saved_matrices_phylum/shannon_phylum.csv                -> subset 'ALL'
  saved_matrices_phylum/endosphere/shannon_phylum.csv     -> subset 'endosphere'
  saved_matrices_phylum/plant_rice/shannon_phylum.csv     -> subset 'plant_rice'
  ...

Run:
  panel serve shannon_viewer_panel_v4.py --show
"""

import os
import sys
import glob
import json
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import panel as pn
pn.extension("matplotlib")


# ---------------- Metadata ----------------
META_CANONICAL = "./metadata_V7_with_taxa.csv"


if os.path.exists(META_CANONICAL):
    METADATA_FILE = META_CANONICAL
elif os.path.exists(META_LOCAL_17M):
    METADATA_FILE = META_LOCAL_17M
elif os.path.exists(META_LOCAL_V7):
    METADATA_FILE = META_LOCAL_V7
else:
    sys.exit(
        "Error: metadata file not found. Tried:\n"
        f"  - {META_CANONICAL}\n"
        f"  - {META_LOCAL_17M}\n"
        f"  - {META_LOCAL_V7}\n"
    )

try:
    metadata = pd.read_csv(METADATA_FILE, index_col="Run", low_memory=False)
except Exception as e:
    sys.exit(f"Error reading metadata file {METADATA_FILE}: {e}")


# ---------------- Presets ----------------
PRESET_FILE = "shannon_viewer_presets.json"

def _load_all_presets() -> Dict[str, dict]:
    """Load presets from PRESET_FILE. Returns dict[preset_name] -> config dict."""
    if not os.path.exists(PRESET_FILE):
        return {}
    try:
        with open(PRESET_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_all_presets(presets: Dict[str, dict]) -> None:
    """Save presets dict to PRESET_FILE as JSON."""
    with open(PRESET_FILE, "w") as f:
        json.dump(presets, f, indent=2)


# --------------- Discover shannon_<level>.csv files (level + subset) ---------------
def find_shannon_files(root=".") -> Dict[str, Dict[str, str]]:
    """
    Recursively find shannon_*.csv under root.

    Returns:
      level_to_subsets: dict[level][subset] = path

    Where:
      - level is e.g. 'ASV', 'phylum', 'family'
      - subset is:
          'ALL' for saved_matrices_<level>/shannon_<level>.csv
          last folder name otherwise for .../saved_matrices_<level>/<subset>/shannon_<level>.csv
    """
    pattern = os.path.join(root, "**", "shannon_*.csv")
    files = glob.glob(pattern, recursive=True)
    level_to_subsets: Dict[str, Dict[str, str]] = {}

    for path in files:
        fname = os.path.basename(path)
        if not fname.startswith("shannon_") or not fname.endswith(".csv"):
            continue
        level = fname[len("shannon_"):-len(".csv")]

        rel = os.path.relpath(path, root)
        dir_rel = os.path.dirname(rel)  # e.g. saved_matrices_phylum/endosphere
        parts = dir_rel.split(os.sep) if dir_rel else []

        if len(parts) <= 1:
            subset = "ALL"
        else:
            subset = parts[-1]

        level_to_subsets.setdefault(level, {})
        if subset not in level_to_subsets[level]:
            level_to_subsets[level][subset] = path

    return level_to_subsets


LEVEL_TO_SUBSETS = find_shannon_files(".")
AVAILABLE_LEVELS = sorted(LEVEL_TO_SUBSETS.keys())

if not AVAILABLE_LEVELS:
    sys.exit("No shannon_<level>.csv files found. Run your compute script first.")

print("[INFO] Found shannon levels and subsets:", file=sys.stderr)
for lvl in AVAILABLE_LEVELS:
    subsets = sorted(LEVEL_TO_SUBSETS[lvl].keys())
    print(f"  {lvl}: {subsets}", file=sys.stderr)


# --------------- Load / cache Shannon+metadata ---------------
_SHANNON_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}

def load_shannon_with_metadata(level: str, subset: str) -> pd.DataFrame:
    """
    Load shannon_<level>.csv for a given subset and join with metadata.
    Returns DataFrame indexed by sample (Run), with columns: ["Shannon", <all metadata columns>]
    """
    key = (level, subset)
    if key in _SHANNON_CACHE:
        return _SHANNON_CACHE[key]

    subset_map = LEVEL_TO_SUBSETS.get(level, {})
    if subset not in subset_map:
        raise RuntimeError(f"No shannon file for level '{level}' and subset '{subset}'.")

    path = subset_map[subset]
    df = pd.read_csv(path)

    if "Sample" not in df.columns or "Shannon" not in df.columns:
        raise ValueError(f"{path} must have columns 'Sample' and 'Shannon'.")

    df = df.set_index("Sample")

    common = df.index.intersection(metadata.index)
    if len(common) == 0:
        raise RuntimeError(
            f"No overlap between samples in {path} and metadata index 'Run'. "
            f"(level={level}, subset={subset})"
        )

    merged = df.loc[common].join(metadata.loc[common])
    _SHANNON_CACHE[key] = merged
    return merged


# --------------- Plotting helpers ---------------
def parse_manual_order(text: str) -> List[str]:
    """Parse a comma-separated manual order string into a list of labels."""
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    return [p for p in parts if p]

def parse_color_mapping(text: str) -> Dict[str, str]:
    """Parse 'group=color,...' into a dict[group]=color."""
    mapping: Dict[str, str] = {}
    if not text:
        return mapping
    for item in text.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, color = item.split("=", 1)
        name = name.strip()
        color = color.strip()
        if name and color:
            mapping[name] = color
    return mapping

def build_color_map(groups: List[str], custom_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build a color map for the given groups.
    Uses Matplotlib colormaps API (no deprecation warning).
    """
    cmap = matplotlib.colormaps.get_cmap("tab20")
    n = max(1, len(groups))
    auto_colors: Dict[str, str] = {}
    for i, g in enumerate(groups):
        rgba = cmap(i / max(1, n - 1))
        auto_colors[g] = matplotlib.colors.to_hex(rgba)

    colors: Dict[str, str] = {}
    for g in groups:
        if g in custom_map:
            colors[g] = custom_map[g]
        elif g.lower() == "other":
            colors[g] = custom_map.get(g, "#aaaaaa")
        else:
            colors[g] = auto_colors[g]
    return colors

def compute_group_stats(df: pd.DataFrame, group_col: str = "_group") -> pd.DataFrame:
    """Compute per-group mean, SE, and n for Shannon."""
    rows = []
    for g, sub in df.groupby(group_col):
        vals = sub["Shannon"].astype(float).values
        n = len(vals)
        mean = float(np.mean(vals)) if n else np.nan
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        rows.append((str(g), mean, se, n))
    return pd.DataFrame(rows, columns=["group", "mean", "se", "n"]).set_index("group")

def get_figsize(n_groups: int, auto_width: bool, width: float, height: float) -> Tuple[float, float]:
    """
    Decide figure size (inches).
    If auto_width is True, width scales with #groups but is capped.
    """
    if auto_width:
        w = max(6.0, 0.45 * max(1, int(n_groups)))
        w = min(w, 16.0)
    else:
        w = max(3.0, float(width))
    h = max(3.0, float(height))
    return (w, h)


def make_shannon_plot(
    level: str,
    subset: str,
    group_factor: str,
    x_order_mode: str,
    manual_order_text: str,
    plot_mode: str,
    x_label_rotation: int,
    filter_factor: Optional[str],
    filter_values: List[str],
    bin_non_selected: bool,
    highlight_groups: List[str],
    dot_size: int,
    per_sample_spread: float,
    custom_colors_text: str,
    show_sample_labels: bool,
    show_se_overlay: bool,
    auto_fig_width: bool,
    fig_width: float,
    fig_height: float,
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """
    Returns (fig, (w_in, h_in)) so the Panel pane can size itself consistently.
    """
    df = load_shannon_with_metadata(level, subset)

    # Apply metadata filter
    if filter_factor is not None and filter_factor in df.columns:
        if filter_values:
            df = df[df[filter_factor].astype(str).isin(filter_values)]
        if df.empty:
            raise RuntimeError("No samples left after applying metadata filter.")

    if group_factor not in df.columns:
        raise ValueError(f"Group x-axis factor '{group_factor}' not found in metadata.")

    df["_group"] = df[group_factor].astype(str)

    # Optional: collapse non-highlighted groups into "Other"
    if bin_non_selected and highlight_groups:
        keep_set = set(str(g) for g in highlight_groups)
        mask_keep = df["_group"].isin(keep_set)
        if (~mask_keep).any():
            df.loc[~mask_keep, "_group"] = "Other"

    group_means = df.groupby("_group")["Shannon"].mean()

    # Order
    if x_order_mode == "Alphabetical":
        ordered_groups = sorted(group_means.index.astype(str))
    elif x_order_mode == "Mean Shannon (ascending)":
        ordered_groups = group_means.sort_values(ascending=True).index.tolist()
    elif x_order_mode == "Mean Shannon (descending)":
        ordered_groups = group_means.sort_values(ascending=False).index.tolist()
    elif x_order_mode == "Manual (custom list)":
        manual_order = parse_manual_order(manual_order_text)
        ordered_groups = []
        for g in manual_order:
            if g in group_means.index and g not in ordered_groups:
                ordered_groups.append(g)
        for g in sorted(group_means.index.astype(str)):
            if g not in ordered_groups:
                ordered_groups.append(g)
    else:
        ordered_groups = list(group_means.index)

    n_groups = len(ordered_groups)
    if n_groups == 0:
        raise RuntimeError("No groups found for the selected factor (after filtering/binning).")

    # Colors
    custom_map = parse_color_mapping(custom_colors_text)
    color_map = build_color_map(list(ordered_groups), custom_map)

    # Figure
    w_in, h_in = get_figsize(n_groups=n_groups, auto_width=auto_fig_width, width=fig_width, height=fig_height)
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    x_positions = np.arange(n_groups)

    if plot_mode == "Per group (mean ± SE)":
        means, ses, bar_colors = [], [], []
        for g in ordered_groups:
            vals = df.loc[df["_group"] == g, "Shannon"].values
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            means.append(m)
            ses.append(se)
            bar_colors.append(color_map.get(g, "#333333"))

        ax.bar(x_positions, np.array(means), yerr=np.array(ses), capsize=4, color=bar_colors)
        ax.set_ylabel("Shannon index (H')")
    else:
        spread = max(0.0, float(per_sample_spread))

        for xi, g in enumerate(ordered_groups):
            df_g = df.loc[df["_group"] == g]
            if df_g.empty:
                continue
            vals = df_g["Shannon"].values
            sample_ids = df_g.index.astype(str)

            n = len(vals)
            if n == 1 or spread == 0.0:
                xs = np.array([xi])
            else:
                xs = xi + np.linspace(-spread, spread, n)

            ax.scatter(xs, vals, alpha=0.7, s=float(dot_size),
                       color=color_map.get(g, "#333333"), zorder=2)

            if show_sample_labels:
                for xj, yj, sid in zip(xs, vals, sample_ids):
                    ax.text(xj, yj, sid, rotation=90, fontsize=7,
                            ha="center", va="bottom", alpha=0.8, zorder=3)

        # Mean ± SE overlay
        if show_se_overlay:
            stats = compute_group_stats(df, group_col="_group")
            for xi, g in enumerate(ordered_groups):
                if g not in stats.index:
                    continue
                m = stats.loc[g, "mean"]
                se = stats.loc[g, "se"]
                ax.errorbar([xi], [m], yerr=[se], fmt="o",
                            markersize=4, capsize=4, color="black",
                            alpha=0.9, zorder=5)

        ax.set_ylabel("Shannon index (H')")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_groups, rotation=x_label_rotation, ha="right")
    ax.set_xlabel(group_factor)
    title_subset = "" if subset == "ALL" else f" ({subset})"
    ax.set_title(f"{level} — Shannon by {group_factor}{title_subset}")

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()

    # Auto-save PNG of current view
    safe_group = group_factor.replace(" ", "_")
    safe_subset = subset.replace(" ", "_")
    png_name = f"shannon_{level}_{safe_subset}_{safe_group}.png"
    fig.savefig(png_name, dpi=150)
    print(f"[SAVE] Plot saved to {png_name}", file=sys.stderr)

    return fig, (w_in, h_in)


# --------------- Panel app ---------------
def build_app():
    default_level = "phylum" if "phylum" in AVAILABLE_LEVELS else AVAILABLE_LEVELS[0]

    level_select = pn.widgets.Select(name="Taxonomic level", options=AVAILABLE_LEVELS, value=default_level)

    initial_subsets = sorted(LEVEL_TO_SUBSETS[default_level].keys())
    subset_select = pn.widgets.Select(
        name="Subset (folder)", options=initial_subsets, value=initial_subsets[0] if initial_subsets else None
    )

    def _update_subset_options(event):
        level = event.new
        subsets = sorted(LEVEL_TO_SUBSETS[level].keys())
        subset_select.options = subsets
        subset_select.value = subsets[0] if subsets else None

    level_select.param.watch(_update_subset_options, "value")

    group_xaxis_select = pn.widgets.Select(
        name="Group x-axis (metadata factor)",
        options=list(metadata.columns),
        value=metadata.columns[0] if len(metadata.columns) > 0 else None,
    )

    x_order_select = pn.widgets.Select(
        name="X-axis order",
        options=["Alphabetical", "Mean Shannon (ascending)", "Mean Shannon (descending)", "Manual (custom list)"],
        value="Alphabetical",
    )

    manual_order_input = pn.widgets.TextInput(
        name="Manual x-axis order (comma-separated labels)",
        placeholder="e.g. bulk_soil,rhizosphere,rhizoplane,endosphere",
        value="",
    )

    plot_mode_select = pn.widgets.RadioButtonGroup(
        name="Plot mode",
        options=["Per group (mean ± SE)", "Per sample"],
        value="Per group (mean ± SE)",
        button_type="default",
    )

    x_label_rot = pn.widgets.IntSlider(name="X label rotation", start=0, end=90, value=45, step=5)

    dot_size_slider = pn.widgets.IntSlider(name="Dot size (per-sample mode)", start=5, end=80, value=20, step=1)

    per_sample_spread_slider = pn.widgets.FloatSlider(
        name="Per-sample X spread (half-width)", start=0.0, end=1.5, step=0.1, value=0.4
    )

    custom_colors_input = pn.widgets.TextInput(
        name="Custom colors (group=color,...)",
        placeholder="e.g. bulk_soil=#1f77b4,rhizosphere=#ff7f0e,Other=#aaaaaa",
        value="",
    )

    show_sample_labels_checkbox = pn.widgets.Checkbox(name="Show sample names (per-sample mode)", value=False)
    show_se_overlay_checkbox = pn.widgets.Checkbox(name="Show mean ± SE overlay (per-sample mode)", value=True)

    # Figure size controls (inches)
    auto_fig_width_checkbox = pn.widgets.Checkbox(name="Auto width by #groups", value=True)
    fig_width_input = pn.widgets.FloatInput(name="Figure width (inches)", value=10.0, step=0.5, start=3.0)
    fig_height_input = pn.widgets.FloatInput(name="Figure height (inches)", value=5.0, step=0.5, start=3.0)

    # NEW: render controls (these affect what you see in the browser)
    render_dpi_slider = pn.widgets.IntSlider(name="Render DPI (browser)", start=72, end=250, value=120, step=1)
    render_scale_slider = pn.widgets.FloatSlider(name="Render scale (browser)", start=0.4, end=2.0, value=1.0, step=0.1)

    # Make it obvious that width is ignored when auto-width is on
    def _sync_disabled(event=None):
        fig_width_input.disabled = bool(auto_fig_width_checkbox.value)

    auto_fig_width_checkbox.param.watch(lambda e: _sync_disabled(), "value")
    _sync_disabled()

    # Filters
    filter_factor_select = pn.widgets.Select(name="Filter by metadata factor", options=[None] + list(metadata.columns), value=None)
    filter_values_multiselect = pn.widgets.MultiSelect(name="Filter values (empty = no filter)", options=[], size=8)

    highlight_groups_multiselect = pn.widgets.MultiSelect(
        name='Groups to show separately (others → "Other")', options=[], size=10
    )
    bin_non_selected_checkbox = pn.widgets.Checkbox(name='Bin non-selected groups into "Other"', value=False)

    # Presets
    presets_dict = _load_all_presets()
    preset_names = sorted(presets_dict.keys())

    preset_name_input = pn.widgets.TextInput(name="New preset name", placeholder="e.g. Andrzej_default", value="")
    preset_select = pn.widgets.Select(
        name="Existing presets", options=preset_names if preset_names else [], value=preset_names[0] if preset_names else None
    )
    save_preset_button = pn.widgets.Button(name="Save preset", button_type="primary")
    load_preset_button = pn.widgets.Button(name="Load preset", button_type="default")
    preset_status = pn.pane.Markdown("")

    def _update_filter_options(event):
        factor = event.new
        if factor is None:
            filter_values_multiselect.options = []
            filter_values_multiselect.value = []
        else:
            vals = metadata[factor].dropna().astype(str).unique()
            vals = sorted(vals)
            filter_values_multiselect.options = vals
            filter_values_multiselect.value = vals

    filter_factor_select.param.watch(_update_filter_options, "value")
    _update_filter_options(type("E", (), {"new": filter_factor_select.value})())

    def _refresh_highlight_options(event=None):
        level = level_select.value
        subset = subset_select.value
        group_factor = group_xaxis_select.value
        filter_factor = filter_factor_select.value
        filter_values = list(filter_values_multiselect.value)

        try:
            df = load_shannon_with_metadata(level, subset)
        except Exception:
            highlight_groups_multiselect.options = []
            highlight_groups_multiselect.value = []
            return

        if filter_factor is not None and filter_factor in df.columns:
            if filter_values:
                df = df[df[filter_factor].astype(str).isin(filter_values)]
            if df.empty:
                highlight_groups_multiselect.options = []
                highlight_groups_multiselect.value = []
                return

        if group_factor not in df.columns:
            highlight_groups_multiselect.options = []
            highlight_groups_multiselect.value = []
            return

        groups = df[group_factor].dropna().astype(str).unique()
        groups = sorted(groups)
        highlight_groups_multiselect.options = groups
        highlight_groups_multiselect.value = groups

    for w in (level_select, subset_select, group_xaxis_select, filter_factor_select, filter_values_multiselect):
        w.param.watch(_refresh_highlight_options, "value")
    _refresh_highlight_options()

    def _collect_current_config() -> dict:
        return {
            "level": level_select.value,
            "subset": subset_select.value,
            "group_factor": group_xaxis_select.value,
            "x_order_mode": x_order_select.value,
            "manual_order_text": manual_order_input.value,
            "plot_mode": plot_mode_select.value,
            "x_label_rotation": x_label_rot.value,
            "dot_size": dot_size_slider.value,
            "per_sample_spread": per_sample_spread_slider.value,
            "custom_colors_text": custom_colors_input.value,
            "show_sample_labels": show_sample_labels_checkbox.value,
            "show_se_overlay": show_se_overlay_checkbox.value,
            "auto_fig_width": auto_fig_width_checkbox.value,
            "fig_width": fig_width_input.value,
            "fig_height": fig_height_input.value,
            "render_dpi": render_dpi_slider.value,
            "render_scale": render_scale_slider.value,
            "filter_factor": filter_factor_select.value,
            "filter_values": list(filter_values_multiselect.value),
            "highlight_groups": list(highlight_groups_multiselect.value),
            "bin_non_selected": bin_non_selected_checkbox.value,
        }

    def _apply_config(config: dict):
        applied, skipped = [], []

        level = config.get("level")
        if level in AVAILABLE_LEVELS:
            level_select.value = level
            applied.append("level")
        else:
            skipped.append("level")

        subset = config.get("subset")
        current_subsets = LEVEL_TO_SUBSETS.get(level_select.value, {}).keys()
        if subset in current_subsets:
            subset_select.value = subset
            applied.append("subset")
        else:
            skipped.append("subset")

        gf = config.get("group_factor")
        if gf in metadata.columns:
            group_xaxis_select.value = gf
            applied.append("group_factor")
        else:
            skipped.append("group_factor")

        ff = config.get("filter_factor")
        if (ff is None) or (ff in metadata.columns):
            filter_factor_select.value = ff
            applied.append("filter_factor")
        else:
            skipped.append("filter_factor")

        # after filter_factor sets options
        fvals = config.get("filter_values", [])
        if ff is not None and ff in metadata.columns:
            valid = set(filter_values_multiselect.options)
            filter_values_multiselect.value = [v for v in fvals if v in valid]
            applied.append("filter_values")
        else:
            skipped.append("filter_values")

        for key, widget, cast in [
            ("x_order_mode", x_order_select, str),
            ("manual_order_text", manual_order_input, str),
            ("plot_mode", plot_mode_select, str),
            ("x_label_rotation", x_label_rot, int),
            ("dot_size", dot_size_slider, int),
            ("per_sample_spread", per_sample_spread_slider, float),
            ("custom_colors_text", custom_colors_input, str),
            ("show_sample_labels", show_sample_labels_checkbox, bool),
            ("show_se_overlay", show_se_overlay_checkbox, bool),
            ("auto_fig_width", auto_fig_width_checkbox, bool),
            ("fig_width", fig_width_input, float),
            ("fig_height", fig_height_input, float),
            ("render_dpi", render_dpi_slider, int),
            ("render_scale", render_scale_slider, float),
            ("bin_non_selected", bin_non_selected_checkbox, bool),
        ]:
            if key in config:
                try:
                    val = cast(config[key])
                    if hasattr(widget, "options") and widget.options is not None:
                        if isinstance(widget.options, list) and (val not in widget.options):
                            skipped.append(key)
                            continue
                    widget.value = val
                    applied.append(key)
                except Exception:
                    skipped.append(key)

        _sync_disabled()
        _refresh_highlight_options()

        hconf = config.get("highlight_groups", [])
        valid_h = set(highlight_groups_multiselect.options)
        new_h = [g for g in hconf if g in valid_h]
        if new_h:
            highlight_groups_multiselect.value = new_h
            applied.append("highlight_groups")
        else:
            skipped.append("highlight_groups")

        msg = f"Loaded preset (applied: {', '.join(applied) or 'none'}"
        msg += f"; skipped: {', '.join(skipped)})" if skipped else ")"
        preset_status.object = msg

    def _on_save_preset(event):
        name = preset_name_input.value.strip()
        if not name:
            preset_status.object = "**Please enter a preset name.**"
            return
        presets_dict[name] = _collect_current_config()
        try:
            _save_all_presets(presets_dict)
        except Exception as e:
            preset_status.object = f"Error saving preset: `{e}`"
            return
        names = sorted(presets_dict.keys())
        preset_select.options = names
        preset_select.value = name
        preset_status.object = f"✅ Preset **{name}** saved."

    def _on_load_preset(event):
        name = preset_select.value
        if not name or name not in presets_dict:
            preset_status.object = "**No preset selected or preset not found.**"
            return
        _apply_config(presets_dict.get(name, {}))

    save_preset_button.on_click(_on_save_preset)
    load_preset_button.on_click(_on_load_preset)

    @pn.depends(
        level=level_select,
        subset=subset_select,
        group_factor=group_xaxis_select,
        x_order_mode=x_order_select,
        manual_order_text=manual_order_input,
        plot_mode=plot_mode_select,
        x_label_rotation=x_label_rot,
        filter_factor=filter_factor_select,
        filter_values=filter_values_multiselect,
        bin_non_selected=bin_non_selected_checkbox,
        highlight_groups=highlight_groups_multiselect,
        dot_size=dot_size_slider,
        per_sample_spread=per_sample_spread_slider,
        custom_colors_text=custom_colors_input,
        show_sample_labels=show_sample_labels_checkbox,
        show_se_overlay=show_se_overlay_checkbox,
        auto_fig_width=auto_fig_width_checkbox,
        fig_width=fig_width_input,
        fig_height=fig_height_input,
        render_dpi=render_dpi_slider,
        render_scale=render_scale_slider,
    )
    def plot_view(
        level, subset, group_factor, x_order_mode, manual_order_text,
        plot_mode, x_label_rotation, filter_factor, filter_values,
        bin_non_selected, highlight_groups, dot_size, per_sample_spread,
        custom_colors_text, show_sample_labels, show_se_overlay,
        auto_fig_width, fig_width, fig_height, render_dpi, render_scale
    ):
        try:
            fig, (w_in, h_in) = make_shannon_plot(
                level=level,
                subset=subset,
                group_factor=group_factor,
                x_order_mode=x_order_mode,
                manual_order_text=manual_order_text,
                plot_mode=plot_mode,
                x_label_rotation=x_label_rotation,
                filter_factor=filter_factor,
                filter_values=list(filter_values),
                bin_non_selected=bin_non_selected,
                highlight_groups=list(highlight_groups),
                dot_size=int(dot_size),
                per_sample_spread=float(per_sample_spread),
                custom_colors_text=custom_colors_text,
                show_sample_labels=bool(show_sample_labels),
                show_se_overlay=bool(show_se_overlay),
                auto_fig_width=bool(auto_fig_width),
                fig_width=float(fig_width),
                fig_height=float(fig_height),
            )

            # Browser display sizing (px)
            dpi = int(render_dpi)
            scale = float(render_scale)
            px_w = int(w_in * dpi * scale)
            px_h = int(h_in * dpi * scale)

            return pn.pane.Matplotlib(
                fig,
                tight=True,
                dpi=dpi,
                width=px_w,
                height=px_h,
                sizing_mode="fixed",
            )
        except Exception as e:
            return pn.pane.Markdown(f"**Error:** {e}")

    controls = pn.WidgetBox(
        "### Level & subset",
        level_select,
        subset_select,
        "### Grouping & order",
        group_xaxis_select,
        x_order_select,
        manual_order_input,
        plot_mode_select,
        x_label_rot,
        "### Per-sample settings",
        dot_size_slider,
        per_sample_spread_slider,
        show_sample_labels_checkbox,
        show_se_overlay_checkbox,
        "### Plot sizing (inches)",
        auto_fig_width_checkbox,
        pn.Row(fig_width_input, fig_height_input),
        "### Display in browser",
        render_dpi_slider,
        render_scale_slider,
        "### Colors",
        custom_colors_input,
        "### Filter & binning",
        filter_factor_select,
        filter_values_multiselect,
        highlight_groups_multiselect,
        bin_non_selected_checkbox,
        "### Presets",
        preset_name_input,
        pn.Row(save_preset_button, load_preset_button),
        preset_select,
        preset_status,
        width=560,
    )

    return pn.Row(controls, plot_view)


app = build_app()
app.servable("Shannon Viewer")

if __name__ == "__main__":
    pn.serve(app, show=True)
