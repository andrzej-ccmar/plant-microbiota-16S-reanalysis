#!/usr/bin/env python3
import os
import sys
import glob
import json  # --- PRESETS: new import
import pandas as pd
import numpy as np

import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.palettes import Category10, Category20  # kept for compatibility elsewhere
from skbio.stats.ordination import pcoa  # optional, kept for compatibility

# --- Optional Plotly import (3D) ---
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go  # noqa: F401
    pn.extension("plotly")
except Exception:
    PLOTLY_AVAILABLE = False
    pn.extension()

# ============================================================
# Paths & constants
# ============================================================
BASE_OUTPUT_PREFIX = "./saved_matrices"
LEVELS = ["ASV", "kingdom", "phylum", "class", "order", "family", "genus"]  # add "species/strain" if you have it

# Metadata: prefer local file, else fallback
META_LOCAL = "./metadata_17March_with_taxa.csv"
META_FALLBACK = "/usr/local/storage/ebi-data/metadata_onlyhere/metadata_17March_with_taxa.csv"
if os.path.exists(META_LOCAL):
    METADATA_FILE = META_LOCAL
elif os.path.exists(META_FALLBACK):
    METADATA_FILE = META_FALLBACK
else:
    sys.exit("Error: metadata_17March_with_taxa.csv not found (tried local and fallback).")

# ============================================================
# Metadata load
# ============================================================
try:
    metadata = pd.read_csv(METADATA_FILE, index_col="Run")
except Exception as e:
    sys.exit(f"Error loading metadata: {e}")

# ============================================================
# Helpers to discover/resolve data subsets
# ============================================================
def _level_root(level: str) -> str:
    return f"{BASE_OUTPUT_PREFIX}_{level}"

def _discover_subfolders(level: str):
    """
    Discover available 'niche' and 'plant_*' subfolders for the given level.
    Any direct subdir not starting with 'plant_' is considered a niche (can be combined names).
    Plant options come from 'plant_<name>' subdirs.
    We also look one level deeper to catch combined layouts.
    """
    root = _level_root(level)
    niches, plants = set(), set()
    if not os.path.isdir(root):
        return [], []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        name = entry.name
        if name.startswith("plant_"):
            plants.add(name.replace("plant_", "", 1))
        else:
            niches.add(name)
    # nested possibilities
    for niche_dir in glob.glob(os.path.join(root, "*/")):
        nname = os.path.basename(os.path.normpath(niche_dir))
        if nname and not nname.startswith("plant_"):
            niches.add(nname)
        for pdir in glob.glob(os.path.join(niche_dir, "plant_*")):
            plants.add(os.path.basename(pdir).replace("plant_", "", 1))
    for pdir in glob.glob(os.path.join(root, "plant_*")):
        plants.add(os.path.basename(pdir).replace("plant_", "", 1))
        for niche_dir in glob.glob(os.path.join(pdir, "*/")):
            nname = os.path.basename(os.path.normpath(niche_dir))
            if nname and not nname.startswith("plant_"):
                niches.add(nname)
    return sorted(niches), sorted(plants)

def _resolve_level_dir(level: str, niche: str|None, plant: str|None):
    """
    Return the most-specific directory that has the required PCoA files for the given level.
    Priority (most specific → least):
      root/<niche>/plant_<plant>/
      root/plant_<plant>/<niche>/
      root/plant_<plant>/
      root/<niche>/
      root/
    A candidate is accepted only if it contains BOTH:
      pcoa_coordinates_<level>.csv  AND  eigenvalues_<level>.csv
    """
    root = _level_root(level)

    def ok(dirpath: str) -> bool:
        return (
            os.path.isdir(dirpath)
            and os.path.exists(os.path.join(dirpath, f"pcoa_coordinates_{level}.csv"))
            and os.path.exists(os.path.join(dirpath, f"eigenvalues_{level}.csv"))
        )

    candidates = []
    if niche and plant:
        candidates += [
            os.path.join(root, niche, f"plant_{plant}"),
            os.path.join(root, f"plant_{plant}", niche),
        ]
    if plant:
        candidates.append(os.path.join(root, f"plant_{plant}"))
    if niche:
        candidates.append(os.path.join(root, niche))
    candidates.append(root)  # least specific

    for c in candidates:
        if ok(c):
            return c
    return root  # fallback; loader will raise if files missing

# ============================================================
# Dynamic BINNING OF CATEGORICAL FACTORS (not abundance)
# ============================================================
BIN_SUFFIX = " [binned]"
ABUNDANCE_BINS_FACTOR = "__AbundanceBins__"  # virtual legend factor for abundance bins

def is_binned(name: str) -> bool:
    return isinstance(name, str) and name.endswith(BIN_SUFFIX)

def base_factor(name: str) -> str:
    return name[:-len(BIN_SUFFIX)] if is_binned(name) else name

def binned_label(name: str) -> str:
    return f"{name}{BIN_SUFFIX}"

_bin_maps = {}           # {factor: {orig_cat -> bin_name}}
_bin_unmapped = {}       # {factor: ("keep"|"other", other_label)}

def apply_binning_series(series: pd.Series, factor: str) -> pd.Series:
    series = series.astype(str)
    mapping = _bin_maps.get(factor, {})
    if not mapping:
        return series
    mode, other_label = _bin_unmapped.get(factor, ("keep", "Other"))
    mapped = series.map(mapping)
    return mapped.fillna(other_label if mode == "other" else series)

# ============================================================
# Widgets
# ============================================================
NICHE_ALL = "(All niches)"
PLANT_ALL = "(All plants)"

level_widget = pn.widgets.Select(name="Taxonomy Level", options=LEVELS, value="phylum")
niche_widget = pn.widgets.Select(name="Niche", options=[NICHE_ALL], value=NICHE_ALL)
plant_widget = pn.widgets.Select(name="Plant species", options=[PLANT_ALL], value=PLANT_ALL)
path_info = pn.pane.Markdown("", sizing_mode="stretch_width")

# Always-visible header toggles
hide_menus = pn.widgets.Toggle(name="Hide all menus", value=False, button_type="primary")
export_mode = pn.widgets.Toggle(name="Export / Clean canvas", value=False, button_type="danger")
export_size = pn.widgets.IntSlider(name="Export square size (px)", start=600, end=2000, step=50, value=900)

pc_options = [f"PC{i}" for i in range(1, 11)]
pc_x_widget = pn.widgets.Select(name="X Axis PC", options=pc_options, value="PC1")
pc_y_widget = pn.widgets.Select(name="Y Axis PC", options=pc_options, value="PC2")
pc_z_widget = pn.widgets.Select(name="Z Axis PC", options=pc_options, value="PC3")

toggle_3d = pn.widgets.Checkbox(name="3D (Plotly)", value=False, disabled=not PLOTLY_AVAILABLE)
flip_x_widget = pn.widgets.Checkbox(name="Flip X (mirror)", value=False)
flip_y_widget = pn.widgets.Checkbox(name="Flip Y (mirror)", value=False)

# Legend positioning & visibility
legend_visible = pn.widgets.Checkbox(name="Show Legend", value=True)
legend_loc = pn.widgets.Select(
    name="Legend Location",
    options=["top_right","top_left","bottom_right","bottom_left","center","Custom (x,y)"],
    value="top_right",
)
legend_x = pn.widgets.IntSlider(name="Legend X (px)", start=0, end=2000, value=50)
legend_y = pn.widgets.IntSlider(name="Legend Y (px)", start=0, end=2000, value=50)

# NEW: Legend sizing controls
legend_text_size = pn.widgets.IntSlider(name="Legend Text Size (pt)", start=6, end=28, step=1, value=10)
legend_dot_size  = pn.widgets.IntSlider(name="Legend Dot Size (px, 2D only)", start=6, end=40, step=1, value=12)

def _toggle_legend_xy_vis(event=None):
    custom = legend_loc.value == "Custom (x,y)"
    legend_x.visible = custom
    legend_y.visible = custom
_toggle_legend_xy_vis()
legend_loc.param.watch(lambda e: _toggle_legend_xy_vis(), "value")

bins_apply_toggle = pn.widgets.Checkbox(name="Apply Binning (if defined)", value=True)

def factor_options():
    cols = list(metadata.columns)
    binned = [binned_label(f) for f, m in _bin_maps.items() if m]
    return ["(None)"] + cols + binned

color_widget = pn.widgets.Select(name="Color Factor", options=factor_options(), value="(None)")
shape_widget = pn.widgets.Select(name="Shape Factor", options=factor_options(), value="(None)")
highlight_widget = pn.widgets.Select(name="Highlight Factor", options=factor_options(), value="(None)")
filter_widget = pn.widgets.Select(name="Filter Factor", options=factor_options(), value="(None)")

highlight_cats = pn.widgets.MultiSelect(name="Highlight Categories", options=[], size=6)
highlight_sort = pn.widgets.RadioButtonGroup(name="Highlight Sort Order", options=["Alphabetical", "Abundance"], value="Alphabetical")

filter_cats = pn.widgets.MultiSelect(name="Filter Categories", options=[], size=6)
filter_sort = pn.widgets.RadioButtonGroup(name="Filter Sort Order", options=["Alphabetical", "Abundance"], value="Alphabetical")

abundance_toggle = pn.widgets.Checkbox(name="Color Points by Highlighted Taxon Abundance", value=False)
abundance_bin_mode = pn.widgets.RadioButtonGroup(name="Abundance bin mode", options=["Automatic", "Manual"], value="Automatic")
abundance_bin_edges = pn.widgets.TextInput(name="Manual bin edges (comma-separated)", placeholder="e.g. 0, 0.5, 1, 2, 5, 10")
abundance_bin_preview = pn.pane.Markdown("Bins: automatic (5 equal bins)", sizing_mode="stretch_width")

show_names = pn.widgets.Checkbox(name="Show Sample Names", value=False)
marker_size = pn.widgets.IntSlider(name="Marker Size", start=1, end=300, step=1, value=100)
zoom_slider = pn.widgets.FloatSlider(name="Zoom (2D only)", start=0.2, end=3.0, step=0.1, value=1.0)

legend_mode = pn.widgets.RadioButtonGroup(name="Legend Order Mode", options=["Alphabetical", "Abundance", "Custom"], value="Alphabetical")
legend_items = pn.widgets.Select(name="Legend Items (select, then use arrows / color)", options=[], size=10)
legend_up_btn = pn.widgets.Button(name="↑ Move Up", button_type="primary")
legend_down_btn = pn.widgets.Button(name="↓ Move Down", button_type="primary")
legend_top_btn = pn.widgets.Button(name="⤒ Top", button_type="success")
legend_bottom_btn = pn.widgets.Button(name="⤓ Bottom", button_type="success")
legend_reset_btn = pn.widgets.Button(name="Reset order from current", button_type="warning")

legend_color_target = pn.widgets.Select(name="Legend Color Target", options=["(auto)"], value="(auto)")
legend_order_tick = pn.widgets.IntInput(name="legend_tick", value=0, visible=False)

color_picker = pn.widgets.ColorPicker(name="Pick Color", value="#1f77b4")
apply_color_btn = pn.widgets.Button(name="Apply to selected", button_type="success")
reset_colors_btn = pn.widgets.Button(name="Auto-assign palette", button_type="warning")
color_info = pn.pane.Markdown("", sizing_mode="stretch_width")

if not PLOTLY_AVAILABLE:
    plotly_help = pn.pane.Markdown(
        "⚠️ **Plotly not installed** — 3D view is disabled.\n\n"
        "Install with:\n\n"
        "```bash\n"
        "conda install -n pcoa -c conda-forge plotly\n"
        "# or\n"
        "pip install plotly\n"
        "```",
        sizing_mode="stretch_width"
    )
else:
    plotly_help = pn.Spacer(height=0)

bin_factor_sel = pn.widgets.Select(name="Binning: Factor", options=list(metadata.columns), value=list(metadata.columns)[0] if len(metadata.columns) else None)
bin_source_cats = pn.widgets.MultiSelect(name="Select categories to merge", options=[], size=8)
bin_name_input = pn.widgets.TextInput(name="New bin name", placeholder="e.g., root")
bin_add_btn = pn.widgets.Button(name="Add/Update bin", button_type="success")
bin_unmapped_mode = pn.widgets.RadioButtonGroup(name="Unmapped categories", options=["Keep original", "Map to Other"], value="Keep original")
bin_other_label = pn.widgets.TextInput(name="Other label", value="Other")
bin_reset_btn = pn.widgets.Button(name="Clear all bins for factor", button_type="warning")
bin_preview = pn.pane.Markdown("No bins defined.", sizing_mode="stretch_width")

# ============================================================
# Legend / Color helpers
# ============================================================
_current_legend_order = []
_color_maps = {}

_GLASBEY_64 = [
    "#000000","#ff0000","#00ff00","#0000ff","#ffff00","#ff00ff","#00ffff","#800000",
    "#008000","#000080","#808000","#800080","#008080","#808080","#c00000","#00c000",
    "#0000c0","#c0c000","#c000c0","#00c0c0","#c0c0c0","#400000","#004000","#000040",
    "#404000","#400040","#004040","#404040","#ff8080","#80ff80","#8080ff","#ffff80",
    "#ff80ff","#80ffff","#ff0080","#80ff00","#0080ff","#ff8000","#80ff80","#0080ff",
    "#8000ff","#00ff80","#ff00c0","#c0ff00","#00c0ff","#ffc000","#c0ffc0","#c0c0ff",
    "#ff8080","#80ff00","#8080c0","#ffc080","#c080ff","#80ffc0","#c0ff80","#80c0ff",
    "#ff80c0","#80ffc0","#c08080","#80c080"
]
def _get_palette(n):
    if n <= 0:
        return []
    if n <= len(_GLASBEY_64):
        return _GLASBEY_64[:n]
    times = (n // len(_GLASBEY_64)) + 1
    return (_GLASBEY_64 * times)[:n]

def _bump_legend_tick():
    legend_order_tick.value += 1

def effective_factor_name(selected: str) -> str:
    if selected == "(None)":
        return "(None)"
    if is_binned(selected):
        return selected
    if not bins_apply_toggle.value:
        return selected
    if _bin_maps.get(selected, {}):
        return binned_label(selected)
    return selected

def ensure_effective_column(df: pd.DataFrame, selected: str):
    eff = effective_factor_name(selected)
    if eff != "(None)" and is_binned(eff):
        bf = base_factor(eff)
        if eff not in df.columns:
            df[eff] = apply_binning_series(df[bf], bf)
    return df, eff

def _set_legend_items(items):
    global _current_legend_order
    _current_legend_order = list(items)
    legend_items.options = list(items)
    legend_items.value = items[0] if items else None
    _update_color_controls_enabled()
    _update_color_picker_to_selection()
    _bump_legend_tick()

def _move_selected(delta):
    if not legend_items.options or legend_items.value is None:
        return
    sel = legend_items.value
    idx = _current_legend_order.index(sel)
    j = idx + delta
    if 0 <= j < len(_current_legend_order):
        _current_legend_order[idx], _current_legend_order[j] = _current_legend_order[j], _current_legend_order[idx]
        _set_legend_items(_current_legend_order)
        legend_items.value = sel

def _move_selected_top(event=None):
    if not legend_items.options or legend_items.value is None:
        return
    sel = legend_items.value
    _current_legend_order[:] = [sel] + [c for c in _current_legend_order if c != sel]
    _set_legend_items(_current_legend_order)
    legend_items.value = sel

def _move_selected_bottom(event=None):
    if not legend_items.options or legend_items.value is None:
        return
    sel = legend_items.value
    _current_legend_order[:] = [c for c in _current_legend_order if c != sel] + [sel]
    _set_legend_items(_current_legend_order)
    legend_items.value = sel

legend_up_btn.on_click(lambda e: _move_selected(-1))
legend_down_btn.on_click(lambda e: _move_selected(1))
legend_top_btn.on_click(_move_selected_top)
legend_bottom_btn.on_click(_move_selected_bottom)
legend_reset_btn.on_click(lambda e: _set_legend_items(legend_items.options))

def _ensure_factor_colors(factor: str, ordered_cats: list):
    if factor == "(None)":
        return {}
    cmap = _color_maps.get(factor, {}).copy()
    palette = _get_palette(len(ordered_cats) if ordered_cats else 1)
    new_map = {}
    for i, cat in enumerate(ordered_cats):
        new_map[cat] = cmap.get(cat, palette[i % len(palette)])
    _color_maps[factor] = new_map
    return new_map

def _update_color_target_options():
    eff_h = effective_factor_name(highlight_widget.value)
    eff_c = effective_factor_name(color_widget.value)
    opts = ["(auto)"]
    if eff_h != "(None)":
        opts.append(eff_h)
    if eff_c != "(None)" and eff_c not in opts:
        opts.append(eff_c)
    # NEW: add bins factor if active
    if abundance_toggle.value and effective_factor_name(highlight_widget.value) != "(None)" and len(highlight_cats.value) == 1:
        if ABUNDANCE_BINS_FACTOR not in opts:
            opts.append(ABUNDANCE_BINS_FACTOR)
    if legend_color_target.value not in opts and legend_color_target.value != "(auto)":
        opts.append(legend_color_target.value)
    legend_color_target.options = opts
    if legend_color_target.value not in opts:
        legend_color_target.value = "(auto)"

def _active_color_target():
    tgt = legend_color_target.value
    if tgt != "(auto)":
        return tgt
    # Prefer bins when abundance mode is active for a single highlighted taxon
    if abundance_toggle.value and effective_factor_name(highlight_widget.value) != "(None)" and len(highlight_cats.value) == 1:
        return ABUNDANCE_BINS_FACTOR
    eff_h = effective_factor_name(highlight_widget.value)
    eff_c = effective_factor_name(color_widget.value)
    if eff_h != "(None)" and len(highlight_cats.value) > 0:
        return eff_h
    return eff_c

def _update_color_controls_enabled():
    tgt = _active_color_target()
    enabled = bool(tgt and tgt != "(None)" and len(legend_items.options) > 0)
    color_picker.disabled = not enabled
    apply_color_btn.disabled = not enabled
    reset_colors_btn.disabled = not enabled

def _update_color_picker_to_selection(event=None):
    factor = _active_color_target()
    cat = legend_items.value
    if factor and factor != "(None)" and cat and factor in _color_maps and cat in _color_maps[factor]:
        color_picker.value = _color_maps[factor][cat]
        color_info.object = f"**Factor:** `{factor}`  •  **Category:** `{cat}`  •  **Color:** `{_color_maps[factor][cat]}`"
    else:
        color_info.object = f"**Factor:** `{factor or '(None)'}`  •  **Category:** `{cat or '(None)'}`"

legend_items.param.watch(_update_color_picker_to_selection, "value")
legend_color_target.param.watch(lambda e: (_update_color_controls_enabled(), _update_color_picker_to_selection(), _bump_legend_tick()), "value")

def _apply_color(event=None):
    factor = _active_color_target()
    cat = legend_items.value
    if not factor or factor == "(None)" or not cat:
        return
    _color_maps.setdefault(factor, {})
    _color_maps[factor][cat] = color_picker.value
    _update_color_picker_to_selection()
    _bump_legend_tick()

apply_color_btn.on_click(_apply_color)

def _reset_colors(event=None):
    factor = _active_color_target()
    if not factor or factor == "(None)" or not _current_legend_order:
        return
    palette = _get_palette(len(_current_legend_order))
    _color_maps[factor] = {cat: palette[i % len(palette)] for i, cat in enumerate(_current_legend_order)}
    _update_color_picker_to_selection()
    _bump_legend_tick()

reset_colors_btn.on_click(_reset_colors)

# ============================================================
# Binning UI behavior
# ============================================================
def _refresh_bin_source_cats(event=None):
    f = bin_factor_sel.value
    if not f:
        bin_source_cats.options = []
        return
    cats = sorted(pd.Categorical(metadata[f].astype(str)).categories)
    bin_source_cats.options = cats

def _refresh_bin_preview():
    f = bin_factor_sel.value
    mapping = _bin_maps.get(f, {})
    mode, other_label = _bin_unmapped.get(f, ("keep", "Other"))
    if not mapping:
        bin_preview.object = "_No bins defined._"
        return
    lines = [f"**Factor:** `{f}`", "**Mappings:**"]
    inv = {}
    for k, v in mapping.items():
        inv.setdefault(v, []).append(k)
    for bname, members in inv.items():
        lines.append(f"- **{bname}** ⟵ {', '.join(sorted(members))}")
    lines.append(f"\n**Unmapped:** {'mapped to **'+other_label+'**' if mode=='other' else 'keep original'}")
    bin_preview.object = "\n".join(lines)

def _push_factor_options_to_selectors():
    opts = factor_options()
    for w in (color_widget, shape_widget, highlight_widget, filter_widget):
        w.options = opts
        if w.value not in opts:
            w.value = "(None)"
    _update_color_target_options()

def _add_or_update_bin(event=None):
    f = bin_factor_sel.value
    cats = list(bin_source_cats.value)
    label = bin_name_input.value.strip()
    if not f or not cats or not label:
        return
    mapping = _bin_maps.setdefault(f, {})
    for c in cats:
        mapping[c] = label
    if bin_unmapped_mode.value == "Map to Other":
        _bin_unmapped[f] = ("other", bin_other_label.value or "Other")
    else:
        _bin_unmapped[f] = ("keep", "Other")
    _refresh_bin_preview()
    _push_factor_options_to_selectors()
    _bump_legend_tick()

def _reset_bins_for_factor(event=None):
    f = bin_factor_sel.value
    if not f:
        return
    _bin_maps[f] = {}
    _bin_unmapped[f] = ("keep", "Other")
    _refresh_bin_preview()
    _push_factor_options_to_selectors()
    _bump_legend_tick()

bin_factor_sel.param.watch(lambda e: (_refresh_bin_source_cats(), _refresh_bin_preview()), "value")
bin_unmapped_mode.param.watch(lambda e: _refresh_bin_preview(), "value")
bin_other_label.param.watch(lambda e: _refresh_bin_preview(), "value")
bin_add_btn.on_click(_add_or_update_bin)
bin_reset_btn.on_click(_reset_bins_for_factor)

_refresh_bin_source_cats()
_refresh_bin_preview()

# ============================================================
# Data loading & UI wiring for subsets
# ============================================================
def _sel_or_none(label, all_label):
    return None if label == all_label else label

def _update_data_source_choices(event=None):
    level = level_widget.value
    niches, plants = _discover_subfolders(level)
    niche_opts = [NICHE_ALL] + niches
    plant_opts = [PLANT_ALL] + plants
    niche_widget.options = niche_opts
    plant_widget.options = plant_opts
    if niche_widget.value not in niche_opts:
        niche_widget.value = NICHE_ALL
    if plant_widget.value not in plant_opts:
        plant_widget.value = PLANT_ALL
    _update_path_info()

def _update_path_info(event=None):
    level = level_widget.value
    niche = _sel_or_none(niche_widget.value, NICHE_ALL)
    plant = _sel_or_none(plant_widget.value, PLANT_ALL)
    d = _resolve_level_dir(level, niche, plant)
    coord_file = os.path.join(d, f"pcoa_coordinates_{level}.csv")
    eig_file   = os.path.join(d, f"eigenvalues_{level}.csv")
    ok = os.path.exists(coord_file) and os.path.exists(eig_file)
    badge = "✅" if ok else "⚠️"
    subset = [f"niche=`{niche}`" if niche else "niche=ALL",
              f"plant=`{plant}`" if plant else "plant=ALL"]
    msg = f"{badge} **Using path**: `{d}`  •  ({', '.join(subset)})"
    if not ok:
        msg += f"\n\nMissing expected files:\n- {coord_file}\n- {eig_file}"
    path_info.object = msg

level_widget.param.watch(_update_data_source_choices, "value")
niche_widget.param.watch(_update_path_info, "value")
plant_widget.param.watch(_update_path_info, "value")
_update_data_source_choices()

def load_pcoa_for_level(level, niche=None, plant=None):
    d = _resolve_level_dir(level, niche, plant)
    coord_file = os.path.join(d, f"pcoa_coordinates_{level}.csv")
    eig_file   = os.path.join(d, f"eigenvalues_{level}.csv")
    if not (os.path.exists(coord_file) and os.path.exists(eig_file)):
        raise SystemExit(
            f"Error: missing files for level '{level}'. Tried: {d}\n"
            f"Expected: {coord_file} and {eig_file}"
        )
    coords = pd.read_csv(coord_file, index_col=0)
    eig    = pd.read_csv(eig_file,   index_col=0)
    vals   = eig["Eigenvalue"].values
    total  = vals.sum() or 1.0
    prop   = vals / total
    return coords, prop, d

# ============================================================
# Filter / Highlight options (respect chosen subset)
# ============================================================
filter_widget.param.watch(lambda e: None, "value")  # placeholder

def update_filter_options(event=None):
    f_sel = filter_widget.value
    if f_sel == "(None)":
        filter_cats.options = []; filter_cats.value = []
        return
    eff = effective_factor_name(f_sel)
    if is_binned(eff):
        bf = base_factor(eff); ser = apply_binning_series(metadata[bf], bf)
    else:
        ser = metadata[eff].astype(str)
    counts = ser.value_counts().to_dict()
    cats_sorted = sorted(counts, key=lambda c: (-counts[c], c)) if filter_sort.value == "Abundance" else sorted(counts)
    filter_cats.options = cats_sorted
    filter_cats.value = [c for c in filter_cats.value if c in cats_sorted]

filter_widget.param.watch(update_filter_options, "value")
filter_sort.param.watch(lambda e: update_filter_options(), "value")
bins_apply_toggle.param.watch(lambda e: (update_filter_options(), None), "value")

def update_highlight_options(event=None):
    hl_sel = highlight_widget.value
    if hl_sel == "(None)":
        highlight_cats.options = []; highlight_cats.value = []; return
    try:
        coords_df, _, _ = load_pcoa_for_level(
            level_widget.value,
            _sel_or_none(niche_widget.value, NICHE_ALL),
            _sel_or_none(plant_widget.value, PLANT_ALL)
        )
    except SystemExit:
        highlight_cats.options = []; highlight_cats.value = []; return
    meta_sub = metadata.loc[metadata.index.intersection(coords_df.index)]
    f_sel = filter_widget.value
    if f_sel != "(None)" and len(filter_cats.value) > 0:
        eff_f = effective_factor_name(f_sel)
        vals = apply_binning_series(meta_sub[base_factor(eff_f)], base_factor(eff_f)) if is_binned(eff_f) else meta_sub[eff_f].astype(str)
        meta_sub = meta_sub.loc[vals.isin(filter_cats.value)]
    eff_h = effective_factor_name(hl_sel)
    vals = apply_binning_series(meta_sub[base_factor(eff_h)], base_factor(eff_h)) if is_binned(eff_h) else meta_sub[eff_h].dropna().astype(str)
    counts = vals.value_counts().to_dict()
    cats_sorted = sorted(counts, key=lambda c: (-counts[c], c)) if highlight_sort.value == "Abundance" else sorted(counts)
    highlight_cats.options = cats_sorted
    highlight_cats.value = [c for c in highlight_cats.value if c in cats_sorted]

for w in [highlight_widget, filter_widget, filter_cats, level_widget, highlight_sort, niche_widget, plant_widget]:
    w.param.watch(lambda e: update_highlight_options(), "value")
bins_apply_toggle.param.watch(lambda e: update_highlight_options(), "value")

# ============================================================
# Legend builder
# ============================================================
def compute_legend_order(df, color_factor_sel, highlight_factor_sel, highlight_list):
    eff_h = effective_factor_name(highlight_factor_sel)
    eff_c = effective_factor_name(color_factor_sel)

    if eff_h != "(None)" and highlight_list:
        factor = eff_h
        vals = df[factor].astype(str)
        present = [c for c in highlight_list if (vals == str(c)).any()]
        base_items = [str(c) for c in present]
        counts = vals.value_counts().to_dict()
    elif eff_c != "(None)" and eff_c in df.columns:
        factor = eff_c
        vals = df[factor].astype(str)
        base_items = list(pd.Categorical(vals).categories)
        counts = vals.value_counts().to_dict()
    else:
        return "(None)", [], []

    if legend_mode.value == "Alphabetical":
        ordered = sorted(base_items)
    elif legend_mode.value == "Abundance":
        ordered = sorted(base_items, key=lambda x: (-counts.get(x, 0), x))
    else:
        custom = [c for c in _current_legend_order if c in base_items]
        missing = [c for c in base_items if c not in custom]
        ordered = custom + sorted(missing)

    _ensure_factor_colors(factor, ordered)
    return factor, base_items, ordered

def _refresh_legend_items(event=None):
    # Try load coords for the current subset
    try:
        coords_df, _, _ = load_pcoa_for_level(
            level_widget.value,
            _sel_or_none(niche_widget.value, NICHE_ALL),
            _sel_or_none(plant_widget.value, PLANT_ALL)
        )
    except SystemExit:
        _set_legend_items([]); return

    df = coords_df.join(metadata, how="inner")

    # Ensure any binned/effective columns exist
    for sel in [color_widget.value, shape_widget.value, filter_widget.value, highlight_widget.value]:
        df, _ = ensure_effective_column(df, sel)

    # Apply current filter to df (so Legend shows only present categories/bins)
    eff_filter = effective_factor_name(filter_widget.value)
    if eff_filter != "(None)" and len(filter_cats.value) > 0:
        df = df.loc[df[eff_filter].astype(str).isin(filter_cats.value)]

    # NEW: if abundance mode is active for a single highlighted taxon, expose bins as the legend items
    if abundance_toggle.value and effective_factor_name(highlight_widget.value) != "(None)" and len(highlight_cats.value) == 1:
        eff_h = effective_factor_name(highlight_widget.value)
        parts = eff_h.split()
        taxon_level = parts[-1] if parts else level_widget.value
        taxon = str(highlight_cats.value[0])

        abund_series_all = get_abundance_series(taxon_level, taxon,
                                                _sel_or_none(niche_widget.value, NICHE_ALL),
                                                _sel_or_none(plant_widget.value, PLANT_ALL))
        abund_series = abund_series_all.reindex(df.index).fillna(0.0)
        max_pct = float(abund_series.max())
        edges, labels, palette = compute_abundance_edges_and_labels(max_pct)

        # Build or refresh colors for the bins factor
        prev = _color_maps.get(ABUNDANCE_BINS_FACTOR, {})
        cmap = {}
        for i, lab in enumerate(labels):
            cmap[lab] = prev.get(lab, palette[i])
        _color_maps[ABUNDANCE_BINS_FACTOR] = cmap

        # Choose legend order
        if legend_mode.value == "Alphabetical":
            ordered = sorted(labels)
        elif legend_mode.value == "Abundance":
            ordered = list(labels)  # natural order (meaningful for bins)
        else:
            existing = [c for c in _current_legend_order if c in labels]
            extras = [c for c in labels if c not in existing]
            ordered = existing + extras

        _update_color_target_options()
        if ABUNDANCE_BINS_FACTOR not in legend_color_target.options:
            legend_color_target.options = legend_color_target.options + [ABUNDANCE_BINS_FACTOR]

        _set_legend_items(ordered)
        return

    # Default path (categorical legend)
    legend_factor, items, ordered = compute_legend_order(df, color_widget.value, highlight_widget.value, highlight_cats.value)
    _update_color_target_options()
    if legend_factor != "(None)" and legend_factor not in legend_color_target.options:
        legend_color_target.options = legend_color_target.options + [legend_factor]
    if legend_mode.value == "Custom":
        existing = [c for c in _current_legend_order if c in items]
        extras = [c for c in ordered if c not in existing]
        _set_legend_items(existing + extras)
    else:
        _set_legend_items(ordered)

def _trigger_refresh(event=None):
    _refresh_legend_items()
    _push_factor_options_to_selectors()

for w in [level_widget, filter_widget, filter_cats, color_widget, highlight_widget, highlight_cats, legend_mode, bins_apply_toggle, niche_widget, plant_widget]:
    w.param.watch(_trigger_refresh, "value")

_refresh_legend_items()
_update_color_controls_enabled()
_update_color_picker_to_selection()

# ============================================================
# Abundance helpers
# ============================================================
def _hex2rgb(h: str):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

def _rgb2hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def _lerp_rgb(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))

def _interpolate_palette(stops_hex: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if n <= len(stops_hex):
        return stops_hex[:n]
    stops_rgb = [_hex2rgb(c) for c in stops_hex]
    stop_pos = np.linspace(0.0, 1.0, len(stops_hex))
    sample_pos = np.linspace(0.0, 1.0, n)
    out = []
    for p in sample_pos:
        if p <= 0: out.append(stops_rgb[0]); continue
        if p >= 1: out.append(stops_rgb[-1]); continue
        i = int(np.searchsorted(stop_pos, p) - 1)
        i = max(0, min(i, len(stops_rgb) - 2))
        t = (p - stop_pos[i]) / (stop_pos[i + 1] - stop_pos[i])
        out.append(_lerp_rgb(stops_rgb[i], stops_rgb[i + 1], t))
    return [_rgb2hex(c) for c in out]

def compute_abundance_edges_and_labels(max_pct: float):
    ABSENT_COLOR = "#d3d3d3"
    ramp_stops = ["#91bfdb", "#4575b4", "#a6d96a", "#fdae61", "#d73027"]
    if max_pct <= 0:
        edges = np.array([0.0, 1.0])
        labels = ["Absent (0%)", ">0.0%"]
        palette = [ABSENT_COLOR, ramp_stops[-1]]
        abundance_bin_preview.object = "⚠️ All zeros; using 0–1 dummy bins."
        return edges, labels, palette
    if abundance_bin_mode.value == "Manual":
        ok = True
        try:
            raw = [float(x) for x in abundance_bin_edges.value.split(",") if x.strip() != ""]
            if not raw: ok = False
        except Exception:
            ok = False
        if ok:
            raw = sorted(set(raw))
            if raw[0] > 0.0: raw = [0.0] + raw
            if raw[-1] < max_pct: raw = raw + [max_pct]
            edges = np.array(raw, dtype=float)
            abundance_bin_preview.object = f"Manual bins: {list(np.round(edges, 3))}"
        else:
            edges = np.linspace(0, max_pct, 5 + 1)
            abundance_bin_preview.object = "⚠️ Invalid manual bins → Automatic 5 equal bins."
    else:
        edges = np.linspace(0, max_pct, 5 + 1)
        abundance_bin_preview.object = f"Automatic {len(edges)-1} equal bins (0–{max_pct:.2f}%)"
    n_bins = len(edges) - 1
    labels = ["Absent (0%)"]
    for i in range(1, n_bins):
        labels.append(f"{edges[i-1]:.1f}–{edges[i]:.1f}%")
    labels.append(f">{edges[n_bins-1]:.1f}%")
    ramp_colors = _interpolate_palette(ramp_stops, n_bins) if n_bins > 0 else []
    palette = [ABSENT_COLOR] + ramp_colors
    return edges, labels, palette

def get_abundance_series(level, taxon, niche=None, plant=None):
    """
    Load abundance for taxon at selected subset dir if rel_abundance exists;
    if not, gracefully fall back to parent (less specific) dirs, then to root.
    """
    root = _level_root(level)
    candidates = []
    if niche and plant:
        candidates += [
            os.path.join(root, niche, f"plant_{plant}"),
            os.path.join(root, f"plant_{plant}", niche),
        ]
    if plant:
        candidates.append(os.path.join(root, f"plant_{plant}"))
    if niche:
        candidates.append(os.path.join(root, niche))
    candidates.append(root)

    rel_file = None
    for c in candidates:
        f = os.path.join(c, f"rel_abundance_{level}.csv")
        if os.path.exists(f):
            rel_file = f
            level_folder = c
            break
    if rel_file is None:
        return pd.Series(dtype=float)

    # cache file encodes subset path so different subsets don't clash
    tag = level_folder.replace("/", "_")
    cache_file = os.path.join(level_folder, f"{taxon}_abundance__{tag}.csv")

    if os.path.exists(cache_file):
        try:
            df_cached = pd.read_csv(cache_file, index_col=0)
            if "percentage" in df_cached.columns:
                return df_cached["percentage"]
        except Exception:
            pass  # rebuild cache

    with open(rel_file, "r") as f:
        header = f.readline().rstrip("\n").split(",")
    sample_ids = header[1:]
    found = False
    percentages = []
    with open(rel_file, "r") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if parts[0] == taxon:
                found = True
                try:
                    percentages = [float(v) * 100.0 for v in parts[1:]]
                except ValueError:
                    percentages = [0.0] * len(parts[1:])
                break
    if not found:
        series = pd.Series(0.0, index=sample_ids)
        pd.DataFrame({"percentage": series}).to_csv(cache_file)
        return series
    series = pd.Series(percentages, index=sample_ids)
    pd.DataFrame({"percentage": series}).to_csv(cache_file)
    return series

# ============================================================
# Plot prep
# ============================================================
def _prep_df_for_plot(level, niche, plant, x_pc, y_pc, color_factor_sel, shape_factor_sel,
                      filter_factor_sel, filter_list, highlight_factor_sel, highlight_list,
                      flip_x, flip_y):
    coords_df, prop, _dir = load_pcoa_for_level(level, niche, plant)
    df = coords_df.join(metadata, how="inner")
    df["sample"] = df.index.astype(str)
    df, eff_color = ensure_effective_column(df, color_factor_sel)
    df, eff_shape = ensure_effective_column(df, shape_factor_sel)
    df, eff_filter = ensure_effective_column(df, filter_factor_sel)
    df, eff_highlight = ensure_effective_column(df, highlight_factor_sel)
    color_factor = eff_color
    shape_factor = eff_shape
    filter_factor = eff_filter
    highlight_factor = eff_highlight
    df["_X"] = df[x_pc] * (-1 if flip_x else 1)
    df["_Y"] = df[y_pc] * (-1 if flip_y else 1)
    if filter_factor != "(None)" and filter_list:
        df = df.loc[df[filter_factor].astype(str).isin(filter_list)].copy()
    return df, prop, color_factor, shape_factor, filter_factor, highlight_factor

# ============================================================
# Plot 2D
# ============================================================
def _apply_bokeh_legend_style(p, legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px, legend_text_pt, legend_dot_px):
    for lg in p.legend:
        lg.visible = legend_visible_flag
        lg.background_fill_alpha = 0.6
        lg.border_line_alpha = 0.6
        lg.label_text_font_size = f"{legend_text_pt}pt"
        # These control only the legend glyph (dot) size
        lg.glyph_height = legend_dot_px
        lg.glyph_width  = legend_dot_px
    if p.legend:
        if legend_loc_sel == "Custom (x,y)":
            p.legend.location = (legend_x_px, legend_y_px)
        else:
            p.legend.location = legend_loc_sel
        p.legend.click_policy = "hide"

def plot_pcoa_2d(level, niche, plant, x_pc, y_pc, color_factor_sel, shape_factor_sel,
                 filter_factor_sel, filter_list, highlight_factor_sel, highlight_list,
                 abundance_flag, show_names_flag, msize, zoom, flip_x, flip_y,
                 legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
                 export_flag, export_px,
                 legend_text_pt, legend_dot_px):
    df, prop, color_factor, shape_factor, filter_factor, highlight_factor = _prep_df_for_plot(
        level, niche, plant, x_pc, y_pc, color_factor_sel, shape_factor_sel, filter_factor_sel, filter_list,
        highlight_factor_sel, highlight_list, flip_x, flip_y
    )

    # Abundance shading if a single taxon is highlighted
    if abundance_flag and highlight_factor != "(None)" and len(highlight_list) == 1:
        parts = highlight_factor.split()
        taxon_level = parts[-1] if parts else level
        taxon = highlight_list[0]
        abund_series = get_abundance_series(taxon_level, taxon, niche, plant)
        df["abundance_pct"] = df["sample"].map(lambda s: abund_series.get(s, 0.0))
        max_pct = float(df["abundance_pct"].max())
        edges, labels, default_palette = compute_abundance_edges_and_labels(max_pct)

        # Build bins for each point
        def assign_bin(p):
            if p == 0: return 0
            for i in range(1, len(edges)-1):
                if p <= edges[i]: return i
            return len(edges)-1

        df["bin_idx"] = df["abundance_pct"].map(assign_bin)
        df["bin_label"] = df["bin_idx"].map(lambda i: labels[i])
        df["alpha"] = 0.9
        df["marker"] = "circle"

        # Legend order & colors come from Legend UI
        ordered = list(legend_items.options) if legend_items.options else list(labels)
        ordered = [lab for lab in ordered if lab in labels] + [lab for lab in labels if lab not in ordered]

        # Colors: prefer user-customized map
        bin_cmap = _color_maps.get(ABUNDANCE_BINS_FACTOR, {})
        if not bin_cmap:
            bin_cmap = {lab: default_palette[i] for i, lab in enumerate(labels)}
            _color_maps[ABUNDANCE_BINS_FACTOR] = bin_cmap

        x_idx = int(x_pc.replace("PC", "")) - 1; y_idx = int(y_pc.replace("PC", "")) - 1
        pct_x = prop[x_idx] * 100; pct_y = prop[y_idx] * 100

        fig_kwargs = dict(
            title=f"PCoA ({level}) {x_pc}={pct_x:.1f}%, {y_pc}={pct_y:.1f}%",
            x_axis_label=f"{x_pc} ({pct_x:.2f}%)", y_axis_label=f"{y_pc} ({pct_y:.2f}%)",
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            tooltips=[("Sample", "@sample"), ("Abundance (%)", "@abundance_pct"), ("Bin", "@bin_label")],
        )
        p_scatter = figure(width=export_px, height=export_px, **fig_kwargs) if export_flag else figure(sizing_mode="stretch_both", **fig_kwargs)

        # Draw per-bin in the order specified by the Legend UI
        for lab in ordered:
            sub = df[df["bin_label"] == lab]
            if sub.empty:
                continue
            col = bin_cmap.get(lab, "lightgray")
            p_scatter.scatter(
                x="_X", y="_Y",
                source=ColumnDataSource(sub),
                size=msize * 0.1,
                fill_color=col, fill_alpha=sub["alpha"].iloc[0],
                line_color=None, marker="circle",
                legend_label=lab,
            )

        _apply_bokeh_legend_style(
            p_scatter, legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
            legend_text_pt, legend_dot_size.value if legend_dot_px is None else legend_dot_px
        )

        if show_names_flag:
            p_scatter.add_layout(LabelSet(x="_X", y="_Y", text="sample", source=ColumnDataSource(df),
                                          x_offset=5, y_offset=5, text_font_size="8pt"))

        # Zoom
        x0, x1 = p_scatter.x_range.start, p_scatter.x_range.end
        y0, y1 = p_scatter.y_range.start, p_scatter.y_range.end
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        xr, yr = (x1 - x0) / (2 * zoom), (y1 - y0) / (2 * zoom)
        p_scatter.x_range.start, p_scatter.x_range.end = xm - xr, xm + xr
        p_scatter.y_range.start, p_scatter.y_range.end = ym - yr, ym + yr

        # Small histogram of bin counts, using the same order & colors
        counts = df["bin_label"].value_counts().reindex(ordered, fill_value=0)
        max_count = counts.max()
        label_offset = max_count * 0.02
        hist_df = pd.DataFrame({
            "bin_label": ordered,
            "count": counts.values,
            "count_plus": counts.values + label_offset,
            "color": [ _color_maps[ABUNDANCE_BINS_FACTOR].get(lab, "lightgray") for lab in ordered ],
            "count_str": counts.values.astype(str)
        })
        hist_source = ColumnDataSource(hist_df)
        p_hist = figure(x_range=ordered, height=200, title="Abundance Bin Counts",
                        tools="", toolbar_location=None, sizing_mode="stretch_width")
        p_hist.vbar(x="bin_label", top="count", width=0.8, fill_color="color", line_color=None, source=hist_source)
        p_hist.xaxis.major_label_orientation = 1.0
        p_hist.y_range.start = 0; p_hist.y_range.end = max(1, max_count * 1.20)
        p_hist.text(x="bin_label", y="count_plus", text="count_str", text_color="color",
                    source=hist_source, text_baseline="bottom", text_font_size="10pt")

        return pn.Column(p_scatter, p_hist, sizing_mode="stretch_both")

    # General categorical path
    legend_factor, items, ordered = compute_legend_order(
        df.copy(), color_widget.value, highlight_widget.value, highlight_cats.value
    )
    active_factor = legend_factor if legend_factor != "(None)" else None

    df["fill_color"] = "lightgray"; df["alpha"] = 0.6
    if active_factor is not None:
        vals = df[active_factor].astype(str)
        _ensure_factor_colors(active_factor, ordered)
        if effective_factor_name(highlight_widget.value) == active_factor and len(highlight_cats.value) > 0:
            mask = vals.isin(set(highlight_cats.value))
            df.loc[mask, "fill_color"] = vals[mask].map(_color_maps[active_factor]).fillna("lightgray")
            df.loc[mask, "alpha"] = 1.0
        else:
            df["fill_color"] = vals.map(_color_maps[active_factor]).fillna("lightgray")
            df["alpha"] = 0.8

    markers = ["circle", "square", "triangle", "diamond", "cross", "asterisk"]
    if shape_factor != "(None)" and shape_factor in df.columns:
        v2 = df[shape_factor].astype(str)
        c2 = list(pd.Categorical(v2).categories)
        smap = dict(zip(c2, (markers * ((len(c2) // len(markers)) + 1))[:len(c2)]))
        df["marker"] = v2.map(smap)
    else:
        df["marker"] = "circle"

    source = ColumnDataSource(df)
    x_idx = int(x_pc.replace("PC", "")) - 1; y_idx = int(y_pc.replace("PC", "")) - 1
    pct_x = prop[x_idx] * 100; pct_y = prop[y_idx] * 100

    tooltips = [("Sample", "@sample")]
    for fac in [color_factor, shape_factor, effective_factor_name(highlight_widget.value), filter_factor]:
        if fac != "(None)" and fac in df.columns:
            tooltips.append((fac, f"@{{{fac}}}"))

    fig_kwargs = dict(
        title=f"PCoA ({level}) {x_pc}={pct_x:.1f}%, {y_pc}={pct_y:.1f}%",
        x_axis_label=f"{x_pc} ({pct_x:.2f}%)",
        y_axis_label=f"{y_pc} ({pct_y:.2f}%)",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        tooltips=tooltips,
    )
    if export_flag:
        p_scatter = figure(width=export_px, height=export_px, **fig_kwargs)
    else:
        p_scatter = figure(sizing_mode="stretch_both", **fig_kwargs)

    if active_factor is not None and effective_factor_name(highlight_widget.value) == active_factor and len(highlight_cats.value) > 0:
        non_mask = ~df[active_factor].astype(str).isin([str(c) for c in items])
        if non_mask.any():
            df_non = df.loc[non_mask].copy()
            p_scatter.scatter(
                x="_X", y="_Y", source=ColumnDataSource(df_non), size=msize * 0.1,
                fill_color="lightgray", fill_alpha=0.6, line_color=None, marker="marker",
            )

    if active_factor is not None and ordered:
        for cat in ordered:
            sub = df[active_factor].astype(str) == str(cat)
            subdf = df[sub]
            if subdf.empty:
                continue
            col = _color_maps.get(active_factor, {}).get(str(cat), "lightgray")
            marker_for_sub = subdf["marker"].iloc[0] if "marker" in subdf else "circle"
            p_scatter.scatter(
                x="_X", y="_Y", source=ColumnDataSource(subdf), size=msize * 0.1,
                fill_color=col, fill_alpha=subdf["alpha"].iloc[0] if "alpha" in subdf else 0.8,
                line_color=None, marker=marker_for_sub, legend_label=str(cat),
            )
        _apply_bokeh_legend_style(
            p_scatter, legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
            legend_text_pt, legend_dot_px
        )
    else:
        p_scatter.scatter(
            x="_X", y="_Y", source=source, size=msize * 0.1,
            fill_color="fill_color", fill_alpha="alpha", line_color=None, marker="marker",
            legend_label="Samples",
        )
        _apply_bokeh_legend_style(
            p_scatter, legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
            legend_text_pt, legend_dot_px
        )

    if show_names_flag:
        p_scatter.add_layout(LabelSet(
            x="_X", y="_Y", text="sample", source=source,
            x_offset=5, y_offset=5, text_font_size="8pt",
        ))

    # Zoom
    x0, x1 = p_scatter.x_range.start, p_scatter.x_range.end
    y0, y1 = p_scatter.y_range.start, p_scatter.y_range.end
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    xr, yr = (x1 - x0) / (2 * zoom), (y1 - y0) / (2 * zoom)
    p_scatter.x_range.start, p_scatter.x_range.end = xm - xr, xm + xr
    p_scatter.y_range.start, p_scatter.y_range.end = ym - yr, ym + yr

    return p_scatter

# ============================================================
# Plot 3D
# ============================================================
def plot_pcoa_3d(level, niche, plant, x_pc, y_pc, z_pc, color_factor_sel, shape_factor_sel,
                 filter_factor_sel, filter_list, highlight_factor_sel, highlight_list,
                 abundance_flag, msize,
                 legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
                 export_flag, export_px,
                 legend_text_pt, legend_dot_px):  # legend_dot_px not used in 3D
    if not PLOTLY_AVAILABLE:
        return pn.pane.Markdown(
            "⚠️ **3D view requires Plotly.** Install with `conda install -n pcoa -c conda-forge plotly` "
            "or `pip install plotly` and restart.",
            sizing_mode="stretch_both"
        )
    import plotly.graph_objects as go

    df, prop, color_factor, shape_factor, filter_factor, highlight_factor = _prep_df_for_plot(
        level, niche, plant, x_pc, y_pc, color_factor_sel, shape_factor_sel, filter_factor_sel, filter_list,
        highlight_factor_sel, highlight_list, flip_x=False, flip_y=False
    )
    point_size = max(2, int(msize * 0.08))
    constant_opacity = 0.9
    fig = go.Figure()

    if abundance_flag and highlight_factor != "(None)" and len(highlight_list) == 1:
        parts = highlight_factor.split()
        taxon_level = parts[-1] if parts else level
        taxon = highlight_list[0]
        abund_series = get_abundance_series(taxon_level, taxon, niche, plant)
        df["abundance_pct"] = df["sample"].map(lambda s: abund_series.get(s, 0.0))
        max_pct = float(df["abundance_pct"].max())
        edges, labels, palette = compute_abundance_edges_and_labels(max_pct)

        def assign_bin(p):
            if p == 0: return 0
            for i in range(1, len(edges)-1):
                if p <= edges[i]: return i
            return len(edges)-1

        df["bin_idx"] = df["abundance_pct"].map(assign_bin)
        df["bin_label"] = df["bin_idx"].map(lambda i: labels[i])

        # Use Legend order (or natural order if Legend is empty)
        ordered = list(legend_items.options) if legend_items.options else list(labels)
        ordered = [lab for lab in ordered if lab in labels] + [lab for lab in labels if lab not in ordered]

        # Colors
        bin_cmap = _color_maps.get(ABUNDANCE_BINS_FACTOR, {})
        if not bin_cmap:
            bin_cmap = {lab: palette[i] for i, lab in enumerate(labels)}
            _color_maps[ABUNDANCE_BINS_FACTOR] = bin_cmap

        for lab in ordered:
            sub = df[df["bin_label"] == lab]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter3d(
                x=sub[x_pc], y=sub[y_pc], z=sub[z_pc],
                mode="markers", name=lab,
                marker=dict(size=point_size, opacity=constant_opacity, color=bin_cmap.get(lab, "lightgray")),
                text=sub["sample"],
                hovertemplate="<b>%{text}</b><br>"+f"{x_pc}: "+"%{x:.3f}<br>"+f"{y_pc}: "+"%{y:.3f}<br>"+f"{z_pc}: "+"%{z:.3f}<extra></extra>",
            ))
    else:
        legend_factor, items, ordered = compute_legend_order(
            df.copy(), color_factor_sel, highlight_factor_sel, highlight_list
        )
        active_factor = legend_factor if legend_factor != "(None)" else None

        if active_factor is None:
            fig.add_trace(go.Scatter3d(
                x=df[x_pc], y=df[y_pc], z=df[z_pc],
                mode="markers", name="Samples",
                marker=dict(size=point_size, opacity=constant_opacity, color="lightgray"),
                text=df["sample"],
                hovertemplate="<b>%{text}</b><br>"+f"{x_pc}: "+"%{x:.3f}<br>"+f"{y_pc}: "+"%{y:.3f}<br>"+f"{z_pc}: "+"%{z:.3f}<extra></extra>",
            ))
        else:
            vals = df[active_factor].astype(str)
            _ensure_factor_colors(active_factor, ordered)
            # Gray "Other" for non-highlight when highlight mode active
            if effective_factor_name(highlight_factor_sel) == active_factor and len(highlight_list) > 0:
                keep = set(map(str, highlight_list))
                non_mask = ~vals.isin(keep)
                sub_non = df[non_mask]
                if not sub_non.empty:
                    fig.add_trace(go.Scatter3d(
                        x=sub_non[x_pc], y=sub_non[y_pc], z=sub_non[z_pc],
                        mode="markers", name="Other",
                        marker=dict(size=point_size, opacity=constant_opacity, color="lightgray"),
                        text=sub_non["sample"],
                        hovertemplate="<b>%{text}</b><br>"+f"{x_pc}: "+"%{x:.3f}<br>"+f"{y_pc}: "+"%{y:.3f}<br>"+f"{z_pc}: "+"%{z:.3f}<extra></extra>",
                    ))
                for cat in ordered:
                    sub = df[vals == str(cat)]
                    if sub.empty: continue
                    col = _color_maps.get(active_factor, {}).get(str(cat), "lightgray")
                    fig.add_trace(go.Scatter3d(
                        x=sub[x_pc], y=sub[y_pc], z=sub[z_pc],
                        mode="markers", name=str(cat),
                        marker=dict(size=point_size, opacity=constant_opacity, color=col),
                        text=sub["sample"],
                        hovertemplate="<b>%{text}</b><br>"+f"{x_pc}: "+"%{x:.3f}<br>"+f"{y_pc}: "+"%{y:.3f}<br>"+f"{z_pc}: "+"%{z:.3f}<extra></extra>",
                    ))
            else:
                for cat in ordered:
                    sub = df[vals == str(cat)]
                    if sub.empty: continue
                    col = _color_maps.get(active_factor, {}).get(str(cat), "lightgray")
                    fig.add_trace(go.Scatter3d(
                        x=sub[x_pc], y=sub[y_pc], z=sub[z_pc],
                        mode="markers", name=str(cat),
                        marker=dict(size=point_size, opacity=constant_opacity, color=col),
                        text=sub["sample"],
                        hovertemplate="<b>%{text}</b><br>"+f"{x_pc}: "+"%{x:.3f}<br>"+f"{y_pc}: "+"%{y:.3f}<br>"+f"{z_pc}: "+"%{z:.3f}<extra></extra>",
                    ))

    def axis_title(pc):
        idx = int(pc.replace("PC", "")) - 1
        return f"{pc} ({prop[idx]*100:.2f}%)"

    fig.update_layout(
        title=f"PCoA (3D) — {level}",
        legend=dict(itemsizing="constant", font=dict(size=int(legend_text_pt))),
        scene=dict(
            xaxis_title=axis_title(x_pc),
            yaxis_title=axis_title(y_pc),
            zaxis_title=axis_title(z_pc),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=legend_visible_flag,
    )
    # Square export sizing
    if export_flag:
        fig.update_layout(width=export_px, height=export_px)

    return pn.pane.Plotly(fig, config={"displaylogo": False}, sizing_mode="stretch_both")

# ============================================================
# Router
# ============================================================
def plot_pcoa(
    level, niche, plant,
    x_pc, y_pc, z_pc, use_3d,
    color_factor, shape_factor,
    filter_factor, filter_list,
    highlight_factor, highlight_list,
    abundance_flag,
    show_names_flag, msize, zoom,
    flip_x, flip_y,
    legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
    export_flag, export_px,
    legend_text_pt, legend_dot_px,
):
    if use_3d:
        return plot_pcoa_3d(
            level, niche, plant,
            x_pc, y_pc, z_pc,
            color_factor, shape_factor, filter_factor, filter_list,
            highlight_factor, highlight_list, abundance_flag, msize,
            legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
            export_flag, export_px,
            legend_text_pt, legend_dot_px
        )
    else:
        return plot_pcoa_2d(
            level, niche, plant,
            x_pc, y_pc,
            color_factor, shape_factor, filter_factor, filter_list,
            highlight_factor, highlight_list, abundance_flag,
            show_names_flag, msize, zoom, flip_x, flip_y,
            legend_visible_flag, legend_loc_sel, legend_x_px, legend_y_px,
            export_flag, export_px,
            legend_text_pt, legend_dot_px
        )

# ============================================================
# Bind & Layout / Serve
# ============================================================
def update_plot(
    level, niche, plant,
    x_pc, y_pc, z_pc,
    use_3d,
    color_factor,
    shape_factor,
    filter_factor, filter_list,
    highlight_factor, highlight_list,
    abundance_flag,
    show_names_flag, msize, zoom,
    flip_x, flip_y,
    _legend_tick,
    _legend_visible, _legend_loc, _legend_x, _legend_y,
    _export_mode, _export_size,
    _legend_text_size, _legend_dot_size,
):
    return plot_pcoa(
        level, niche, plant,
        x_pc, y_pc, z_pc,
        use_3d,
        color_factor, shape_factor,
        filter_factor, filter_list,
        highlight_factor, highlight_list,
        abundance_flag,
        show_names_flag, msize, zoom,
        flip_x, flip_y,
        _legend_visible, _legend_loc, _legend_x, _legend_y,
        _export_mode, _export_size,
        _legend_text_size, _legend_dot_size,
    )

update_plot_pane = pn.bind(
    update_plot,
    level_widget.param.value,
    niche_widget.param.value,
    plant_widget.param.value,
    pc_x_widget.param.value,
    pc_y_widget.param.value,
    pc_z_widget.param.value,
    toggle_3d.param.value,
    color_widget.param.value,
    shape_widget.param.value,
    filter_widget.param.value,
    filter_cats.param.value,
    highlight_widget.param.value,
    highlight_cats.param.value,
    abundance_toggle.param.value,
    show_names.param.value,
    marker_size.param.value,
    zoom_slider.param.value,
    flip_x_widget.param.value,
    flip_y_widget.param.value,
    legend_order_tick.param.value,
    legend_visible.param.value,
    legend_loc.param.value,
    legend_x.param.value,
    legend_y.param.value,
    export_mode.param.value,
    export_size.param.value,
    legend_text_size.param.value,
    legend_dot_size.param.value,
)

def _refresh_all(event=None):
    _refresh_legend_items()
    _push_factor_options_to_selectors()
    update_filter_options()
    update_highlight_options()
    _update_color_target_options()
    _update_color_controls_enabled()
    _update_color_picker_to_selection()
    _update_path_info()

for w in [bins_apply_toggle, level_widget, filter_widget, filter_cats, color_widget, highlight_widget, highlight_cats, legend_mode, toggle_3d,
          abundance_bin_mode, abundance_bin_edges, abundance_toggle, niche_widget, plant_widget, legend_text_size, legend_dot_size]:
    w.param.watch(lambda e: _refresh_all(), "value")

# ============================================================
# --- PRESETS: save / load current configuration ------------
# ============================================================
PRESET_FILE = "pcoa_viewer_presets.json"

def _load_all_presets() -> dict:
    if not os.path.exists(PRESET_FILE):
        return {}
    try:
        with open(PRESET_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_all_presets(presets: dict) -> None:
    with open(PRESET_FILE, "w") as f:
        json.dump(presets, f, indent=2)

presets_dict = _load_all_presets()
preset_names = sorted(presets_dict.keys())

preset_name_input = pn.widgets.TextInput(
    name="New preset name",
    placeholder="e.g. Andrzej_default",
    value="",
)

preset_select = pn.widgets.Select(
    name="Existing presets",
    options=preset_names if preset_names else [],
    value=preset_names[0] if preset_names else None,
)

save_preset_button = pn.widgets.Button(
    name="Save preset",
    button_type="primary",
)

load_preset_button = pn.widgets.Button(
    name="Load preset",
    button_type="default",
)

preset_status = pn.pane.Markdown("")

def _collect_current_config() -> dict:
    """Collect current widget/global state into a config dict."""
    return {
        # basic subset & PCs
        "level": level_widget.value,
        "niche": niche_widget.value,
        "plant": plant_widget.value,
        "x_pc": pc_x_widget.value,
        "y_pc": pc_y_widget.value,
        "z_pc": pc_z_widget.value,
        "use_3d": toggle_3d.value,

        # appearance / legend / export
        "color_factor": color_widget.value,
        "shape_factor": shape_widget.value,
        "highlight_factor": highlight_widget.value,
        "filter_factor": filter_widget.value,
        "filter_values": list(filter_cats.value),
        "highlight_values": list(highlight_cats.value),
        "abundance_toggle": abundance_toggle.value,
        "abundance_bin_mode": abundance_bin_mode.value,
        "abundance_bin_edges": abundance_bin_edges.value,

        "show_names": show_names.value,
        "marker_size": int(marker_size.value),
        "zoom": float(zoom_slider.value),
        "flip_x": flip_x_widget.value,
        "flip_y": flip_y_widget.value,

        "legend_mode": legend_mode.value,
        "legend_visible": legend_visible.value,
        "legend_loc": legend_loc.value,
        "legend_x": int(legend_x.value),
        "legend_y": int(legend_y.value),
        "legend_text_size": int(legend_text_size.value),
        "legend_dot_size": int(legend_dot_size.value),
        "legend_color_target": legend_color_target.value,

        "export_mode": export_mode.value,
        "export_size": int(export_size.value),

        # binning
        "bins_apply_toggle": bins_apply_toggle.value,
        "bin_factor_sel": bin_factor_sel.value,
        "bin_unmapped_mode": bin_unmapped_mode.value,
        "bin_other_label": bin_other_label.value,
        "bin_maps": _bin_maps,
        "bin_unmapped": _bin_unmapped,

        # legend state & colors
        "legend_order": list(_current_legend_order),
        "color_maps": _color_maps,
    }

def _apply_config(config: dict):
    """Best-effort apply a config dict to current widgets/global state."""
    global _bin_maps, _bin_unmapped, _color_maps

    applied = []
    skipped = []

    # ------- subset & level -------
    level = config.get("level")
    if level in LEVELS:
        level_widget.value = level
        applied.append("level")
    else:
        skipped.append("level")

    # refresh niche/plant options for this level
    _update_data_source_choices()

    niche_val = config.get("niche")
    if niche_val in niche_widget.options:
        niche_widget.value = niche_val
        applied.append("niche")
    else:
        skipped.append("niche")

    plant_val = config.get("plant")
    if plant_val in plant_widget.options:
        plant_widget.value = plant_val
        applied.append("plant")
    else:
        skipped.append("plant")

    # PCs + 3D
    for key, widget, opts in [
        ("x_pc", pc_x_widget, pc_x_widget.options),
        ("y_pc", pc_y_widget, pc_y_widget.options),
        ("z_pc", pc_z_widget, pc_z_widget.options),
    ]:
        val = config.get(key)
        if val in opts:
            widget.value = val
            applied.append(key)
        else:
            skipped.append(key)

    use_3d = config.get("use_3d")
    if isinstance(use_3d, bool) and not toggle_3d.disabled:
        toggle_3d.value = use_3d
        applied.append("use_3d")
    else:
        skipped.append("use_3d")

    # ------- binning globals first, since they affect factor options -------
    _bin_maps = config.get("bin_maps", _bin_maps)
    _bin_unmapped = config.get("bin_unmapped", _bin_unmapped)
    _refresh_bin_preview()
    _push_factor_options_to_selectors()

    # bin UI
    if "bin_factor_sel" in config and config["bin_factor_sel"] in bin_factor_sel.options:
        bin_factor_sel.value = config["bin_factor_sel"]; applied.append("bin_factor_sel")
    else:
        skipped.append("bin_factor_sel")

    if "bin_unmapped_mode" in config:
        if config["bin_unmapped_mode"] in bin_unmapped_mode.options:
            bin_unmapped_mode.value = config["bin_unmapped_mode"]; applied.append("bin_unmapped_mode")
        else:
            skipped.append("bin_unmapped_mode")
    if "bin_other_label" in config:
        bin_other_label.value = config["bin_other_label"]; applied.append("bin_other_label")

    if "bins_apply_toggle" in config:
        bins_apply_toggle.value = bool(config["bins_apply_toggle"]); applied.append("bins_apply_toggle")

    # refresh factor selectors again after binning
    _push_factor_options_to_selectors()

    # ------- factor selectors -------
    def _apply_factor(key, widget):
        val = config.get(key)
        if val in widget.options:
            widget.value = val
            applied.append(key)
        else:
            skipped.append(key)

    _apply_factor("color_factor", color_widget)
    _apply_factor("shape_factor", shape_widget)
    _apply_factor("highlight_factor", highlight_widget)
    _apply_factor("filter_factor", filter_widget)

    # filter & highlight categories
    # let watchers repopulate options before setting values
    update_filter_options()
    update_highlight_options()

    f_vals = config.get("filter_values", [])
    valid_f = set(filter_cats.options)
    filter_cats.value = [v for v in f_vals if v in valid_f]
    applied.append("filter_values")

    h_vals = config.get("highlight_values", [])
    valid_h = set(highlight_cats.options)
    highlight_cats.value = [v for v in h_vals if v in valid_h]
    applied.append("highlight_values")

    # abundance settings
    if "abundance_toggle" in config:
        abundance_toggle.value = bool(config["abundance_toggle"]); applied.append("abundance_toggle")
    if "abundance_bin_mode" in config and config["abundance_bin_mode"] in abundance_bin_mode.options:
        abundance_bin_mode.value = config["abundance_bin_mode"]; applied.append("abundance_bin_mode")
    if "abundance_bin_edges" in config:
        abundance_bin_edges.value = config["abundance_bin_edges"]; applied.append("abundance_bin_edges")

    # label, size, zoom, flips
    if "show_names" in config:
        show_names.value = bool(config["show_names"]); applied.append("show_names")
    if "marker_size" in config:
        marker_size.value = int(config["marker_size"]); applied.append("marker_size")
    if "zoom" in config:
        zoom_slider.value = float(config["zoom"]); applied.append("zoom")
    if "flip_x" in config:
        flip_x_widget.value = bool(config["flip_x"]); applied.append("flip_x")
    if "flip_y" in config:
        flip_y_widget.value = bool(config["flip_y"]); applied.append("flip_y")

    # legend
    if "legend_mode" in config and config["legend_mode"] in legend_mode.options:
        legend_mode.value = config["legend_mode"]; applied.append("legend_mode")
    if "legend_visible" in config:
        legend_visible.value = bool(config["legend_visible"]); applied.append("legend_visible")
    if "legend_loc" in config and config["legend_loc"] in legend_loc.options:
        legend_loc.value = config["legend_loc"]; applied.append("legend_loc")
    if "legend_x" in config:
        legend_x.value = int(config["legend_x"]); applied.append("legend_x")
    if "legend_y" in config:
        legend_y.value = int(config["legend_y"]); applied.append("legend_y")
    if "legend_text_size" in config:
        legend_text_size.value = int(config["legend_text_size"]); applied.append("legend_text_size")
    if "legend_dot_size" in config:
        legend_dot_size.value = int(config["legend_dot_size"]); applied.append("legend_dot_size")

    # export
    if "export_mode" in config:
        export_mode.value = bool(config["export_mode"]); applied.append("export_mode")
    if "export_size" in config:
        export_size.value = int(config["export_size"]); applied.append("export_size")

    # color maps & legend order
    _color_maps = config.get("color_maps", _color_maps)
    legend_order = config.get("legend_order", [])
    if legend_order:
        _set_legend_items(legend_order)
        applied.append("legend_order")
    else:
        _refresh_legend_items()

    # legend color target
    _update_color_target_options()
    if "legend_color_target" in config and config["legend_color_target"] in legend_color_target.options:
        legend_color_target.value = config["legend_color_target"]; applied.append("legend_color_target")
    else:
        skipped.append("legend_color_target")

    # final UI refresh
    _refresh_all()

    msg = f"Loaded preset (applied: {', '.join(applied) or 'none'}"
    if skipped:
        msg += f"; skipped: {', '.join(skipped)})"
    else:
        msg += ")"
    preset_status.object = msg

def _on_save_preset(event=None):
    name = preset_name_input.value.strip()
    if not name:
        preset_status.object = "**Please enter a preset name.**"
        return
    config = _collect_current_config()
    presets_dict[name] = config
    try:
        _save_all_presets(presets_dict)
    except Exception as e:
        preset_status.object = f"Error saving preset: `{e}`"
        return
    names = sorted(presets_dict.keys())
    preset_select.options = names
    preset_select.value = name
    preset_status.object = f"✅ Preset **{name}** saved."

def _on_load_preset(event=None):
    name = preset_select.value
    if not name or name not in presets_dict:
        preset_status.object = "**No preset selected or preset not found.**"
        return
    config = presets_dict.get(name, {})
    _apply_config(config)

save_preset_button.on_click(_on_save_preset)
load_preset_button.on_click(_on_load_preset)

presets_card = pn.Column(
    "### Presets",
    preset_name_input,
    pn.Row(save_preset_button, load_preset_button),
    preset_select,
    preset_status,
    sizing_mode="stretch_width",
)

# ---------- UI Layout ----------
data_source_card = pn.Column(
    pn.Row(level_widget, niche_widget, plant_widget),
    path_info,
    sizing_mode="stretch_width"
)

top_row = pn.GridBox(
    pc_x_widget, pc_y_widget, pc_z_widget,
    pn.Row(flip_x_widget, flip_y_widget), toggle_3d,
    ncols=4, sizing_mode="stretch_width"
)

filter_card = pn.Column(filter_widget, pn.Row(filter_cats, pn.Column(filter_sort)), sizing_mode="stretch_width")

# Appearance card (tiny hint about shapes)
appearance_card = pn.Column(
    color_widget, shape_widget,
    pn.pane.Markdown("_Shapes cycle through: circle, square, triangle, diamond, cross, asterisk._"),
    sizing_mode="stretch_width"
)

highlight_card = pn.Column(
    highlight_widget,
    pn.Row(
        highlight_cats,
        pn.Column(
            highlight_sort,
            abundance_toggle,
            abundance_bin_mode,
            abundance_bin_edges,
            abundance_bin_preview
        )
    ),
    sizing_mode="stretch_width"
)
legend_card = pn.Column(
    pn.Row(legend_mode, pn.Spacer(width=20), pn.Column(pn.pane.Markdown("Legend Color Target"), legend_color_target)),
    pn.Row(legend_items, pn.Column(legend_up_btn, legend_down_btn, legend_top_btn, legend_bottom_btn, legend_reset_btn)),
    pn.layout.Divider(),
    pn.Row(legend_visible, legend_loc),
    pn.Row(legend_x, legend_y),
    pn.Row(legend_text_size, legend_dot_size),
    pn.pane.Markdown("_Note: Legend dot size applies to 2D only._"),
    sizing_mode="stretch_width"
)
colors_card = pn.Column(color_picker, pn.Row(apply_color_btn, reset_colors_btn), color_info, sizing_mode="stretch_width")
labels_card = pn.Column(pn.Row(show_names, marker_size, zoom_slider), sizing_mode="stretch_width")
binning_card = pn.Column(
    pn.Row(bin_factor_sel, bins_apply_toggle, bin_unmapped_mode),
    pn.Row(bin_other_label),
    pn.Row(bin_source_cats, pn.Column(bin_name_input, bin_add_btn, bin_reset_btn)),
    bin_preview,
    sizing_mode="stretch_width"
)

accordion_items = [
    ("Filter", filter_card),
    ("Appearance", appearance_card),
    ("Highlight", highlight_card),
    ("Legend", legend_card),
    ("Colors", colors_card),
    ("Labels & Size", labels_card),
    ("Dynamic Binning", binning_card),
]
if not PLOTLY_AVAILABLE:
    accordion_items.insert(0, ("3D Help", plotly_help))

accordion = pn.Accordion(*accordion_items, active=[0, 2, 3, 4], sizing_mode="stretch_width")

menus_container = pn.Column(
    data_source_card,
    top_row,
    presets_card,          # --- PRESETS: added here ---
    accordion,
    sizing_mode="stretch_width"
)

plot_area = pn.panel(update_plot_pane, sizing_mode="stretch_both", min_height=600)

def _toggle_menus(event=None):
    # Hide the whole control stack when either "Hide all menus" or "Export" is active
    hide = hide_menus.value or export_mode.value
    menus_container.visible = not hide

hide_menus.param.watch(lambda e: _toggle_menus(), "value")
export_mode.param.watch(lambda e: _toggle_menus(), "value")
_toggle_menus()  # initialize

# Always-visible header: both toggles here
header_bar = pn.Row(
    hide_menus,
    pn.Spacer(width=16),
    export_mode, export_size,
    sizing_mode="stretch_width"
)

root = pn.Column(
    header_bar,
    menus_container,
    pn.layout.Divider(),
    plot_area,
    legend_order_tick,  # hidden tick
    sizing_mode="stretch_both"
)

if __name__ == "__main__":
    pn.serve(
        root,
        address="0.0.0.0",
        port=5008,
        allow_websocket_origin=["127.0.0.1:5008", "localhost:5008"],
        show=True,
    )

