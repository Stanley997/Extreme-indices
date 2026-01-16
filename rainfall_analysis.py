# ============================================================
# FULL CLEAN SCRIPT (Python) - Compute-only (Plotting handled in R)
#
# TASKS:
# PAPER 1
# 1) First exceedance decade map (grey = none by 2100) for SSP370 & SSP585
#    - Spectral_r discrete decades, ONE bottom colorbar
# 2) Magnitude of exceedance map (STANDARDISED, z-score; max over 2015–2100)
#    - z-score computed using historical mean/std
#    - magnitude is max z during years that exceed historical record
#    - ONE bottom colorbar, shared scale
#
# PAPER 2 (now using Persistence instead of Probability)
# 3) Persistence map: longest consecutive run of exceedance years (2015–2100)
#    - ONE bottom colorbar, shared scale
# 4) Counts: how many grid cells have persistence >= {1,2,3,5,10}
#
# COLOR RULES (your requirement):
# - WHITE = no data / invalid (NaN)
# - GREY  = no change (no exceedance / 0)
#
# IMPORTANT FIX (recommended):
# - rechunk flag time dimension to a single chunk before apply_ufunc:
#   flag.chunk({"time": -1})
#   This resolves the ValueError you hit.
# ------------------------------------------------------------
# Plotting:
# - Use the companion R script `plot_unprecedented_outputs.R` to generate maps.
# ============================================================

from __future__ import annotations

import numpy as np
import xarray as xr
import xclim as xc
import pandas as pd


# -----------------------
# PATHS & SETTINGS
# -----------------------
BASE_DIR = r"C:\Users\Stanley\Downloads"

HIST_FILE   = rf"{BASE_DIR}\AF_Pr_day_CanESM5_historical_1950_2014.nc"
SSP370_FILE = rf"{BASE_DIR}\AF_Pr_day_CanESM5_ssp370_2015_2100.nc"
SSP585_FILE = rf"{BASE_DIR}\AF_Pr_day_CanESM5_ssp585_2015_2100.nc"

OUT_NC  = rf"{BASE_DIR}\unprecedented_outputs_ssp370_ssp585.nc"
OUT_CSV = rf"{BASE_DIR}\persistence_counts_ssp370_ssp585.csv"

PR_VAR   = "pr"
LAT_NAME = "lat"
LON_NAME = "lon"

HIST_START, HIST_END = "1950-01-01", "2014-12-31"
FUT_START, FUT_END   = "2015-01-01", "2100-12-31"

FREQ = "YS"
CHUNKS = {"time": 365, "lat": 292, "lon": 280}

CDD_THRESH = "1 mm/day"

# Choose one:
PLOT_INDEX = "rx1day"  # "rx1day" | "rx5day" | "cdd"

FORCE_PR_CONVERSION = False
IMPOSSIBLE_MM_DAY_Q99 = 1000.0

# Magnitude (z-score) scaling
Z_VMAX_PERCENTILE = 99.0   # shared vmax across SSPs (robust to outliers)
Z_VMAX_FLOOR = 10.0        # ensure contrast


# -----------------------
# HELPERS
# -----------------------
def _rename_latlon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if LAT_NAME not in ds.coords:
        for cand in ["latitude", "nav_lat", "y"]:
            if cand in ds.coords:
                rename[cand] = LAT_NAME
                break
    if LON_NAME not in ds.coords:
        for cand in ["longitude", "nav_lon", "x"]:
            if cand in ds.coords:
                rename[cand] = LON_NAME
                break
    return ds.rename(rename) if rename else ds


def _safe_pr_to_mmday(pr: xr.DataArray) -> xr.DataArray:
    pr = pr.copy()
    units = (pr.attrs.get("units") or "").strip().lower()

    if FORCE_PR_CONVERSION:
        pr2 = xc.core.units.convert_units_to(pr, "mm/day")
        pr2.name = "pr"
        return pr2

    # Already daily totals
    if "mm" in units and ("day" in units or "d-1" in units or "/day" in units):
        pr.name = "pr"
        return pr

    # Not the suspicious flux case
    if not ("kg" in units and "s-1" in units):
        pr2 = xc.core.units.convert_units_to(pr, "mm/day")
        pr2.name = "pr"
        return pr2

    # Suspicious flux: sanity-check magnitudes using q99 on a mid-domain sample
    lon = LON_NAME if LON_NAME in pr.coords else "lon"
    lat = LAT_NAME if LAT_NAME in pr.coords else "lat"
    i0 = pr.sizes[lat] // 4
    j0 = pr.sizes[lon] // 4

    sample = pr.isel(
        {
            lon: slice(j0, min(j0 + 50, pr.sizes[lon])),
            lat: slice(i0, min(i0 + 50, pr.sizes[lat])),
            "time": slice(0, min(365, pr.sizes.get("time", 365))),
        }
    )

    finite = sample.where(np.isfinite(sample))
    try:
        q99 = float(finite.quantile(0.99, skipna=True).compute().item())
    except Exception:
        q99 = np.nan

    q99_converted = q99 * 86400.0 if np.isfinite(q99) else np.nan
    print(f"[unit-check] units='{units}', q99={q99}, q99*86400={q99_converted}")

    if not np.isfinite(q99):
        pr2 = xc.core.units.convert_units_to(pr, "mm/day")
        pr2.name = "pr"
        print("[unit-check] Sample all-NaN -> converting assuming true kg m-2 s-1 flux.")
        return pr2

    if q99_converted > IMPOSSIBLE_MM_DAY_Q99:
        pr.attrs["units"] = "mm/day"
        pr.name = "pr"
        print("[unit-check] Looks like daily totals already -> NOT converting; set units to mm/day.")
        return pr

    pr2 = xc.core.units.convert_units_to(pr, "mm/day")
    pr2.name = "pr"
    print("[unit-check] Converting from kg m-2 s-1 to mm/day.")
    return pr2


def _compute_indices(pr_mmday: xr.DataArray) -> dict[str, xr.DataArray]:
    rx1day = pr_mmday.resample(time=FREQ).max("time", skipna=True)
    rx1day.name = "rx1day"

    pr5 = pr_mmday.rolling(time=5, min_periods=5).sum()
    rx5day = pr5.resample(time=FREQ).max("time", skipna=True)
    rx5day.name = "rx5day"

    cdd = xc.indices.maximum_consecutive_dry_days(pr_mmday, thresh=CDD_THRESH, freq=FREQ)
    cdd.name = "cdd"

    return {"rx1day": rx1day, "rx5day": rx5day, "cdd": cdd}


def _split_hist_fut(idx: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    hist = idx.sel(time=slice(HIST_START, HIST_END))
    fut  = idx.sel(time=slice(FUT_START, FUT_END))
    return hist, fut


def first_exceedance_year(flag: xr.DataArray) -> xr.DataArray:
    years = flag["time"].dt.year
    year_vals = xr.where(flag, years, np.nan)
    return year_vals.min("time", skipna=True).astype("float32")


def first_exceedance_decade(foe_year: xr.DataArray) -> xr.DataArray:
    return (np.floor(foe_year / 10) * 10).astype("float32")


def frequency_after_emergence(flag: xr.DataArray, foe_year: xr.DataArray) -> xr.DataArray:
    years = flag["time"].dt.year
    after = xr.where(np.isfinite(foe_year), years >= foe_year, False)
    return (flag & after).sum("time").astype("int16")


def _report_zscale(zmag_370: xr.DataArray, zmag_585: xr.DataArray) -> float:
    finite_z = np.concatenate(
        [
            zmag_370.values[np.isfinite(zmag_370.values)],
            zmag_585.values[np.isfinite(zmag_585.values)],
        ]
    )
    if finite_z.size == 0:
        return Z_VMAX_FLOOR
    z_vmax = float(np.nanpercentile(finite_z, Z_VMAX_PERCENTILE))
    return max(z_vmax, Z_VMAX_FLOOR)


# -----------------------
# Persistence (longest consecutive run of exceedance years)
# -----------------------
def _max_run_bool_1d(a: np.ndarray) -> np.int16:
    # a: 1D boolean array (time)
    if a.size == 0:
        return np.int16(0)
    a = np.asarray(a, dtype=bool)
    best = 0
    cur = 0
    for v in a:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return np.int16(best)


def persistence_longest_run(flag: xr.DataArray) -> xr.DataArray:
    # flag dims: time, lat, lon -> returns lat, lon
    return xr.apply_ufunc(
        _max_run_bool_1d,
        flag,
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int16],
    ).astype("int16")


def count_persistence_cells(persist: xr.DataArray, thresholds=(1, 2, 3, 5, 10)) -> dict[int, int]:
    out = {}
    valid = np.isfinite(persist.values)
    for k in thresholds:
        out[k] = int(np.sum(valid & (persist.values >= k)))
    return out


# -----------------------
# MAIN
# -----------------------
print("Opening NetCDF files...")
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

ds_hist = xr.open_dataset(HIST_FILE,   chunks=CHUNKS, decode_times=time_coder)[[PR_VAR]]
ds_370  = xr.open_dataset(SSP370_FILE, chunks=CHUNKS, decode_times=time_coder)[[PR_VAR]]
ds_585  = xr.open_dataset(SSP585_FILE, chunks=CHUNKS, decode_times=time_coder)[[PR_VAR]]

ds_hist = _rename_latlon(ds_hist)
ds_370  = _rename_latlon(ds_370)
ds_585  = _rename_latlon(ds_585)

print("Preparing precipitation as mm/day (safe unit handling) ...")
pr_hist = _safe_pr_to_mmday(ds_hist[PR_VAR])
pr_370  = _safe_pr_to_mmday(ds_370[PR_VAR])
pr_585  = _safe_pr_to_mmday(ds_585[PR_VAR])

print("Computing annual indices ...")
idx_hist = _compute_indices(pr_hist)[PLOT_INDEX]
idx_370  = _compute_indices(pr_370)[PLOT_INDEX]
idx_585  = _compute_indices(pr_585)[PLOT_INDEX]

print("Splitting baseline vs future ...")
hist, _     = _split_hist_fut(idx_hist)
_, fut_370  = _split_hist_fut(idx_370)
_, fut_585  = _split_hist_fut(idx_585)

print("Computing baseline record (historical max per grid cell) ...")
hist_max = hist.max("time", skipna=True).astype("float32")

print("Computing exceedance flags (future > historical record) ...")
flag_370 = (fut_370 > hist_max)
flag_585 = (fut_585 > hist_max)

# -----------------------
# PAPER 1: First emergence
# -----------------------
print("Computing first emergence year/decade ...")
foe_year_370 = first_exceedance_year(flag_370).compute()
foe_year_585 = first_exceedance_year(flag_585).compute()
foe_dec_370  = first_exceedance_decade(foe_year_370)
foe_dec_585  = first_exceedance_decade(foe_year_585)

# -----------------------
# PAPER 1: Magnitude (STANDARDISED z-score, no-change=0)
# -----------------------
print("Computing standardised magnitude (z-score) ...")
hist_mean = hist.mean("time", skipna=True).astype("float32")
hist_std  = hist.std("time", skipna=True, ddof=1).astype("float32")
hist_std_safe = hist_std.where(hist_std > 0)  # std==0 -> NaN (white)

# no exceedance -> 0 (grey); true missing stays NaN (white)
z_370 = ((fut_370 - hist_mean) / hist_std_safe).where(flag_370, 0.0)
z_585 = ((fut_585 - hist_mean) / hist_std_safe).where(flag_585, 0.0)

zmag_370 = z_370.max("time", skipna=True).astype("float32").compute()
zmag_585 = z_585.max("time", skipna=True).astype("float32").compute()

z_vmax = _report_zscale(zmag_370, zmag_585)

# -----------------------
# PAPER 2: Persistence (RECOMMENDED FIX: rechunk time to one chunk)
# -----------------------
print("Computing persistence (longest consecutive run of exceedance years) ...")
flag_370_1c = flag_370.chunk({"time": -1})
flag_585_1c = flag_585.chunk({"time": -1})

persist_370 = persistence_longest_run(flag_370_1c).compute()
persist_585 = persistence_longest_run(flag_585_1c).compute()

finite_p = np.concatenate(
    [
        persist_370.values[np.isfinite(persist_370.values)],
        persist_585.values[np.isfinite(persist_585.values)],
    ]
)
p_vmax = float(np.nanmax(finite_p)) if finite_p.size else 1.0
p_vmax = max(p_vmax, 1.0)

# Keep frequency-after-emergence too (useful)
freq_after_370 = frequency_after_emergence(flag_370, foe_year_370).compute()
freq_after_585 = frequency_after_emergence(flag_585, foe_year_585).compute()

# -----------------------
# Counts: how many grid cells have persistence >= k
# -----------------------
thresholds = (1, 2, 3, 5, 10)
counts_370 = count_persistence_cells(persist_370, thresholds=thresholds)
counts_585 = count_persistence_cells(persist_585, thresholds=thresholds)

print("\nPERSISTENCE COUNTS (number of grid cells):")
print("SSP370:", counts_370)
print("SSP585:", counts_585)

pd.DataFrame(
    {
        "threshold_years": list(thresholds),
        "gridcells_ssp370": [counts_370[k] for k in thresholds],
        "gridcells_ssp585": [counts_585[k] for k in thresholds],
    }
).to_csv(OUT_CSV, index=False)
print(f"✅ Wrote: {OUT_CSV}")

# -----------------------
# SAVE NetCDF
# -----------------------
print("Saving outputs (NetCDF) ...")
out = xr.Dataset(
    {
        "hist_max": hist_max,
        "hist_mean": hist_mean,
        "hist_std": hist_std,
        "foe_year_ssp370": foe_year_370,
        "foe_year_ssp585": foe_year_585,
        "foe_decade_ssp370": foe_dec_370,
        "foe_decade_ssp585": foe_dec_585,
        "max_std_exceed_z_ssp370": zmag_370,
        "max_std_exceed_z_ssp585": zmag_585,
        "persistence_longest_run_ssp370": persist_370,
        "persistence_longest_run_ssp585": persist_585,
        "freq_after_emergence_ssp370": freq_after_370,
        "freq_after_emergence_ssp585": freq_after_585,
    }
)
out.attrs.update(
    {
        "z_vmax_percentile": Z_VMAX_PERCENTILE,
        "z_vmax_floor": Z_VMAX_FLOOR,
        "z_vmax": float(z_vmax),
        "persistence_vmax": float(p_vmax),
    }
)
encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
out.to_netcdf(OUT_NC, encoding=encoding)
print(f"✅ Wrote: {OUT_NC}")

print("\nDONE ✅")
print(f"Outputs saved:\n- NetCDF: {OUT_NC}\n- CSV: {OUT_CSV}")
print(
    "Suggested plot scales (for R):\n"
    f"- z_vmax (percentile={Z_VMAX_PERCENTILE}): {z_vmax:.2f}\n"
    f"- persistence vmax: {p_vmax:.0f}"
)
