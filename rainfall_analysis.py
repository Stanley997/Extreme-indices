# ============================================================
# FULL CLEAN SCRIPT (Python) - Compute-only (Plotting handled in R)
#
# TASKS:
# PAPER 1
# 1) First exceedance decade map (grey = none by 2100) for SSP370 & SSP585
#    - Spectral_r discrete decades, ONE bottom colorbar
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


# -----------------------
# PATHS & SETTINGS
# -----------------------
BASE_DIR = r"C:\Users\Stanley\Downloads"

HIST_FILE   = rf"{BASE_DIR}\AF_Pr_day_CanESM5_historical_1950_2014.nc"
SSP370_FILE = rf"{BASE_DIR}\AF_Pr_day_CanESM5_ssp370_2015_2100.nc"
SSP585_FILE = rf"{BASE_DIR}\AF_Pr_day_CanESM5_ssp585_2015_2100.nc"

OUT_NC  = rf"{BASE_DIR}\unprecedented_outputs_ssp370_ssp585.nc"

# Optional SSA + land mask (set APPLY_SSA_LAND_MASK=True to enable).
APPLY_SSA_LAND_MASK = False
SSA_SHAPEFILE = rf"{BASE_DIR}\ssa_boundary.shp"
LAND_SHAPEFILE = rf"{BASE_DIR}\land.shp"

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


def _load_ssa_land_regions():
    if not APPLY_SSA_LAND_MASK:
        return None
    try:
        import geopandas as gpd
        import regionmask
    except ImportError as exc:
        raise ImportError(
            "Enable APPLY_SSA_LAND_MASK requires geopandas and regionmask."
        ) from exc

    ssa = gpd.read_file(SSA_SHAPEFILE).to_crs("EPSG:4326")
    if LAND_SHAPEFILE:
        land = gpd.read_file(LAND_SHAPEFILE).to_crs("EPSG:4326")
        ssa = gpd.overlay(ssa, land, how="intersection")

    geometries = [geom for geom in ssa.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError("SSA/land mask geometry is empty. Check shapefiles.")

    return regionmask.Regions(outlines=geometries)


def _apply_region_mask(ds: xr.Dataset, regions) -> xr.Dataset:
    if regions is None:
        return ds
    mask = regions.mask(ds, lat_name=LAT_NAME, lon_name=LON_NAME)
    return ds.where(mask.notnull())


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

regions = _load_ssa_land_regions()
ds_hist = _apply_region_mask(ds_hist, regions)
ds_370  = _apply_region_mask(ds_370, regions)
ds_585  = _apply_region_mask(ds_585, regions)

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
# SAVE NetCDF
# -----------------------
print("Saving outputs (NetCDF) ...")
out = xr.Dataset(
    {
        "hist_max": hist_max,
        "foe_year_ssp370": foe_year_370,
        "foe_year_ssp585": foe_year_585,
        "foe_decade_ssp370": foe_dec_370,
        "foe_decade_ssp585": foe_dec_585,
    }
)
encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
out.to_netcdf(OUT_NC, encoding=encoding)
print(f"✅ Wrote: {OUT_NC}")

print("\nDONE ✅")
print(f"Outputs saved:\n- NetCDF: {OUT_NC}")
