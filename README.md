## Usage

### 1) Run the Python computation

Update the base path in `rainfall_analysis.py` if needed. To restrict computation to
sub-Saharan Africa land only, set `APPLY_SSA_LAND_MASK = True` and point
`SSA_SHAPEFILE` and `LAND_SHAPEFILE` to your shapefiles (EPSG:4326 recommended).
Then run:

```bash
python rainfall_analysis.py
```

By default the script reads/writes to:

```
C:\Users\Stanley\Downloads
```

Outputs:

- `unprecedented_outputs_ssp370_ssp585.nc`

### 2) Run the R plotting script

The R script reads the NetCDF file and writes PNG plots to the output directory.
If you keep the default path above, you can run:

```bash
Rscript plot_unprecedented_outputs.R
```

Or specify paths explicitly:

```bash
UNPRECEDENTED_NC="C:/Users/Stanley/Downloads/unprecedented_outputs_ssp370_ssp585.nc" \
UNPRECEDENTED_PLOT_DIR="C:/Users/Stanley/Downloads" \
Rscript plot_unprecedented_outputs.R
```
