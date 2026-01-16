## Usage

### 1) Run the Python computation

Update the base path in `rainfall_analysis.py` if needed, then run:

```bash
python rainfall_analysis.py
```

By default the script reads/writes to:

```
C:\Users\Stanley\Downloads
```

Outputs:

- `unprecedented_outputs_ssp370_ssp585.nc`
- `persistence_counts_ssp370_ssp585.csv`

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
