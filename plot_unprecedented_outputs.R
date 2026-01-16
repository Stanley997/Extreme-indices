#!/usr/bin/env Rscript

library(ncdf4)
library(terra)
library(viridisLite)
library(RColorBrewer)

nc_path <- Sys.getenv("UNPRECEDENTED_NC", unset = "C:/Users/Stanley/Downloads/unprecedented_outputs_ssp370_ssp585.nc")
out_dir <- Sys.getenv("UNPRECEDENTED_PLOT_DIR", unset = "C:/Users/Stanley/Downloads")

nc <- nc_open(nc_path)
z_vmax <- ncatt_get(nc, 0, "z_vmax")$value
p_vmax <- ncatt_get(nc, 0, "persistence_vmax")$value
nc_close(nc)

if (is.null(z_vmax) || is.na(z_vmax)) {
  z_vmax <- NA_real_
}
if (is.null(p_vmax) || is.na(p_vmax)) {
  p_vmax <- NA_real_
}

hist_max <- rast(nc_path, subds = "hist_max")
foe_370 <- rast(nc_path, subds = "foe_decade_ssp370")
foe_585 <- rast(nc_path, subds = "foe_decade_ssp585")
z_370 <- rast(nc_path, subds = "max_std_exceed_z_ssp370")
z_585 <- rast(nc_path, subds = "max_std_exceed_z_ssp585")
persist_370 <- rast(nc_path, subds = "persistence_longest_run_ssp370")
persist_585 <- rast(nc_path, subds = "persistence_longest_run_ssp585")

no_data_mask <- is.na(hist_max)
foe_370 <- ifel(is.na(foe_370) & !no_data_mask, 2000, foe_370)
foe_585 <- ifel(is.na(foe_585) & !no_data_mask, 2000, foe_585)

plot_pair <- function(r1, r2, title, colors, breaks, filename, na_col = "white") {
  png(filename, width = 1600, height = 800)
  par(mfrow = c(1, 2), mar = c(3, 3, 4, 6))
  plot(r1, col = colors, breaks = breaks, main = paste(title, "SSP370"), axes = TRUE, legend = TRUE,
       na.col = na_col)
  plot(r2, col = colors, breaks = breaks, main = paste(title, "SSP585"), axes = TRUE, legend = TRUE,
       na.col = na_col)
  dev.off()
}

dec_breaks <- seq(2010, 2110, by = 10)
dec_colors <- rev(brewer.pal(length(dec_breaks) - 1, "Spectral"))
dec_breaks <- c(2000, dec_breaks)
dec_colors <- c("#bdbdbd", dec_colors)

plot_pair(
  foe_370,
  foe_585,
  title = "First exceedance decade",
  colors = dec_colors,
  breaks = dec_breaks,
  filename = file.path(out_dir, "first_exceedance_decade.png")
)

if (is.na(z_vmax)) {
  z_vmax <- max(values(z_370), values(z_585), na.rm = TRUE)
}
z_breaks <- c(0, seq(0.1, z_vmax, length.out = 6))
z_colors <- c("#bdbdbd", viridis(length(z_breaks) - 2))
plot_pair(
  z_370,
  z_585,
  title = "Max standardized exceedance (z-score)",
  colors = z_colors,
  breaks = z_breaks,
  filename = file.path(out_dir, "max_standardized_exceedance.png")
)

if (is.na(p_vmax)) {
  p_vmax <- max(values(persist_370), values(persist_585), na.rm = TRUE)
}
p_breaks <- c(0, seq(1, p_vmax, by = 1))
p_colors <- c("#bdbdbd", plasma(length(p_breaks) - 2))
plot_pair(
  persist_370,
  persist_585,
  title = "Persistence (years)",
  colors = p_colors,
  breaks = p_breaks,
  filename = file.path(out_dir, "persistence_longest_run.png")
)
