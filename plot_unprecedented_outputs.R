#!/usr/bin/env Rscript

library(ncdf4)
library(terra)
library(RColorBrewer)
library(ggplot2)

nc_path <- Sys.getenv("UNPRECEDENTED_NC", unset = "C:/Users/Stanley/Downloads/unprecedented_outputs_ssp370_ssp585.nc")
out_dir <- Sys.getenv("UNPRECEDENTED_PLOT_DIR", unset = "C:/Users/Stanley/Downloads")

hist_max <- rast(nc_path, subds = "hist_max")
foe_370 <- rast(nc_path, subds = "foe_decade_ssp370")
foe_585 <- rast(nc_path, subds = "foe_decade_ssp585")

no_data_mask <- is.na(hist_max)
foe_370 <- ifel(is.na(foe_370) & !no_data_mask, 2000, foe_370)
foe_585 <- ifel(is.na(foe_585) & !no_data_mask, 2000, foe_585)

to_df <- function(r, label) {
  df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
  names(df) <- c("lon", "lat", "value")
  df$scenario <- label
  df
}

plot_pair_discrete <- function(r1, r2, title, breaks, colors, filename, legend_title) {
  df <- rbind(to_df(r1, "SSP370"), to_df(r2, "SSP585"))
  df$bin <- cut(df$value, breaks = breaks, include.lowest = TRUE, right = FALSE)
  df$bin <- addNA(df$bin)
  labels <- levels(df$bin)
  color_map <- c(colors, "#ffffff")
  names(color_map) <- c(labels, NA)

  p <- ggplot(df, aes(x = lon, y = lat, fill = bin)) +
    geom_raster() +
    facet_wrap(~scenario, ncol = 2) +
    coord_equal() +
    scale_fill_manual(values = color_map, drop = FALSE, name = legend_title) +
    labs(title = title, x = "Longitude", y = "Latitude") +
    theme_minimal()

  ggsave(filename, p, width = 12, height = 6, dpi = 150)
}

dec_breaks <- seq(2010, 2110, by = 10)
dec_colors <- rev(brewer.pal(length(dec_breaks) - 1, "Spectral"))
dec_breaks <- c(2000, dec_breaks)
dec_colors <- c("#bdbdbd", dec_colors)

plot_pair_discrete(
  foe_370,
  foe_585,
  title = "First exceedance decade",
  breaks = dec_breaks,
  colors = dec_colors,
  filename = file.path(out_dir, "first_exceedance_decade.png"),
  legend_title = "Decade (grey = none by 2100; white = no data)"
)
