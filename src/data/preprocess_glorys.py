from pathlib import Path
import xarray as xr
import numpy as np

# ---------------------------------------------------------
# OceanEmbed - GLORYS preprocessing
# ---------------------------------------------------------

INPUT = Path("data/raw/glorys_bob_2020-01-15.nc")
OUTPUT = Path("data/processed/glorys_bob_2020-01-15_025deg.nc")

# SIH26066 required depths
TARGET_DEPTHS = np.array([
    0, 5, 10, 20, 30,
    50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000
], dtype=float)

# Required spatial resolution
TARGET_LAT = np.arange(5.0, 20.0001, 0.25)
TARGET_LON = np.arange(80.0, 90.0001, 0.25)

print("Opening GLORYS dataset...")
ds = xr.open_dataset(INPUT)

print("\nOriginal dataset:")
print(ds)

# ---------------------------------------------------------
# 1. Horizontal regridding
# ---------------------------------------------------------

print("\nRegridding to 0.25° × 0.25°...")

ds_grid = ds.interp(
    latitude=TARGET_LAT,
    longitude=TARGET_LON,
    method="linear"
)

# ---------------------------------------------------------
# 2. Vertical interpolation
# ---------------------------------------------------------

print("Interpolating temperature to SIH depth levels...")

temperature = ds_grid["thetao"]

# Interpolate depths 5 m and deeper.
deep_targets = TARGET_DEPTHS[TARGET_DEPTHS > 0]

temperature_deep = temperature.interp(
    depth=deep_targets,
    method="linear"
)

# For 0 m, use the shallowest available GLORYS level.
surface_temperature = temperature.sel(
    depth=temperature.depth.values[0]
)

surface_temperature = surface_temperature.expand_dims(
    depth=[0.0]
)

# Combine surface + interpolated deeper levels
temperature_final = xr.concat(
    [
        surface_temperature,
        temperature_deep
    ],
    dim="depth"
)

# Make sure depth coordinate is exactly what SIH requires.
temperature_final = temperature_final.assign_coords(
    depth=TARGET_DEPTHS
)

temperature_final.name = "temperature"

temperature_final.attrs["units"] = "degrees_C"
temperature_final.attrs["description"] = (
    "GLORYS temperature interpolated to SIH26066 "
    "standard depths and 0.25 degree grid"
)

# ---------------------------------------------------------
# 3. Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

temperature_final.to_netcdf(OUTPUT)

print("\n==========================================")
print("PREPROCESSING COMPLETE")
print("==========================================")

print("\nFinal dataset:")
print(temperature_final)

print(f"\nSaved to:")
print(OUTPUT)