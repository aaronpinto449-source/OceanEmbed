import xarray as xr
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# OceanEmbed - Build multi-day training dataset
# ---------------------------------------------------------

RAW = Path("data/raw")
OUT = Path("data/processed")

dates = [
    f"2020-01-{day:02d}"
    for day in range(1, 31)
]
# SIH target grid
target_lat = np.arange(5.0, 20.0 + 0.001, 0.25)
target_lon = np.arange(80.0, 90.0 + 0.001, 0.25)

target_depths = np.array([
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
], dtype=np.float32)


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def squeeze_time(da):
    if "time" in da.dims and da.sizes["time"] == 1:
        return da.isel(time=0, drop=True)
    return da


# ---------------------------------------------------------
# Process one date
# ---------------------------------------------------------

def process_date(date):

    print()
    print("=" * 60)
    print(f"Processing {date}")
    print("=" * 60)

    glorys_file = RAW / f"glorys_bob_{date}.nc"
    sss_file = RAW / f"sss_bob_{date}.nc"
    wind_file = RAW / f"wind_bob_{date}.nc"

    # Check files exist
    missing = []

    for file in [glorys_file, sss_file, wind_file]:
        if not file.exists():
            missing.append(str(file))

    if missing:
        print("Missing files:")
        for file in missing:
            print(f"  {file}")

        return None

    print("Loading datasets...")

    glorys = xr.open_dataset(glorys_file)
    sss = xr.open_dataset(sss_file)
    wind = xr.open_dataset(wind_file)

    # -----------------------------------------------------
    # GLORYS
    # -----------------------------------------------------

    print("Processing GLORYS...")

    # Surface temperature
    sst = glorys["thetao"].isel(depth=0)

    # Surface currents
    u_current = glorys["uo"].isel(depth=0)
    v_current = glorys["vo"].isel(depth=0)

    # Sea surface height
    ssh = glorys["zos"]

    # -----------------------------------------------------
    # Target subsurface temperature
    # -----------------------------------------------------

    temperature = glorys["thetao"].interp(
        depth=target_depths
    )

    # GLORYS first level is approximately 0.494 m.
    # Use it as the SIH 0 m surface value.
    temperature.loc[dict(depth=0)] = (
        glorys["thetao"].isel(depth=0)
    )

    # -----------------------------------------------------
    # Regrid GLORYS
    # -----------------------------------------------------

    sst = sst.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    u_current = u_current.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    v_current = v_current.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    ssh = ssh.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    temperature = temperature.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    # -----------------------------------------------------
    # SSS
    # -----------------------------------------------------

    print("Processing SSS...")

    sss_data = sss["sos"].isel(depth=0)

    sss_data = sss_data.interp(
        latitude=target_lat,
        longitude=target_lon
    )

    # -----------------------------------------------------
    # WIND
    # -----------------------------------------------------

    print("Processing wind...")

    u_wind = wind["eastward_wind"]
    v_wind = wind["northward_wind"]

    # Average 24 hourly fields into a daily mean
    u_wind = u_wind.mean(dim="time")
    v_wind = v_wind.mean(dim="time")

    # Regrid to SIH 0.25 degree grid
    u_wind = u_wind.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"}
    )

    v_wind = v_wind.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"}
    )

    # -----------------------------------------------------
    # Remove time dimensions
    # -----------------------------------------------------

    sst = squeeze_time(sst)
    sss_data = squeeze_time(sss_data)
    ssh = squeeze_time(ssh)
    u_current = squeeze_time(u_current)
    v_current = squeeze_time(v_current)
    u_wind = squeeze_time(u_wind)
    v_wind = squeeze_time(v_wind)
    temperature = squeeze_time(temperature)

    # -----------------------------------------------------
    # Build daily dataset
    # -----------------------------------------------------

    daily = xr.Dataset(
        {
            "sst": sst,
            "sss": sss_data,
            "ssh": ssh,
            "u_current": u_current,
            "v_current": v_current,
            "u_wind": u_wind,
            "v_wind": v_wind,
            "temperature": temperature,
        }
    )

    daily["temperature"] = daily["temperature"].transpose(
        "depth",
        "latitude",
        "longitude"
    )

    # Add time dimension
    daily = daily.expand_dims(
        time=[np.datetime64(date)]
    )

    print("Daily sample created.")

    print(
        f"Grid: {len(target_lat)} lat × "
        f"{len(target_lon)} lon"
    )

    return daily


# ---------------------------------------------------------
# Build all dates
# ---------------------------------------------------------

print()
print("========================================")
print("OceanEmbed Multi-Day Dataset Builder")
print("========================================")

daily_datasets = []

for date in dates:

    daily = process_date(date)

    if daily is not None:
        daily_datasets.append(daily)


# ---------------------------------------------------------
# Check results
# ---------------------------------------------------------

if len(daily_datasets) == 0:

    raise RuntimeError(
        "No daily datasets were created. "
        "Check that the required files exist in data/raw."
    )


print()
print("=" * 60)
print("Combining daily datasets...")
print("=" * 60)

ds = xr.concat(
    daily_datasets,
    dim="time"
)


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

ds.attrs["project"] = "OceanEmbed"
ds.attrs["problem_statement"] = "SIH26066"
ds.attrs["region"] = "Bay of Bengal prototype"

ds.attrs["input_channels"] = (
    "SST, SSS, SSH, U current, V current, "
    "U wind, V wind"
)

ds.attrs["target"] = (
    "GLORYS subsurface temperature"
)

ds.attrs["target_depths_m"] = str(
    target_depths.tolist()
)

ds.attrs["resolution"] = "0.25 degree"

ds.attrs["dates"] = (
    f"{dates[0]} to {dates[-1]}"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True
)

output_file = OUT / "oceanembed_training_30days.nc"


print()
print("Saving dataset...")
print(f"Output: {output_file}")

ds.to_netcdf(output_file)


# ---------------------------------------------------------
# Final information
# ---------------------------------------------------------

print()
print("========================================")
print("MULTI-DAY DATASET CREATED")
print("========================================")

print(ds)

print()
print("Number of days:", ds.sizes["time"])
print("Latitude:", ds.sizes["latitude"])
print("Longitude:", ds.sizes["longitude"])
print("Depth levels:", ds.sizes["depth"])

print()
print(f"Saved to: {output_file}")