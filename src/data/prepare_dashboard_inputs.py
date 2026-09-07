import xarray as xr
import os


SOURCE = "data/processed/oceanembed_training_30days.nc"

OUTPUT = (
    "data/processed/"
    "oceanembed_dashboard_inputs_30days.nc"
)


INPUT_VARIABLES = [
    "sst",
    "sss",
    "ssh",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
]


print("=" * 65)
print("PREPARING OCEANEMBED 30-DAY DASHBOARD INPUT DATA")
print("=" * 65)


ds = xr.open_dataset(
    SOURCE
)


dashboard = ds[
    INPUT_VARIABLES
]


dashboard.attrs.update({
    "project": "OceanEmbed",
    "problem_statement": "SIH26066",
    "purpose": "Interactive dashboard surface inputs",
    "input_channels": (
        "SST, SSS, SSH, U current, V current, "
        "U wind, V wind"
    ),
    "resolution": "0.25 degree",
    "dates": "2020-01-01 to 2020-01-30",
})


os.makedirs(
    "data/processed",
    exist_ok=True
)


dashboard.to_netcdf(
    OUTPUT
)


print(
    "\nCreated:",
    OUTPUT
)

print(
    "Dimensions:",
    dict(dashboard.sizes)
)

print(
    "Variables:",
    list(dashboard.data_vars)
)

print(
    "File size:",
    os.path.getsize(OUTPUT) / 1024 / 1024,
    "MB"
)

print("=" * 65)
