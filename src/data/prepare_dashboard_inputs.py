import xarray as xr
import os


SOURCE = "data/processed/oceanembed_training_30days.nc"

OUTPUT = (
    "data/processed/"
    "oceanembed_dashboard_inputs_4days.nc"
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


TEST_DATES = [
    "2020-01-27",
    "2020-01-28",
    "2020-01-29",
    "2020-01-30",
]


print("=" * 65)
print("PREPARING OCEANEMBED DASHBOARD INPUT DATA")
print("=" * 65)


ds = xr.open_dataset(
    SOURCE
)


dashboard = ds[
    INPUT_VARIABLES
].sel(
    time=TEST_DATES
)


dashboard.attrs.update({
    "project": "OceanEmbed",
    "problem_statement": "SIH26066",
    "purpose": "Interactive dashboard surface inputs",
    "input_channels": (
        "SST, SSS, SSH, U current, V current, "
        "U wind, V wind"
    ),
    "resolution": "0.25 degree",
    "dates": "2020-01-27 to 2020-01-30",
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