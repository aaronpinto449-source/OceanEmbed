from pathlib import Path
import xarray as xr

path = Path("data/raw/glorys_bob_2020-01-15.nc")
if not path.exists():
    raise FileNotFoundError(f"{path} not found. Run download_glorys.py first.")

ds = xr.open_dataset(path)

print("\n=== DATASET ===")
print(ds)

print("\n=== VARIABLES ===")
for name, da in ds.data_vars.items():
    print(f"{name:10s} dims={da.dims} shape={da.shape} units={da.attrs.get('units', '?')}")
