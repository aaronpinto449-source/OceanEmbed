from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt

path = Path("data/raw/glorys_bob_2020-01-15.nc")
ds = xr.open_dataset(path)
surface = ds["thetao"].isel(time=0, depth=0)

plt.figure(figsize=(9, 5))
surface.plot()
plt.title("GLORYS potential temperature — shallowest available depth")
plt.tight_layout()
plt.show()
