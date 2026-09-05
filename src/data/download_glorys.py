from pathlib import Path
import copernicusmarine

LON_MIN, LON_MAX = 80.0, 90.0
LAT_MIN, LAT_MAX = 5.0, 20.0
DATE = "2020-01-15"
DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "glorys_bob_2020-01-15.nc"

if OUT_FILE.exists():
    print(f"Already exists: {OUT_FILE}")
    raise SystemExit

copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=["thetao", "so", "uo", "vo", "zos"],
    minimum_longitude=LON_MIN,
    maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN,
    maximum_latitude=LAT_MAX,
    start_datetime=f"{DATE}T00:00:00",
    end_datetime=f"{DATE}T00:00:00",
    minimum_depth=0,
    maximum_depth=1000,
    coordinates_selection_method="outside",
    output_filename=OUT_FILE.name,
    output_directory=str(OUT_DIR),
)

print(f"Saved: {OUT_FILE}")
