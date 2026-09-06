from pathlib import Path
import argparse
import copernicusmarine


# ---------------------------------------------------------
# OceanEmbed - Wind downloader
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Download daily wind data for OceanEmbed"
)

parser.add_argument("--lon-min", type=float, required=True)
parser.add_argument("--lon-max", type=float, required=True)
parser.add_argument("--lat-min", type=float, required=True)
parser.add_argument("--lat-max", type=float, required=True)

parser.add_argument("--start", required=True)
parser.add_argument("--end", required=True)

parser.add_argument("--output", required=True)

args = parser.parse_args()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET_ID = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H"

OUT_FILE = Path(args.output)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Check existing file
# ---------------------------------------------------------

if OUT_FILE.exists():
    print(f"Already exists: {OUT_FILE}")
    raise SystemExit


# ---------------------------------------------------------
# Download
# ---------------------------------------------------------

print("Downloading wind...")
print(f"Date: {args.start}")
print(
    f"Region: "
    f"{args.lon_min}–{args.lon_max}°E, "
    f"{args.lat_min}–{args.lat_max}°N"
)

copernicusmarine.subset(
    dataset_id=DATASET_ID,

    variables=[
        "eastward_wind",
        "northward_wind",
    ],

    minimum_longitude=args.lon_min,
    maximum_longitude=args.lon_max,

    minimum_latitude=args.lat_min,
    maximum_latitude=args.lat_max,

    start_datetime=f"{args.start}T00:00:00",
    end_datetime=f"{args.end}T23:00:00",

    coordinates_selection_method="outside",

    output_filename=OUT_FILE.name,
    output_directory=str(OUT_FILE.parent),
)


print()
print("========================================")
print("WIND DOWNLOAD COMPLETE")
print("========================================")
print(f"Saved: {OUT_FILE}")