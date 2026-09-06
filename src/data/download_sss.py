from pathlib import Path
import argparse
import copernicusmarine


# ---------------------------------------------------------
# OceanEmbed - SSS downloader
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Download SSS data for OceanEmbed"
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

DATASET_ID = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"

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

print("Downloading SSS...")
print(f"Date: {args.start}")
print(
    f"Region: "
    f"{args.lon_min}–{args.lon_max}°E, "
    f"{args.lat_min}–{args.lat_max}°N"
)

copernicusmarine.subset(
    dataset_id=DATASET_ID,
    variables=["sos"],

    minimum_longitude=args.lon_min,
    maximum_longitude=args.lon_max,

    minimum_latitude=args.lat_min,
    maximum_latitude=args.lat_max,

    start_datetime=f"{args.start}T00:00:00",
    end_datetime=f"{args.end}T00:00:00",

    coordinates_selection_method="outside",

    output_filename=OUT_FILE.name,
    output_directory=str(OUT_FILE.parent),
)

print()
print("========================================")
print("SSS DOWNLOAD COMPLETE")
print("========================================")
print(f"Saved: {OUT_FILE}")