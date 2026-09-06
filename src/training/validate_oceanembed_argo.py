import glob
import os

import numpy as np
import pandas as pd
import xarray as xr


PREDICTION_FILE = (
    "data/processed/oceanembed_prediction_depthbaseline_4days.nc"
)

CORA_DIR = "data/raw/cora_test"

POINT_OUTPUT = (
    "data/processed/oceanembed_argo_standardized.csv"
)

DEPTH_OUTPUT = (
    "data/processed/oceanembed_argo_standardized_depth.csv"
)

PROFILE_OUTPUT = (
    "data/processed/oceanembed_argo_standardized_profiles.csv"
)


TARGET_DEPTHS = np.array([
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
], dtype=float)


def pressure_to_depth(pressure, latitude):
    """
    Convert pressure (dbar) to approximate geometric depth (m)
    using the UNESCO/Saunders pressure-depth relationship.
    """

    p = np.asarray(pressure, dtype=float)
    lat = np.asarray(latitude, dtype=float)

    sin2 = np.sin(
        np.deg2rad(lat)
    ) ** 2

    gravity = (
        9.780318
        * (
            1
            + (
                5.2788e-3
                + 2.36e-5 * sin2
            )
            * sin2
        )
        + 1.092e-6 * p
    )

    depth = (
        (
            (
                -1.82e-15 * p
                + 2.279e-10
            ) * p
            - 2.2512e-5
        ) * p
        + 9.72659
    ) * p / gravity

    return np.abs(depth)


def correlation(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 3:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def get_prediction(
    prediction,
    date,
    latitude,
    longitude,
    depths
):

    pred = prediction.sel(
        time=np.datetime64(date),
        method="nearest"
    )

    pred = pred.sel(
        latitude=float(latitude),
        longitude=float(longitude),
        method="nearest"
    )

    model_depth = np.asarray(
        pred.depth.values,
        dtype=float
    )

    model_temp = np.asarray(
        pred.temperature_prediction.values,
        dtype=float
    )

    valid = (
        np.isfinite(model_depth)
        & np.isfinite(model_temp)
    )

    model_depth = model_depth[valid]
    model_temp = model_temp[valid]

    order = np.argsort(model_depth)

    model_depth = model_depth[order]
    model_temp = model_temp[order]

    result = np.full(
        len(depths),
        np.nan
    )

    inside = (
        (depths >= model_depth.min())
        & (depths <= model_depth.max())
    )

    result[inside] = np.interp(
        depths[inside],
        model_depth,
        model_temp
    )

    return result


print("=" * 70)
print("OCEANEMBED — STANDARDIZED-DEPTH CORA/ARGO VALIDATION")
print("=" * 70)


prediction = xr.open_dataset(
    PREDICTION_FILE
)

print(
    "\nPrediction grid:",
    dict(prediction.sizes)
)


files = sorted(
    f
    for f in glob.glob(
        os.path.join(
            CORA_DIR,
            "**",
            "*.nc"
        ),
        recursive=True
    )
    if "\\global\\" in f
    and "_TS_PF.nc" in f
)

print(
    "CORA global profile files:",
    len(files)
)


records = []
profile_results = []


for filename in files:

    print(
        "\nProcessing:",
        os.path.basename(filename)
    )

    ds = xr.open_dataset(filename)

    latitudes = np.asarray(
        ds.LATITUDE.values,
        dtype=float
    )

    longitudes = np.asarray(
        ds.LONGITUDE.values,
        dtype=float
    )

    for i in range(len(latitudes)):

        lat = latitudes[i]
        lon = longitudes[i]

        if not (
            5 <= lat <= 20
            and 80 <= lon <= 90
        ):
            continue

        date = pd.to_datetime(
            ds.TIME.values[i]
        ).strftime("%Y-%m-%d")

        if date not in {
            "2020-01-27",
            "2020-01-28",
            "2020-01-29",
            "2020-01-30"
        }:
            continue


        # ----------------------------------------------------
        # Adjusted pressure and temperature
        # ----------------------------------------------------

        pressure = np.asarray(
            ds.PRES_ADJUSTED.values[i],
            dtype=float
        )

        temperature = np.asarray(
            ds.TEMP_ADJUSTED.values[i],
            dtype=float
        )

        qc = np.asarray(
            ds.TEMP_ADJUSTED_QC.values[i],
            dtype=float
        )


        # ----------------------------------------------------
        # Valid QC: 1 = good, 2 = probably good
        # ----------------------------------------------------

        valid = (
            np.isfinite(pressure)
            & np.isfinite(temperature)
            & np.isfinite(qc)
            & np.isin(qc, [1.0, 2.0])
        )

        pressure = pressure[valid]
        temperature = temperature[valid]


        if len(pressure) < 5:
            continue


        # ----------------------------------------------------
        # Pressure → actual depth
        # ----------------------------------------------------

        depth = pressure_to_depth(
            pressure,
            lat
        )


        valid = (
            np.isfinite(depth)
            & (depth >= 0)
            & (depth <= 1000)
        )

        depth = depth[valid]
        temperature = temperature[valid]


        if len(depth) < 5:
            continue


        # ----------------------------------------------------
        # Sort profile
        # ----------------------------------------------------

        order = np.argsort(depth)

        depth = depth[order]
        temperature = temperature[order]


        # Remove duplicate depths
        unique_depth, unique_idx = np.unique(
            depth,
            return_index=True
        )

        depth = unique_depth
        temperature = temperature[unique_idx]


        # ----------------------------------------------------
        # Interpolate ARGO to exact SIH depths
        # ----------------------------------------------------

        argo_standard = np.full(
            len(TARGET_DEPTHS),
            np.nan
        )

        inside = (
            (TARGET_DEPTHS >= depth.min())
            & (TARGET_DEPTHS <= depth.max())
        )

        argo_standard[inside] = np.interp(
            TARGET_DEPTHS[inside],
            depth,
            temperature
        )


        # ----------------------------------------------------
        # OceanEmbed at same depths
        # ----------------------------------------------------

        oceanembed_standard = get_prediction(
            prediction,
            date,
            lat,
            lon,
            TARGET_DEPTHS
        )


        # ----------------------------------------------------
        # Save profile/depth observations
        # ----------------------------------------------------

        platform = str(
            ds.PLATFORM_NUMBER.values[i]
        ).strip()

        profile_errors = []

        for z, argo_temp, model_temp in zip(
            TARGET_DEPTHS,
            argo_standard,
            oceanembed_standard
        ):

            if not (
                np.isfinite(argo_temp)
                and np.isfinite(model_temp)
            ):
                continue

            error = (
                model_temp
                - argo_temp
            )

            records.append({
                "date": date,
                "platform": platform,
                "latitude": lat,
                "longitude": lon,
                "depth_m": z,
                "argo_temperature": argo_temp,
                "oceanembed_temperature": model_temp,
                "error": error
            })

            profile_errors.append(error)


        if profile_errors:

            profile_errors = np.asarray(
                profile_errors
            )

            profile_results.append({
                "date": date,
                "platform": platform,
                "latitude": lat,
                "longitude": lon,
                "n_depths": len(profile_errors),
                "rmse_c": np.sqrt(
                    np.mean(
                        profile_errors ** 2
                    )
                ),
                "bias_c": np.mean(
                    profile_errors
                )
            })


    ds.close()


prediction.close()


if not records:

    raise RuntimeError(
        "No standardized-depth ARGO observations survived."
    )


df = pd.DataFrame(records)

profiles = pd.DataFrame(
    profile_results
)


os.makedirs(
    "data/processed",
    exist_ok=True
)


df.to_csv(
    POINT_OUTPUT,
    index=False
)

profiles.to_csv(
    PROFILE_OUTPUT,
    index=False
)


# ============================================================
# Overall statistics
# ============================================================

obs = df.argo_temperature.values
pred = df.oceanembed_temperature.values
err = pred - obs


overall_rmse = np.sqrt(
    np.mean(err ** 2)
)

overall_bias = np.mean(err)

overall_corr = correlation(
    obs,
    pred
)


print("\n" + "=" * 70)
print("OVERALL STANDARDIZED-DEPTH VALIDATION")
print("=" * 70)

print(
    f"Profiles     : {profiles.platform.nunique()}"
)

print(
    f"Comparisons  : {len(df)}"
)

print(
    f"RMSE         : {overall_rmse:.4f} °C"
)

print(
    f"Bias         : {overall_bias:.4f} °C"
)

print(
    f"Correlation  : {overall_corr:.4f}"
)


# ============================================================
# Depth-wise statistics
# ============================================================

results = []


for z in TARGET_DEPTHS:

    subset = df[
        df.depth_m == z
    ]

    n = len(subset)

    profile_count = (
        subset.platform.nunique()
    )

    if n < 3:

        results.append({
            "depth_m": z,
            "n": n,
            "profiles": profile_count,
            "rmse_c": np.nan,
            "bias_c": np.nan,
            "correlation": np.nan
        })

        continue


    obs = subset.argo_temperature.values
    pred = subset.oceanembed_temperature.values

    err = pred - obs


    results.append({
        "depth_m": z,
        "n": n,
        "profiles": profile_count,
        "rmse_c": np.sqrt(
            np.mean(err ** 2)
        ),
        "bias_c": np.mean(err),
        "correlation": correlation(
            obs,
            pred
        )
    })


depth_results = pd.DataFrame(
    results
)


depth_results.to_csv(
    DEPTH_OUTPUT,
    index=False
)


print("\n" + "=" * 70)
print("DEPTH-WISE STANDARDIZED-DEPTH VALIDATION")
print("=" * 70)

print(
    depth_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("PROFILE SUMMARY")
print("=" * 70)

print(
    profiles.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    POINT_OUTPUT
)

print(
    DEPTH_OUTPUT
)

print(
    PROFILE_OUTPUT
)

print("=" * 70)