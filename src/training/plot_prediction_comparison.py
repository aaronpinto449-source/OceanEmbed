import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# OceanEmbed — Prediction vs Reference Diagnostic
# ---------------------------------------------------------

REFERENCE_FILE = Path(
    "data/processed/oceanembed_prediction_test_2020-01-30.nc"
)

OUTPUT_DIR = Path(
    "data/processed/diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("Loading prediction dataset...")

ds = xr.open_dataset(
    REFERENCE_FILE
)

reference = ds["reference_temperature"]
prediction = ds["predicted_temperature"]


print(ds)
print()


# ---------------------------------------------------------
# Depths to inspect
# ---------------------------------------------------------

depths_to_plot = [
    0,
    50,
    100,
    200,
    500,
    1000,
]


# ---------------------------------------------------------
# Create comparison plots
# ---------------------------------------------------------

for depth in depths_to_plot:

    print(f"Plotting {depth} m...")

    actual = reference.sel(
        depth=depth
    )

    predicted = prediction.sel(
        depth=depth
    )

    difference = predicted - actual

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )

    # -----------------------------------------------------
    # Reference
    # -----------------------------------------------------

    actual.plot(
        ax=axes[0],
        cmap="turbo"
    )

    axes[0].set_title(
        f"GLORYS Reference — {depth} m"
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predicted.plot(
        ax=axes[1],
        cmap="turbo"
    )

    axes[1].set_title(
        f"OceanEmbed Prediction — {depth} m"
    )

    # -----------------------------------------------------
    # Difference
    # -----------------------------------------------------

    difference.plot(
        ax=axes[2],
        cmap="RdBu_r"
    )

    axes[2].set_title(
        f"Prediction Error — {depth} m"
    )

    fig.suptitle(
        f"OceanEmbed Diagnostic — 2020-01-30 — {depth} m",
        fontsize=14
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"comparison_{depth:04d}m.png"
    )

    plt.savefig(
        output_file,
        dpi=150
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


print()
print("=" * 60)
print("DIAGNOSTIC PLOTS CREATED")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}")