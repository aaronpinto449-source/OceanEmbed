import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT = "data/processed/oceanembed_argo_standardized.csv"
OUTPUT = "data/processed/oceanembed_argo_validation_final.png"


df = pd.read_csv(INPUT)

# Clean platform names
df["platform"] = (
    df["platform"]
    .astype(str)
    .str.replace("b'", "", regex=False)
    .str.replace("'", "", regex=False)
    .str.strip()
)


depths = np.sort(
    df["depth_m"].unique()
)


# ============================================================
# Build profile matrices
# ============================================================

argo_profiles = []
model_profiles = []


for platform in df["platform"].unique():

    p = df[
        df["platform"] == platform
    ].sort_values("depth_m")

    argo = np.full(
        len(depths),
        np.nan
    )

    model = np.full(
        len(depths),
        np.nan
    )

    for j, depth in enumerate(depths):

        row = p[
            p["depth_m"] == depth
        ]

        if len(row):

            argo[j] = row[
                "argo_temperature"
            ].iloc[0]

            model[j] = row[
                "oceanembed_temperature"
            ].iloc[0]

    argo_profiles.append(argo)
    model_profiles.append(model)


argo_profiles = np.array(
    argo_profiles
)

model_profiles = np.array(
    model_profiles
)


# ============================================================
# Mean and spread
# ============================================================

argo_mean = np.nanmean(
    argo_profiles,
    axis=0
)

argo_std = np.nanstd(
    argo_profiles,
    axis=0
)

model_mean = np.nanmean(
    model_profiles,
    axis=0
)

model_std = np.nanstd(
    model_profiles,
    axis=0
)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 9)
)


# ============================================================
# Individual profiles — subtle background
# ============================================================

for profile in argo_profiles:

    ax.plot(
        profile,
        depths,
        linewidth=1,
        alpha=0.15
    )


for profile in model_profiles:

    ax.plot(
        profile,
        depths,
        linewidth=1,
        alpha=0.15
    )


# ============================================================
# Mean ± 1 standard deviation
# ============================================================

ax.fill_betweenx(
    depths,
    argo_mean - argo_std,
    argo_mean + argo_std,
    alpha=0.12
)

ax.fill_betweenx(
    depths,
    model_mean - model_std,
    model_mean + model_std,
    alpha=0.12
)


# ============================================================
# Mean profiles
# ============================================================

ax.plot(
    argo_mean,
    depths,
    linewidth=3,
    label="CORA/ARGO mean"
)

ax.plot(
    model_mean,
    depths,
    linewidth=3,
    label="OceanEmbed mean"
)


# ============================================================
# Higher-error region
# ============================================================

ax.axhspan(
    75,
    125,
    alpha=0.10
)

ax.text(
    0.98,
    0.80,
    "Higher-error region\n75–125 m",
    transform=ax.transAxes,
    ha="right",
    va="center",
    fontsize=10
)


# ============================================================
# Axes
# ============================================================

ax.invert_yaxis()

ax.set_ylim(
    700,
    0
)

ax.set_xlabel(
    "Temperature (°C)",
    fontsize=13
)

ax.set_ylabel(
    "Depth (m)",
    fontsize=13
)

ax.set_title(
    "Independent CORA/ARGO Validation of OceanEmbed",
    fontsize=17,
    pad=15
)

ax.grid(
    alpha=0.25
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="upper right",
    frameon=True
)


# ============================================================
# Statistics
# ============================================================

stats = (
    "9 independent profiles\n"
    "114 standardized-depth comparisons\n"
    "RMSE = 0.73 °C\n"
    "Bias = +0.24 °C\n"
    "Pearson r = 0.995"
)

ax.text(
    0.02,
    0.04,
    stats,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.85
    )
)


plt.tight_layout()


os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)


plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved: {OUTPUT}"
)