import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
import torch

from src.models.oceanembed_cnn import OceanEmbedCNN


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OceanEmbed",
    page_icon="🌊",
    layout="wide",
)


# ============================================================
# FILES
# ============================================================

PREDICTION_FILE = (
    "data/processed/"
    "oceanembed_prediction_depthbaseline_30days.nc"
)
MODEL_FILE = (
    "models/oceanembed_depthbaseline_30day.pt"
)
INPUT_FILE = (
    "data/processed/"
    "oceanembed_dashboard_inputs_30days.nc"
)

ARGO_FILE = (
    "data/processed/"
    "oceanembed_argo_standardized.csv"
)

ARGO_DEPTH_FILE = (
    "data/processed/"
    "oceanembed_argo_standardized_depth.csv"
)

ARGO_PROFILE_FILE = (
    "data/processed/"
    "oceanembed_argo_standardized_profiles.csv"
)

ARGO_FIGURE = (
    "data/processed/"
    "oceanembed_argo_validation_final.png"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_prediction_30day():

    return xr.open_dataset(
        PREDICTION_FILE
    )


@st.cache_data
def load_inputs():

    return xr.open_dataset(
        INPUT_FILE
    )


@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    model = OceanEmbedCNN(
        in_channels=7,
        out_channels=len(
            checkpoint["target_depths"]
        )
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint
@st.cache_data
def load_argo():

    return pd.read_csv(
        ARGO_FILE
    )


@st.cache_data
def load_argo_depth():

    return pd.read_csv(
        ARGO_DEPTH_FILE
    )


@st.cache_data
def load_argo_profiles():

    return pd.read_csv(
        ARGO_PROFILE_FILE
    )


prediction = load_prediction_30day()
inputs = load_inputs()

model, checkpoint = load_model()
argo = load_argo()
argo_depth = load_argo_depth()
argo_profiles = load_argo_profiles()

def run_oceanembed_inference(
    input_dataset,
    selected_date,
    model,
    checkpoint
):

    variables = checkpoint[
        "input_variables"
    ]

    input_means = np.asarray(
        checkpoint["input_means"],
        dtype=np.float32
    )

    input_stds = np.asarray(
        checkpoint["input_stds"],
        dtype=np.float32
    )

    target_means = np.asarray(
        checkpoint["target_means"],
        dtype=np.float32
    )

    target_stds = np.asarray(
        checkpoint["target_stds"],
        dtype=np.float32
    )

    target_depths = np.asarray(
        checkpoint["target_depths"],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Extract the seven surface channels
    # --------------------------------------------------------

    day = input_dataset.sel(
        time=np.datetime64(selected_date), method="nearest"
    )

    X = np.stack(
        [
            day[var].values
            for var in variables
        ],
        axis=0
    ).astype(np.float32)


    # --------------------------------------------------------
    # Normalize exactly as during training
    # --------------------------------------------------------

    X_normalized = (
        X
        - input_means[
            :, None, None
        ]
    ) / input_stds[
        :, None, None
    ]


    # --------------------------------------------------------
    # Missing-value handling exactly as training
    # --------------------------------------------------------

    input_valid_mask = np.all(
        np.isfinite(X),
        axis=0
    )

    X_normalized = np.nan_to_num(
        X_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


    # --------------------------------------------------------
    # Build depth-dependent baseline
    #
    # Same method used during training:
    # climatological target mean at depth,
    # with observed SST at 0 m.
    # --------------------------------------------------------

    baseline = np.broadcast_to(
        target_means[
            :, None, None
        ],
        (
            len(target_depths),
            X.shape[1],
            X.shape[2]
        )
    ).copy()


    # 0 m = observed SST

    baseline[0] = X[0]


    baseline_normalized = (
        baseline
        - target_means[
            :, None, None
        ]
    ) / target_stds[
        :, None, None
    ]


    baseline_normalized = np.nan_to_num(
        baseline_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)


    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tensor = torch.from_numpy(
        X_normalized
    ).unsqueeze(0)

    baseline_tensor = torch.from_numpy(
        baseline_normalized
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Neural-network inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction_normalized = model(
            X_tensor,
            baseline_tensor
        )


    prediction_normalized = (
        prediction_normalized
        .squeeze(0)
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Convert back to °C
    # --------------------------------------------------------

    prediction_celsius = (
        prediction_normalized
        * target_stds[
            :, None, None
        ]
        + target_means[
            :, None, None
        ]
    )


    return (
        prediction_celsius,
        input_valid_mask
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌊 OceanEmbed"
)

st.subheader(
    "Satellite-Driven Reconstruction of Subsurface Ocean Temperature"
)

st.markdown(
    """
**North Indian Ocean • 0.25° daily grid • 15 depth levels**

OceanEmbed uses seven surface ocean-atmosphere observations to
reconstruct the three-dimensional subsurface temperature structure
from the surface to 1000 m.
"""
)


# ============================================================
# KEY RESULTS
# ============================================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Surface Inputs",
        "7"
    )

with c2:
    st.metric(
        "Depth Levels",
        "15"
    )

with c3:
    st.metric(
        "GLORYS Test RMSE",
        "0.44 °C"
    )

with c4:
    st.metric(
        "Independent CORA/ARGO RMSE",
        "0.73 °C"
    )


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header(
    "Reconstruction Explorer"
)

dates = pd.to_datetime(
    prediction.time.values
)

selected_date = st.sidebar.selectbox(
    "Date",
    dates,
    format_func=lambda x: x.strftime("%Y-%m-%d")
)


depths = prediction.depth.values

selected_depth = st.sidebar.selectbox(
    "Depth",
    depths,
    format_func=lambda x: f"{int(x)} m"
)


latitudes = prediction.latitude.values
longitudes = prediction.longitude.values


selected_lat = st.sidebar.slider(
    "Latitude",
    float(latitudes.min()),
    float(latitudes.max()),
    float(latitudes.mean()),
    0.25
)


selected_lon = st.sidebar.slider(
    "Longitude",
    float(longitudes.min()),
    float(longitudes.max()),
    float(longitudes.mean()),
    0.25
)

# ============================================================
# LIVE MODEL INFERENCE
# ============================================================

with st.spinner(
    "Running OceanEmbed neural-network reconstruction..."
):

    live_prediction, input_valid_mask = (
        run_oceanembed_inference(
            inputs,
            selected_date,
            model,
            checkpoint
        )
    )
st.success(
    "✓ OceanEmbed CNN inference executed using "
    "the selected seven surface input fields."
)
# ============================================================
# SELECT DATA
# ============================================================

day = prediction.sel(
    time=np.datetime64(selected_date), method="nearest"
)

field = day.sel(
    depth=selected_depth
)


live_pred_map = live_prediction[
    np.argmin(
        np.abs(
            prediction.depth.values
            - selected_depth
        )
    )
]

ref_map = field[
    "temperature_reference"
].values

pred_map = live_pred_map

error_map = (
    live_pred_map
    - ref_map
)


# ============================================================
# POINT METRICS
# ============================================================

point = day.sel(
    latitude=selected_lat,
    longitude=selected_lon,
    method="nearest"
)

profile_pred = point[
    "temperature_prediction"
].values

profile_ref = point[
    "temperature_reference"
].values

profile_error = (
    profile_pred
    - profile_ref
)


point_idx = np.argmin(
    np.abs(
        latitudes
        - selected_lat
    )
)

lon_idx = np.argmin(
    np.abs(
        longitudes
        - selected_lon
    )
)


# ============================================================
# MAP METRICS
# ============================================================

valid = (
    np.isfinite(pred_map)
    & np.isfinite(ref_map)
)

if np.any(valid):

    rmse = float(
        np.sqrt(
            np.mean(
                error_map[valid] ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(error_map[valid])
        )
    )

    valid_pixels = int(np.sum(valid))
    total_pixels = int(valid.size)
    coverage = (
        100.0 * valid_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    bias = float(
        np.mean(
            error_map[valid]
        )
    )

    if (
        np.std(pred_map[valid]) > 0
        and np.std(ref_map[valid]) > 0
    ):

        corr = float(
            np.corrcoef(
                pred_map[valid],
                ref_map[valid]
            )[0, 1]
        )

    else:

        corr = np.nan

else:

    rmse = np.nan
    bias = np.nan
    corr = np.nan


m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "RMSE",
        f"{rmse:.3f} C"
    )

with m2:
    st.metric(
        "MAE",
        f"{mae:.3f} C"
    )

with m3:
    st.metric(
        "Bias",
        f"{bias:+.3f} C"
    )

with m4:
    st.metric(
        "Correlation",
        f"{corr:.3f}"
        if np.isfinite(corr)
        else "N/A"
    )

st.caption(
    f"Validation coverage: {valid_pixels:,} / {total_pixels:,} "
    f"valid pixels ({coverage:.1f}%) at {int(selected_depth)} m depth."
)


# ============================================================
# MAP HELPERS
# ============================================================
def make_map(values, title, colorscale, symmetric=False):
    values = np.asarray(values, dtype=float)

    if symmetric:
        limit = 1.0

        fig = go.Figure(
            go.Heatmap(
                x=longitudes,
                y=latitudes,
                z=values,
                colorscale=colorscale,
                zmin=-limit,
                zmax=limit,
                colorbar=dict(title="°C"),
                hovertemplate=(
                    "Lon: %{x:.2f}°E"
                    "<br>"
                    "Lat: %{y:.2f}°N"
                    "<br>"
                    f"{title}: %{{z:.3f}} °C"
                    "<extra></extra>"
                )
            )
        )

    else:
        fig = go.Figure(
            go.Heatmap(
                x=longitudes,
                y=latitudes,
                z=values,
                colorscale=colorscale,
                colorbar=dict(title="°C"),
                hovertemplate=(
                    "Lon: %{x:.2f}°E"
                    "<br>"
                    "Lat: %{y:.2f}°N"
                    "<br>"
                    f"{title}: %{{z:.3f}} °C"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=430,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return fig


def make_input_map(values, title, units, colorscale, symmetric=False):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if not np.any(finite):
        zmin = None
        zmax = None
    elif symmetric:
        limit = float(np.nanmax(np.abs(values[finite])))
        limit = max(limit, 1.0)
        zmin = -limit
        zmax = limit
    else:
        zmin = float(np.nanpercentile(values[finite], 2))
        zmax = float(np.nanpercentile(values[finite], 98))

    fig = go.Figure(
        go.Heatmap(
            x=longitudes,
            y=latitudes,
            z=values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=units),
            hovertemplate=(
                "Longitude: %{x:.2f}"
                "<br>"
                "Latitude: %{y:.2f}"
                "<br>"
                f"{title}: %{{z:.3f}} {units}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=330,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10
        )
    )

    return fig


# ============================================================
# SURFACE OBSERVATION INPUTS
# ============================================================

st.markdown("---")

st.header(
    "Surface Observation Inputs"
)

st.caption(
    f"Showing the seven surface observation fields for "
    f"{selected_date.strftime('%Y-%m-%d')}"
)


# ------------------------------------------------------------
# Select the exact input day used by the CNN
# ------------------------------------------------------------

input_day = inputs.sel(
    time=np.datetime64(selected_date), method="nearest"
)


input_definitions = [
    (
        "sst",
        "Sea Surface Temperature",
        "°C",
        "Turbo"
    ),
    (
        "sss",
        "Sea Surface Salinity",
        "PSU",
        "Viridis"
    ),
    (
        "ssh",
        "Sea Surface Height",
        "m",
        "RdBu_r"
    ),
    (
        "u_current",
        "Zonal Surface Current",
        "m/s",
        "RdBu_r"
    ),
    (
        "v_current",
        "Meridional Surface Current",
        "m/s",
        "RdBu_r"
    ),
    (
        "u_wind",
        "Zonal Surface Wind",
        "m/s",
        "RdBu_r"
    ),
    (
        "v_wind",
        "Meridional Surface Wind",
        "m/s",
        "RdBu_r"
    ),
]


# ------------------------------------------------------------
# First four input maps
# ------------------------------------------------------------

row1 = st.columns(4)

for column, definition in zip(
    row1,
    input_definitions[:4]
):

    variable, title, units, colorscale = definition

    with column:

        st.plotly_chart(
            make_input_map(
                input_day[variable].values,
                title,
                units,
                colorscale,
                symmetric=variable in {
                    "ssh",
                    "u_current",
                    "v_current",
                    "u_wind",
                    "v_wind"
                }
            ),
            width="stretch"
        )


# ------------------------------------------------------------
# Remaining three input maps
# ------------------------------------------------------------

row2 = st.columns(4)

for column, definition in zip(
    row2[:3],
    input_definitions[4:]
):

    variable, title, units, colorscale = definition

    with column:

        st.plotly_chart(
            make_input_map(
                input_day[variable].values,
                title,
                units,
                colorscale,
                symmetric=True
            ),
            width="stretch"
        )


st.info(
    """
These seven surface fields form the model input tensor:

**[SST, SSS, SSH, U-current, V-current, U-wind, V-wind]**

They are harmonized to the same 0.25° spatial grid and daily
temporal resolution before being passed to the reconstruction model.
"""
)


# ============================================================
# OCEANEMBED OUTPUT MAPS
# ============================================================

st.markdown("---")

st.header(
    "OceanEmbed Reconstruction"
)

st.caption(
    f"{selected_date.strftime('%Y-%m-%d')} • "
    f"{int(selected_depth)} m depth"
)


map1, map2, map3 = st.columns(3)


with map1:

    st.plotly_chart(
        make_map(
            live_pred_map,
            "OceanEmbed Prediction",
            "Turbo"
        ),
        width="stretch"
    )


with map2:

    st.plotly_chart(
        make_map(
            ref_map,
            "GLORYS Reference",
            "Turbo"
        ),
        width="stretch"
    )


with map3:

    st.plotly_chart(
        make_map(
    error_map,
    "Reconstruction Error",
    "RdBu_r",
    symmetric=True
),
        width="stretch"
    )
# ============================================================
# 3D OCEAN RECONSTRUCTION
# ============================================================

st.markdown("---")

st.header(
    "3D Ocean Reconstruction"
)

st.caption(
    f"Interactive temperature volume • "
    f"{selected_date.strftime('%Y-%m-%d')}"
)

view_mode = st.radio(
    "3D View",
    [
        "OceanEmbed Prediction",
        "GLORYS Reference",
        "Reconstruction Error"
    ],
    horizontal=True,
    key="3d_view_mode"
)

max_depth_3d = st.select_slider(
    "Maximum Depth",
    options=[int(d) for d in depths],
    value=int(depths[-1]),
    key="3d_max_depth"
)

depth_mask = depths <= max_depth_3d

depth_3d = depths[depth_mask]

prediction_3d = live_prediction[
    depth_mask
]

reference_3d = day[
    "temperature_reference"
].values[
    depth_mask
]

if view_mode == "OceanEmbed Prediction":

    volume_3d = prediction_3d
    colorscale_3d = "Turbo"
    title_3d = "OceanEmbed Temperature Reconstruction"
    colorbar_title = "Temperature (°C)"

elif view_mode == "GLORYS Reference":

    volume_3d = reference_3d
    colorscale_3d = "Turbo"
    title_3d = "GLORYS Reference Temperature"
    colorbar_title = "Temperature (°C)"

else:

    volume_3d = (
        prediction_3d
        - reference_3d
    )

    colorscale_3d = "RdBu_r"
    title_3d = "OceanEmbed Reconstruction Error"
    colorbar_title = "Error (°C)"


depth_grid, lat_grid, lon_grid = np.meshgrid(
    depth_3d,
    latitudes,
    longitudes,
    indexing="ij"
)

finite_3d = np.isfinite(volume_3d)

if np.any(finite_3d):

    if view_mode == "Reconstruction Error":

        error_limit = float(
            np.nanpercentile(
                np.abs(volume_3d[finite_3d]),
                98
            )
        )

        error_limit = max(
            error_limit,
            0.05
        )

        zmin_3d = -error_limit
        zmax_3d = error_limit

    else:

        zmin_3d = float(
            np.nanpercentile(
                volume_3d[finite_3d],
                2
            )
        )

        zmax_3d = float(
            np.nanpercentile(
                volume_3d[finite_3d],
                98
            )
        )

else:

    zmin_3d = None
    zmax_3d = None


fig_3d = go.Figure(
    data=go.Isosurface(
        x=lon_grid.flatten(),
        y=lat_grid.flatten(),
        z=depth_grid.flatten(),
        value=volume_3d.flatten(),
        isomin=zmin_3d,
        isomax=zmax_3d,
        surface_count=8,
        colorscale=colorscale_3d,
        cmin=zmin_3d,
        cmax=zmax_3d,
        colorbar=dict(
            title=colorbar_title
        ),
        caps=dict(
            x_show=False,
            y_show=False,
            z_show=False
        ),
        hovertemplate=(
            "Longitude: %{x:.2f}°"
            "<br>Latitude: %{y:.2f}°"
            "<br>Depth: %{z:.0f} m"
            "<br>Value: %{value:.3f}"
            "<extra></extra>"
        )
    )
)

fig_3d.update_layout(
    title=title_3d,
    height=700,
    margin=dict(
        l=0,
        r=0,
        t=50,
        b=0
    ),
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis_title="Depth (m)",
        zaxis=dict(
            autorange="reversed"
        )
    )
)

st.plotly_chart(
    fig_3d,
    width="stretch",
    key=f"ocean_3d_{selected_date.strftime('%Y%m%d')}_{max_depth_3d}_{view_mode}"
)

# ============================================================
# VERTICAL PROFILE
# ============================================================

st.markdown("---")

st.header(
    "Vertical Temperature Profile"
)

st.caption(
    f"Nearest grid point: "
    f"{latitudes[point_idx]:.2f}°N, "
    f"{longitudes[lon_idx]:.2f}°E"
)


profile_fig = go.Figure()


profile_fig.add_trace(
    go.Scatter(
        x=profile_ref,
        y=depths,
        mode="lines+markers",
        name="GLORYS Reference"
    )
)


profile_fig.add_trace(
    go.Scatter(
        x=profile_pred,
        y=depths,
        mode="lines+markers",
        name="OceanEmbed"
    )
)


profile_fig.update_yaxes(
    autorange="reversed",
    title="Depth (m)"
)

profile_fig.update_xaxes(
    title="Temperature (°C)"
)

profile_fig.update_layout(
    height=600,
    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    )
)


st.plotly_chart(
    profile_fig,
    width="stretch"
)


# ============================================================
# INDEPENDENT ARGO VALIDATION
# ============================================================

st.markdown("---")

st.header(
    "Independent CORA/ARGO Validation"
)

st.markdown(
    """
The model is independently evaluated against CORA profiling-float
temperature observations rather than only against the GLORYS
reanalysis used as the training target.
"""
)


v1, v2, v3, v4 = st.columns(4)

with v1:
    st.metric(
        "Profiles",
        "9"
    )

with v2:
    st.metric(
        "Depth Comparisons",
        "114"
    )

with v3:
    st.metric(
        "RMSE",
        "0.73 °C"
    )

with v4:
    st.metric(
        "Pearson r",
        "0.995"
    )


# ============================================================
# ARGO VALIDATION FIGURE
# ============================================================

if os.path.exists(
    ARGO_FIGURE
):

    st.image(
        ARGO_FIGURE,
        caption=(
            "Independent CORA/ARGO validation "
            "at standardized OceanEmbed depth levels."
        ),
        width="stretch"
    )


# ============================================================
# DEPTH VALIDATION TABLE
# ============================================================

st.subheader(
    "Validation by Reconstruction Depth"
)


display_depth = argo_depth.copy()

display_depth.columns = [
    "Depth (m)",
    "N",
    "Profiles",
    "RMSE (°C)",
    "Bias (°C)",
    "Correlation"
]


display_depth = display_depth.round(
    {
        "RMSE (°C)": 3,
        "Bias (°C)": 3,
        "Correlation": 3
    }
)


st.dataframe(
    display_depth,
    width="stretch",
    hide_index=True
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

st.markdown("---")

st.header(
    "OceanEmbed Architecture"
)

st.markdown(
    """
**Seven surface channels**

SST • SSS • SSH • U/V surface currents • U/V surface winds

↓

**Residual CNN embedding / reconstruction network**

↓

**15 depth-wise temperature fields**

0 • 5 • 10 • 20 • 30 • 50 • 75 • 100 • 125 •
150 • 200 • 300 • 500 • 700 • 1000 m
"""
)


# ============================================================
# INTERPRETATION
# ============================================================

st.markdown("---")

st.header(
    "Model Interpretation"
)

st.info(
    """
OceanEmbed performs strongest in the upper mixed layer and deeper
ocean where the temperature structure is smoother. The main
reconstruction challenge in this experiment occurs around
75–125 m, where the independent CORA/ARGO validation shows higher
errors. This highlights the physical difficulty of inferring
rapidly varying subsurface structure from surface observations alone.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "OceanEmbed • SIH 2026 Proof of Concept • "
    "North Indian Ocean"
)
