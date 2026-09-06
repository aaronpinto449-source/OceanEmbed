import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr


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
    "oceanembed_prediction_depthbaseline_4days.nc"
)
INPUT_FILE = (
    "data/processed/"
    "oceanembed_dashboard_inputs_4days.nc"
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
def load_prediction():

    return xr.open_dataset(
        PREDICTION_FILE
    )
@st.cache_data
def load_inputs():

    return xr.open_dataset(
        INPUT_FILE
    )

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


prediction = load_prediction()
inputs = load_inputs()
argo = load_argo()
argo_depth = load_argo_depth()
argo_profiles = load_argo_profiles()


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
# SELECT DATA
# ============================================================

day = prediction.sel(
    time=np.datetime64(selected_date)
)

field = day.sel(
    depth=selected_depth
)


pred_map = field[
    "temperature_prediction"
].values

ref_map = field[
    "temperature_reference"
].values

error_map = (
    pred_map
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
        "Depth",
        f"{int(selected_depth)} m"
    )

with m2:
    st.metric(
        "RMSE",
        f"{rmse:.3f} °C"
    )

with m3:
    st.metric(
        "Bias",
        f"{bias:+.3f} °C"
    )

with m4:
    st.metric(
        "Correlation",
        f"{corr:.3f}"
        if np.isfinite(corr)
        else "N/A"
    )


# ============================================================
# TEMPERATURE MAPS
# ============================================================

st.markdown("---")

st.header(
    "Subsurface Temperature Reconstruction"
)

st.caption(
    f"{selected_date.strftime('%Y-%m-%d')} • "
    f"{int(selected_depth)} m depth"
)


map1, map2, map3 = st.columns(3)


def make_map(
    values,
    title,
    colorscale
):

    fig = go.Figure(
        go.Heatmap(
            x=longitudes,
            y=latitudes,
            z=values,
            colorscale=colorscale,
            colorbar=dict(
                title="°C"
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E"
                "<br>"
                "Lat: %{y:.2f}°N"
                "<br>"
                "Temperature: %{z:.2f}°C"
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


with map1:

    st.plotly_chart(
        make_map(
            pred_map,
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
            "RdBu_r"
        ),
        width="stretch"
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
# 7 INPUT VARIABLES
# ============================================================

st.markdown("---")

st.header(
    "Surface Observation Inputs"
)

st.caption(
    "The seven surface fields supplied to the OceanEmbed "
    "reconstruction model"
)


# ------------------------------------------------------------
# Select the same date as the reconstruction explorer
# ------------------------------------------------------------

input_day = inputs.sel(
    time=np.datetime64(selected_date)
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


def make_input_map(
    values,
    title,
    units,
    colorscale,
    symmetric=False
):

    values = np.asarray(
        values,
        dtype=float
    )

    finite = np.isfinite(values)

    if not np.any(finite):

        zmin = None
        zmax = None

    elif symmetric:

        limit = float(
            np.nanmax(
                np.abs(values[finite])
            )
        )

        zmin = -limit
        zmax = limit

    else:

        zmin = float(
            np.nanpercentile(
                values[finite],
                2
            )
        )

        zmax = float(
            np.nanpercentile(
                values[finite],
                98
            )
        )

    fig = go.Figure(
        go.Heatmap(
            x=inputs.longitude.values,
            y=inputs.latitude.values,
            z=values,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(
                title=units
            ),
            hovertemplate=(
                "Longitude: %{x:.2f}°E"
                "<br>"
                "Latitude: %{y:.2f}°N"
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


# ------------------------------------------------------------
# First four inputs
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
# Remaining three inputs
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