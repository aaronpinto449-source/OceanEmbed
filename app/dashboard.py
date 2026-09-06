import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="OceanEmbed",
    page_icon="🌊",
    layout="wide"
)


# =========================================================
# FILES
# =========================================================

REFERENCE_PATH = Path(
    "data/processed/glorys_bob_2020-01-15_025deg.nc"
)

PREDICTION_PATH = Path(
    "data/processed/oceanembed_prediction_depthbaseline_4days.nc"
)


# =========================================================
# HEADER
# =========================================================

st.title("🌊 OceanEmbed")

st.markdown(
    """
    ### Satellite Embedding-Based Deep Learning Framework
    ### for Subsurface Ocean Temperature Reconstruction

    **SIH26066 • North Indian Ocean • Bay of Bengal Prototype**
    """
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

if not PREDICTION_PATH.exists():

    st.warning(
        "Model #3 prediction data is not available yet. "
        "Run the Model #3 evaluation first."
    )

    st.stop()


@st.cache_data
def load_prediction():

    return xr.open_dataset(PREDICTION_PATH)


@st.cache_data
def load_reference():

    return xr.open_dataset(REFERENCE_PATH)


pred_ds = load_prediction()
ref_ds = load_reference()


prediction = pred_ds["temperature_prediction"]
reference = pred_ds["temperature_reference"]


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🌊 Ocean Controls")

available_depths = prediction.depth.values.tolist()

depth = st.sidebar.selectbox(
    "Select reconstruction depth",
    available_depths,
    index=7
)


date_values = prediction.time.values

selected_date = st.sidebar.selectbox(
    "Select test date",
    date_values,
    format_func=lambda x: str(x)[:10]
)


st.sidebar.divider()

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=float(prediction.latitude.min()),
    max_value=float(prediction.latitude.max()),
    value=15.0,
    step=0.25
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=float(prediction.longitude.min()),
    max_value=float(prediction.longitude.max()),
    value=85.0,
    step=0.25
)


# =========================================================
# SELECT DATA
# =========================================================

pred_at_depth = prediction.sel(
    time=selected_date,
    depth=depth
)

ref_at_depth = reference.sel(
    time=selected_date,
    depth=depth
)

error_at_depth = pred_at_depth - ref_at_depth


# =========================================================
# METRICS
# =========================================================

import numpy as np

pred_values = pred_at_depth.values
ref_values = ref_at_depth.values

valid = np.isfinite(pred_values) & np.isfinite(ref_values)

if valid.sum() > 0:

    rmse = float(
        np.sqrt(
            np.mean(
                (pred_values[valid] - ref_values[valid]) ** 2
            )
        )
    )

    bias = float(
        np.mean(
            pred_values[valid] - ref_values[valid]
        )
    )

    if (
        np.std(pred_values[valid]) > 0
        and np.std(ref_values[valid]) > 0
    ):
        correlation = float(
            np.corrcoef(
                pred_values[valid],
                ref_values[valid]
            )[0, 1]
        )
    else:
        correlation = float("nan")

else:

    rmse = float("nan")
    bias = float("nan")
    correlation = float("nan")


# =========================================================
# TOP METRIC CARDS
# =========================================================

st.subheader(
    f"Model #3 Reconstruction — {depth:.0f} m"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "RMSE",
    f"{rmse:.3f} °C"
)

m2.metric(
    "Bias",
    f"{bias:+.3f} °C"
)

m3.metric(
    "Correlation",
    f"{correlation:.3f}"
)

m4.metric(
    "Test Date",
    str(selected_date)[:10]
)


# =========================================================
# MAPS
# =========================================================

st.subheader(
    f"Temperature Field at {depth:.0f} m"
)

map1, map2, map3 = st.columns(3)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

with map1:

    st.markdown("#### 🧠 OceanEmbed Prediction")

    fig_pred = px.imshow(
        pred_at_depth,
        x=prediction.longitude,
        y=prediction.latitude,
        origin="lower",
        aspect="auto",
        labels={
            "x": "Longitude",
            "y": "Latitude",
            "color": "Temperature (°C)"
        }
    )

    fig_pred.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig_pred,
        width="stretch"
    )


# ---------------------------------------------------------
# REFERENCE
# ---------------------------------------------------------

with map2:

    st.markdown("#### 🌐 GLORYS Reference")

    fig_ref = px.imshow(
        ref_at_depth,
        x=prediction.longitude,
        y=prediction.latitude,
        origin="lower",
        aspect="auto",
        labels={
            "x": "Longitude",
            "y": "Latitude",
            "color": "Temperature (°C)"
        }
    )

    fig_ref.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig_ref,
        width="stretch"
    )


# ---------------------------------------------------------
# ERROR
# ---------------------------------------------------------

with map3:

    st.markdown("#### 📊 Reconstruction Error")

    fig_error = px.imshow(
        error_at_depth,
        x=prediction.longitude,
        y=prediction.latitude,
        origin="lower",
        aspect="auto",
        labels={
            "x": "Longitude",
            "y": "Latitude",
            "color": "Error (°C)"
        }
    )

    fig_error.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(
        fig_error,
        width="stretch"
    )


# =========================================================
# VERTICAL PROFILE
# =========================================================

st.divider()

st.subheader("🌡️ Vertical Temperature Profile")

pred_profile = prediction.sel(
    time=selected_date,
    latitude=latitude,
    longitude=longitude,
    method="nearest"
)

ref_profile = reference.sel(
    time=selected_date,
    latitude=latitude,
    longitude=longitude,
    method="nearest"
)


fig_profile = go.Figure()

fig_profile.add_trace(
    go.Scatter(
        x=pred_profile.values,
        y=pred_profile.depth.values,
        mode="lines+markers",
        name="OceanEmbed"
    )
)

fig_profile.add_trace(
    go.Scatter(
        x=ref_profile.values,
        y=ref_profile.depth.values,
        mode="lines+markers",
        name="GLORYS Reference"
    )
)

fig_profile.update_layout(
    height=600,
    xaxis_title="Temperature (°C)",
    yaxis_title="Depth (m)",
    title=(
        f"Temperature Profile — "
        f"{latitude:.2f}°N, {longitude:.2f}°E"
    )
)

fig_profile.update_yaxes(
    autorange="reversed"
)

st.plotly_chart(
    fig_profile,
    width="stretch"
)


# =========================================================
# VALIDATION TABLE
# =========================================================

st.divider()

st.subheader("📈 Model Validation Across Depths")

depths = prediction.depth.values

rows = []

for d in depths:

    p = prediction.sel(
        depth=d
    ).values

    r = reference.sel(
        depth=d
    ).values

    mask = np.isfinite(p) & np.isfinite(r)

    if mask.sum() > 1:

        p = p[mask]
        r = r[mask]

        depth_rmse = np.sqrt(
            np.mean((p - r) ** 2)
        )

        depth_bias = np.mean(
            p - r
        )

        if np.std(p) > 0 and np.std(r) > 0:

            depth_corr = np.corrcoef(
                p,
                r
            )[0, 1]

        else:

            depth_corr = np.nan

    else:

        depth_rmse = np.nan
        depth_bias = np.nan
        depth_corr = np.nan

    rows.append(
        {
            "Depth (m)": int(d),
            "RMSE (°C)": round(
                float(depth_rmse), 3
            ),
            "Bias (°C)": round(
                float(depth_bias), 3
            ),
            "Correlation": round(
                float(depth_corr), 3
            )
        }
    )


st.dataframe(
    rows,
    width="stretch",
    hide_index=True
)


# =========================================================
# ARCHITECTURE
# =========================================================

st.divider()

st.subheader("🧠 OceanEmbed Pipeline")

st.markdown(
    """
    **Surface observations**

    SST • SSS • SSH • Surface Current U/V • Wind U/V

    ↓

    **7-channel ocean representation**

    ↓

    **Residual CNN embedding engine**

    ↓

    **Depth-dependent reconstruction**

    ↓

    **15 subsurface temperature levels**

    **0 • 5 • 10 • 20 • 30 • 50 • 75 • 100 •
    125 • 150 • 200 • 300 • 500 • 700 • 1000 m**

    ↓

    **Validation against GLORYS reanalysis**
    """
)


# =========================================================
# RESULTS HIGHLIGHT
# =========================================================

st.success(
    """
    **Current prototype result:** Model #3 achieves approximately
    **0.44 °C overall RMSE** on the four held-out test days
    (27–30 January 2020), with approximately **0.19 °C RMSE
    at 1000 m**.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "OceanEmbed • SIH26066 • "
    "Bay of Bengal prototype • "
    "GLORYS validation reference"
)