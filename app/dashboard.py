import streamlit as st
import xarray as xr
import plotly.express as px
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="OceanEmbed",
    page_icon="🌊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🌊 OceanEmbed")

st.markdown(
    """
    ### Satellite Embedding-Based Subsurface Ocean Temperature Reconstruction

    **SIH26066 — North Indian Ocean**
    """
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

DATA_PATH = Path(
    "data/processed/glorys_bob_2020-01-15_025deg.nc"
)

if not DATA_PATH.exists():

    st.error(
        "GLORYS processed data was not found."
    )

    st.stop()


ds = xr.open_dataset(DATA_PATH)

temperature = ds["temperature"]


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Ocean Controls")

depth = st.sidebar.selectbox(
    "Select depth (m)",
    temperature.depth.values.tolist()
)


# ---------------------------------------------------------
# SELECT DEPTH
# ---------------------------------------------------------

temp_at_depth = temperature.sel(
    depth=depth
).squeeze()


# ---------------------------------------------------------
# TEMPERATURE MAP
# ---------------------------------------------------------

st.subheader(
    f"Ocean Temperature at {depth:.0f} m"
)

fig = px.imshow(
    temp_at_depth,
    x=ds.longitude,
    y=ds.latitude,
    origin="lower",
    aspect="auto",
    labels={
        "x": "Longitude",
        "y": "Latitude",
        "color": "Temperature (°C)"
    }
)

fig.update_layout(
    height=550
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# DATA INFORMATION
# ---------------------------------------------------------

st.subheader("Dataset Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latitude points",
    len(ds.latitude)
)

col2.metric(
    "Longitude points",
    len(ds.longitude)
)

col3.metric(
    "Depth levels",
    len(ds.depth)
)

col4.metric(
    "Date",
    str(ds.time.values[0])[:10]
)


# ---------------------------------------------------------
# TEMPERATURE PROFILE
# ---------------------------------------------------------

st.subheader("Vertical Temperature Profile")

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=float(ds.latitude.min()),
    max_value=float(ds.latitude.max()),
    value=15.0,
    step=0.25
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=float(ds.longitude.min()),
    max_value=float(ds.longitude.max()),
    value=85.0,
    step=0.25
)


profile = temperature.sel(
    latitude=latitude,
    longitude=longitude,
    method="nearest"
).squeeze()


fig_profile = px.line(
    x=profile.values,
    y=profile.depth.values,
    markers=True,
    labels={
        "x": "Temperature (°C)",
        "y": "Depth (m)"
    },
    title=(
        f"Temperature Profile "
        f"({latitude:.2f}°N, {longitude:.2f}°E)"
    )
)

fig_profile.update_yaxes(
    autorange="reversed"
)

fig_profile.update_layout(
    height=600
)

st.plotly_chart(
    fig_profile,
    use_container_width=True
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "OceanEmbed • SIH26066 • GLORYS reference dataset"
)