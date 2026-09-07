import numpy as np
import torch
import xarray as xr

from src.models.oceanembed_cnn import OceanEmbedCNN


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "data/processed/oceanembed_training_30days.nc"
MODEL_FILE = "models/oceanembed_residual_30day_prototype.pt"

OUTPUT_FILE = "data/processed/oceanembed_prediction_test_4days.nc"

INPUT_VARIABLES = [
    "sst",
    "sss",
    "ssh",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
]

TARGET_VARIABLE = "temperature"

TARGET_DEPTHS = [
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000,
]


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(reference, prediction, mask):
    """
    Calculate RMSE, bias and correlation using only valid points.
    """

    ref = reference[mask]
    pred = prediction[mask]

    if len(ref) == 0:
        return np.nan, np.nan, np.nan

    rmse = np.sqrt(
        np.mean((pred - ref) ** 2)
    )

    bias = np.mean(
        pred - ref
    )

    if len(ref) > 1:
        correlation = np.corrcoef(
            ref,
            pred
        )[0, 1]
    else:
        correlation = np.nan

    return rmse, bias, correlation


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("=" * 60)
    print("OceanEmbed residual CNN evaluation")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    ds = xr.open_dataset(DATA_FILE)

    print(
        f"Dataset dimensions: "
        f"time={ds.sizes['time']}, "
        f"latitude={ds.sizes['latitude']}, "
        f"longitude={ds.sizes['longitude']}, "
        f"depth={ds.sizes['depth']}"
    )

    # --------------------------------------------------------
    # Load model checkpoint
    # --------------------------------------------------------

    print("\nLoading model...")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
         weights_only=False
    )

    model = OceanEmbedCNN(
        in_channels=7,
        out_channels=15
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Retrieve normalization statistics
    # --------------------------------------------------------

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

    print("\nNormalization statistics loaded.")

    # --------------------------------------------------------
    # Test period
    #
    # Training:
    #   Jan 1-22
    #
    # Validation:
    #   Jan 23-26
    #
    # Test:
    #   Jan 27-30
    # --------------------------------------------------------

    test_dates = ds.time.values[-4:]

    print("\nTest dates:")

    for date in test_dates:
        print(
            " ",
            str(date)[:10]
        )

    # --------------------------------------------------------
    # Prepare all test input data
    # --------------------------------------------------------

    X_list = []
    baseline_list = []
    target_list = []

    for date in test_dates:

        date_string = str(date)[:10]

        daily = ds.sel(
            time=date
        )

        # ----------------------------------------------------
        # Inputs
        # ----------------------------------------------------

        raw_inputs = np.stack(
            [
                daily[var].values
                for var in INPUT_VARIABLES
            ],
            axis=0
        ).astype(np.float32)

        # ----------------------------------------------------
        # Input validity mask BEFORE filling NaNs
        # ----------------------------------------------------

        input_valid = np.all(
            np.isfinite(raw_inputs),
            axis=0
        )

        # ----------------------------------------------------
        # Normalize inputs
        # ----------------------------------------------------

        normalized_inputs = (
            raw_inputs
            - input_means[:, None, None]
        ) / input_stds[:, None, None]

        # ----------------------------------------------------
        # Replace NaNs after mask creation
        # ----------------------------------------------------

        normalized_inputs = np.nan_to_num(
            normalized_inputs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Raw SST
        # ----------------------------------------------------

        raw_sst = raw_inputs[0]

        # ----------------------------------------------------
        # SST residual baseline
        #
        # Convert raw SST to target-normalized temperature
        # scale separately for every depth.
        # ----------------------------------------------------

        baseline = (
            raw_sst[None, :, :]
            - target_means[:, None, None]
        ) / target_stds[:, None, None]

        # ----------------------------------------------------
        # Target temperature
        # ----------------------------------------------------

        target = daily[
            TARGET_VARIABLE
        ].values.astype(
            np.float32
        )

        # Ensure depth is first dimension
        if target.shape[0] != len(TARGET_DEPTHS):
            raise ValueError(
                f"Unexpected target shape: {target.shape}"
            )

        target_valid = np.isfinite(
            target
        )

        # ----------------------------------------------------
        # Fill invalid target values
        # ----------------------------------------------------

        target_filled = np.nan_to_num(
            target,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Normalize target
        # ----------------------------------------------------

        normalized_target = (
            target_filled
            - target_means[:, None, None]
        ) / target_stds[:, None, None]

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_list.append(
            normalized_inputs
        )

        baseline_list.append(
            baseline
        )

        target_list.append(
            target
        )

        print(
            f"Prepared {date_string}: "
            f"valid input grid = "
            f"{input_valid.mean() * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X = torch.tensor(
        np.stack(X_list),
        dtype=torch.float32
    )

    baseline = torch.tensor(
        np.stack(baseline_list),
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    print("\nRunning model...")

    with torch.no_grad():

        prediction_normalized = model(
            X,
            baseline
        )

    prediction_normalized = (
        prediction_normalized
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert prediction back to °C
    # --------------------------------------------------------

    prediction = (
        prediction_normalized
        * target_stds[None, :, None, None]
        + target_means[None, :, None, None]
    )

    reference = np.stack(
        target_list
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print(
        f"{'Depth (m)':>10} "
        f"{'RMSE (°C)':>15} "
        f"{'Bias (°C)':>15} "
        f"{'Correlation':>15}"
    )
    print("=" * 90)

    all_errors = []
    all_reference = []
    all_predictions = []

    metrics = []

    for depth_index, depth in enumerate(
        TARGET_DEPTHS
    ):

        ref_depth = reference[
            :,
            depth_index
        ]

        pred_depth = prediction[
            :,
            depth_index
        ]

        mask = (
            np.isfinite(ref_depth)
            & np.isfinite(pred_depth)
        )

        rmse, bias, correlation = calculate_metrics(
            ref_depth,
            pred_depth,
            mask
        )

        metrics.append(
            (
                depth,
                rmse,
                bias,
                correlation
            )
        )

        print(
            f"{depth:>10} "
            f"{rmse:>15.4f} "
            f"{bias:>15.4f} "
            f"{correlation:>15.4f}"
        )

        if np.any(mask):

            all_errors.extend(
                (
                    pred_depth[mask]
                    - ref_depth[mask]
                ).tolist()
            )

            all_reference.extend(
                ref_depth[mask].tolist()
            )

            all_predictions.extend(
                pred_depth[mask].tolist()
            )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    all_errors = np.asarray(
        all_errors
    )

    all_reference = np.asarray(
        all_reference
    )

    all_predictions = np.asarray(
        all_predictions
    )

    overall_rmse = np.sqrt(
        np.mean(
            all_errors ** 2
        )
    )

    overall_bias = np.mean(
        all_errors
    )

    overall_correlation = np.corrcoef(
        all_reference,
        all_predictions
    )[0, 1]

    print("=" * 90)

    print(
        f"\nOverall RMSE : "
        f"{overall_rmse:.4f} °C"
    )

    print(
        f"Overall Bias : "
        f"{overall_bias:.4f} °C"
    )

    print(
        f"Overall Corr. : "
        f"{overall_correlation:.4f}"
    )

    # --------------------------------------------------------
    # Save prediction dataset
    # --------------------------------------------------------

    prediction_da = xr.DataArray(
        prediction,
        dims=[
            "time",
            "depth",
            "latitude",
            "longitude"
        ],
        coords={
            "time": test_dates,
            "depth": TARGET_DEPTHS,
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
        name="predicted_temperature",
        attrs={
            "units": "degrees_C",
            "model": "OceanEmbed residual CNN",
            "target": "GLORYS temperature",
        }
    )

    reference_da = xr.DataArray(
        reference,
        dims=[
            "time",
            "depth",
            "latitude",
            "longitude"
        ],
        coords={
            "time": test_dates,
            "depth": TARGET_DEPTHS,
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
        name="reference_temperature",
        attrs={
            "units": "degrees_C",
            "source": "GLORYS",
        }
    )

    output_ds = xr.Dataset(
        {
            "predicted_temperature": prediction_da,
            "reference_temperature": reference_da,
        }
    )

    output_ds.attrs = {
        "project": "OceanEmbed",
        "problem_statement": "SIH26066",
        "region": "Bay of Bengal prototype",
        "test_period": "2020-01-27 to 2020-01-30",
    }

    output_ds.to_netcdf(
        OUTPUT_FILE
    )

    print(
        f"\nPrediction saved to: "
        f"{OUTPUT_FILE}"
    )

    ds.close()


if __name__ == "__main__":
    main()