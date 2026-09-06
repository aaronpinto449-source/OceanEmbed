import numpy as np
import torch
import xarray as xr

from src.models.oceanembed_cnn import OceanEmbedCNN


DATA_FILE = "data/processed/oceanembed_training_30days.nc"
MODEL_FILE = "models/oceanembed_depthbaseline_30day.pt"
OUTPUT_FILE = "data/processed/oceanembed_prediction_depthbaseline_4days.nc"

INPUT_VARS = [
    "sst",
    "sss",
    "ssh",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
]

TARGET_VAR = "temperature"


def safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 2:
        return np.nan

    a = a[mask]
    b = b[mask]

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


print("Loading dataset...")
ds = xr.open_dataset(DATA_FILE)

print("Loading Model #3...")
checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False,
)

target_depths = np.asarray(checkpoint["target_depths"], dtype=np.float32)

input_means = np.asarray(checkpoint["input_means"], dtype=np.float32)
input_stds = np.asarray(checkpoint["input_stds"], dtype=np.float32)

target_means = np.asarray(checkpoint["target_means"], dtype=np.float32)
target_stds = np.asarray(checkpoint["target_stds"], dtype=np.float32)

model = OceanEmbedCNN(
    in_channels=len(INPUT_VARS),
    out_channels=len(target_depths),
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print()
print("Model information")
print("-----------------")
print("Best epoch:", checkpoint.get("best_epoch"))
print("Best validation loss:", checkpoint.get("best_val_loss"))
print("Target depths:", target_depths.tolist())

# Test dates = final four days
test_dates = ds.time.values[-4:]

print()
print("Test dates:")
for date in test_dates:
    print(" ", str(date)[:10])

all_predictions = []

# Statistics for overall evaluation
all_y_true = []
all_y_pred = []

print()
print("Running inference...")
print()

with torch.no_grad():

    for date in test_dates:

        daily = ds.sel(time=date)

        # ---------------------------------------------------------
        # RAW INPUT DATA
        # ---------------------------------------------------------
        X_raw = np.stack(
            [
                daily[var].values
                for var in INPUT_VARS
            ],
            axis=0,
        ).astype(np.float32)

        # Shape: channels, latitude, longitude
        if X_raw.ndim == 4:
            X_raw = X_raw[:, 0]

        # Input validity BEFORE NaN filling
        input_mask = np.all(
            np.isfinite(X_raw),
            axis=0,
        )

        # ---------------------------------------------------------
        # TARGET
        # ---------------------------------------------------------
        Y_raw = daily[TARGET_VAR].values.astype(np.float32)

        # depth, latitude, longitude
        if Y_raw.ndim == 4:
            Y_raw = Y_raw[:, 0]

        target_mask = np.isfinite(Y_raw)

        valid_mask = input_mask[None, :, :] & target_mask

        # ---------------------------------------------------------
        # NORMALIZE INPUTS
        # ---------------------------------------------------------
        X = X_raw.copy()

        for c in range(len(INPUT_VARS)):
            X[c] = (
                X[c] - input_means[c]
            ) / input_stds[c]

        # Fill missing input pixels after creating mask
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------
        # DEPTH-DEPENDENT BASELINE
        # ---------------------------------------------------------
        #
        # Every depth starts from its training climatological mean.
        # Surface (0 m) starts from the observed SST.
        #
        baseline_raw = np.zeros_like(Y_raw)

        for d in range(len(target_depths)):
            baseline_raw[d] = target_means[d]

        # Surface constraint: 0 m = SST
        baseline_raw[0] = X_raw[0]

        # Normalize baseline using target statistics
        baseline = (
            baseline_raw
            - target_means[:, None, None]
        ) / target_stds[:, None, None]

        baseline = np.nan_to_num(
            baseline,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------
        X_tensor = torch.from_numpy(X).unsqueeze(0)
        baseline_tensor = torch.from_numpy(baseline).unsqueeze(0)

        prediction_normalized = model(
            X_tensor,
            baseline_tensor,
        )

        prediction_normalized = (
            prediction_normalized
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # ---------------------------------------------------------
        # DENORMALIZE TO °C
        # ---------------------------------------------------------
        prediction = (
            prediction_normalized
            * target_stds[:, None, None]
            + target_means[:, None, None]
        )

        all_predictions.append(prediction)

        all_y_true.append(Y_raw[valid_mask])
        all_y_pred.append(prediction[valid_mask])

        print(
            f"Processed {str(date)[:10]} | "
            f"valid input coverage: "
            f"{100 * input_mask.mean():.2f}%"
        )


# -------------------------------------------------------------
# COMBINE TEST DATA
# -------------------------------------------------------------
all_y_true_flat = np.concatenate(all_y_true)
all_y_pred_flat = np.concatenate(all_y_pred)

overall_rmse = np.sqrt(
    np.mean(
        (all_y_pred_flat - all_y_true_flat) ** 2
    )
)

overall_bias = np.mean(
    all_y_pred_flat - all_y_true_flat
)

overall_corr = safe_corr(
    all_y_true_flat,
    all_y_pred_flat,
)

# -------------------------------------------------------------
# PER-DEPTH METRICS
# -------------------------------------------------------------
print()
print("=" * 72)
print("MODEL #3 TEST RESULTS")
print("=" * 72)

print(
    f"{'Depth':>8} "
    f"{'RMSE (°C)':>12} "
    f"{'Bias (°C)':>12} "
    f"{'Correlation':>14}"
)

print("-" * 72)

pred_stack = np.stack(all_predictions, axis=0)
true_stack = np.stack(
    [
        ds.sel(time=date)[TARGET_VAR].values.squeeze()
        for date in test_dates
    ],
    axis=0,
)

for d, depth in enumerate(target_depths):

    true_d = true_stack[:, d]
    pred_d = pred_stack[:, d]

    mask = np.isfinite(true_d)

    true_values = true_d[mask]
    pred_values = pred_d[mask]

    if len(true_values) == 0:
        rmse = np.nan
        bias = np.nan
        corr = np.nan

    else:
        rmse = np.sqrt(
            np.mean(
                (pred_values - true_values) ** 2
            )
        )

        bias = np.mean(
            pred_values - true_values
        )

        corr = safe_corr(
            true_values,
            pred_values,
        )

    print(
        f"{depth:8.0f} "
        f"{rmse:12.4f} "
        f"{bias:12.4f} "
        f"{corr:14.4f}"
    )


print("-" * 72)

print(
    f"{'OVERALL':>8} "
    f"{overall_rmse:12.4f} "
    f"{overall_bias:12.4f} "
    f"{overall_corr:14.4f}"
)

# -------------------------------------------------------------
# MEAN TEMPERATURE DIAGNOSTIC
# -------------------------------------------------------------
print()
print("Mean temperature diagnostic")
print("-" * 72)

print(
    f"{'Depth':>8} "
    f"{'Reference':>14} "
    f"{'Prediction':>14} "
    f"{'Difference':>14}"
)

print("-" * 72)

for d, depth in enumerate(target_depths):

    true_values = true_stack[:, d]
    pred_values = pred_stack[:, d]

    mask = np.isfinite(true_values)

    true_mean = np.mean(true_values[mask])
    pred_mean = np.mean(pred_values[mask])

    print(
        f"{depth:8.0f} "
        f"{true_mean:14.3f} "
        f"{pred_mean:14.3f} "
        f"{pred_mean - true_mean:14.3f}"
    )

# -------------------------------------------------------------
# SAVE PREDICTIONS
# -------------------------------------------------------------
prediction_ds = xr.Dataset(
    {
        "temperature_prediction": (
            ("time", "depth", "latitude", "longitude"),
            pred_stack,
        ),
        "temperature_reference": (
            ("time", "depth", "latitude", "longitude"),
            true_stack,
        ),
    },
    coords={
        "time": test_dates,
        "depth": target_depths,
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values,
    },
)

prediction_ds.attrs["project"] = "OceanEmbed"
prediction_ds.attrs["model"] = "OceanEmbedCNN depth-dependent baseline"
prediction_ds.attrs["test_period"] = (
    f"{str(test_dates[0])[:10]} to "
    f"{str(test_dates[-1])[:10]}"
)
prediction_ds.attrs["overall_rmse_celsius"] = float(
    overall_rmse
)
prediction_ds.attrs["overall_bias_celsius"] = float(
    overall_bias
)
prediction_ds.attrs["overall_correlation"] = float(
    overall_corr
)

prediction_ds.to_netcdf(
    OUTPUT_FILE,
)

print()
print("Prediction file saved to:")
print(OUTPUT_FILE)

print()
print("=" * 72)
print("EVALUATION COMPLETE")
print("=" * 72)