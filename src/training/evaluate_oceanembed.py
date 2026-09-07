import numpy as np
import torch
import xarray as xr

from src.models.oceanembed_cnn import OceanEmbedCNN


DATA_FILE = "data/processed/oceanembed_training_2020-01-15.nc"
MODEL_FILE = "models/oceanembed_prototype.pt"


print("Loading data...")
ds = xr.open_dataset(DATA_FILE)

input_variables = [
    "sst",
    "sss",
    "ssh",
    "u_current",
    "v_current",
    "u_wind",
    "v_wind",
]

# -----------------------------
# Prepare inputs
# -----------------------------

input_arrays = []

for variable in input_variables:
    values = ds[variable].values.astype(np.float32)
    input_arrays.append(values)

X = np.stack(input_arrays, axis=0)

# Only use cells where all 7 inputs exist
input_mask = np.all(np.isfinite(X), axis=0)

# -----------------------------
# Normalize inputs
# -----------------------------

means = np.zeros(7, dtype=np.float32)
stds = np.ones(7, dtype=np.float32)

for i in range(7):
    valid_values = X[i][input_mask]

    means[i] = valid_values.mean()
    stds[i] = valid_values.std()

    if stds[i] < 1e-6:
        stds[i] = 1.0

    X[i] = (X[i] - means[i]) / stds[i]

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------
# Prepare target
# -----------------------------

Y = ds["temperature"].values.astype(np.float32)

target_mask = np.isfinite(Y)

Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

# -----------------------------
# Convert to tensors
# -----------------------------

X_tensor = torch.from_numpy(X).unsqueeze(0)

# -----------------------------
# Load model
# -----------------------------

print("Loading trained model...")

model = OceanEmbedCNN(
    in_channels=7,
    out_channels=15
)

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Model loaded successfully.")

# -----------------------------
# Run prediction
# -----------------------------

print("Running prediction...")

with torch.no_grad():
    prediction = model(X_tensor)

prediction = prediction.squeeze(0).numpy()

# -----------------------------
# Calculate metrics
# -----------------------------

print()
print("=" * 55)
print("OCEANEMBED MODEL EVALUATION")
print("=" * 55)

depths = ds.depth.values

for i, depth in enumerate(depths):

    valid = input_mask & target_mask[i]

    actual = Y[i][valid]
    predicted = prediction[i][valid]

    if len(actual) == 0:
        print(f"{depth:7.1f} m | No valid data")
        continue

    error = predicted - actual

    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)

    if len(actual) > 1:
        correlation = np.corrcoef(actual, predicted)[0, 1]
    else:
        correlation = np.nan

    print(
        f"{depth:7.1f} m | "
        f"RMSE = {rmse:7.3f} °C | "
        f"Bias = {bias:7.3f} °C | "
        f"Correlation = {correlation:6.3f}"
    )

print("=" * 55)

# -----------------------------
# Overall metrics
# -----------------------------

valid_all = input_mask[None, :, :] & target_mask

actual_all = Y[valid_all]
predicted_all = prediction[valid_all]

overall_error = predicted_all - actual_all

overall_rmse = np.sqrt(
    np.mean(overall_error ** 2)
)

overall_bias = np.mean(overall_error)

overall_correlation = np.corrcoef(
    actual_all,
    predicted_all
)[0, 1]

print()
print("OVERALL PERFORMANCE")
print("-" * 55)
print(f"Overall RMSE       : {overall_rmse:.3f} °C")
print(f"Overall Bias       : {overall_bias:.3f} °C")
print(f"Overall Correlation: {overall_correlation:.3f}")
print("-" * 55)

# -----------------------------
# Save prediction
# -----------------------------

prediction_da = xr.DataArray(
    prediction,
    dims=("depth", "latitude", "longitude"),
    coords={
        "depth": ds.depth.values,
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values,
    },
    name="predicted_temperature",
)

output = xr.Dataset({
    "predicted_temperature": prediction_da,
    "reference_temperature": ds["temperature"],
})

output.attrs["project"] = "OceanEmbed"
output.attrs["problem_statement"] = "SIH26066"
output.attrs["model"] = "OceanEmbedCNN"

output_file = "data/processed/oceanembed_prediction_2020-01-15.nc"

output.to_netcdf(output_file)

print()
print(f"Prediction saved to: {output_file}")