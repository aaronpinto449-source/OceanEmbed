import os
import numpy as np
import torch
import xarray as xr

from src.models.oceanembed_cnn import OceanEmbedCNN


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "data/processed/oceanembed_training_30days.nc"

MODEL_DIR = "models"

MODEL_FILE = (
    "models/oceanembed_depthbaseline_30day.pt"
)

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

EPOCHS = 150

LEARNING_RATE = 5e-4

WEIGHT_DECAY = 1e-5


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("OceanEmbed depth-baseline training")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    ds = xr.open_dataset(DATA_FILE)

    print(ds)

    dates = ds.time.values

    n_days = len(dates)

    print(
        f"\nTotal days: {n_days}"
    )

    # --------------------------------------------------------
    # Train / validation / test split
    #
    # Jan 1-22  -> training
    # Jan 23-26  -> validation
    # Jan 27-30  -> test
    # --------------------------------------------------------

    train_indices = np.arange(
        0,
        22
    )

    val_indices = np.arange(
        22,
        26
    )

    test_indices = np.arange(
        26,
        30
    )

    print("\nData split:")

    print(
        "Training:",
        str(dates[train_indices[0]])[:10],
        "to",
        str(dates[train_indices[-1]])[:10]
    )

    print(
        "Validation:",
        str(dates[val_indices[0]])[:10],
        "to",
        str(dates[val_indices[-1]])[:10]
    )

    print(
        "Testing:",
        str(dates[test_indices[0]])[:10],
        "to",
        str(dates[test_indices[-1]])[:10]
    )

    # --------------------------------------------------------
    # Load input variables
    # --------------------------------------------------------

    print("\nLoading input variables...")

    X = np.stack(
        [
            ds[var].values
            for var in INPUT_VARIABLES
        ],
        axis=1
    ).astype(
        np.float32
    )

    # X shape:
    #
    # (time, channels, latitude, longitude)
    #

    print(
        "Input shape:",
        X.shape
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    Y = ds[
        TARGET_VARIABLE
    ].values.astype(
        np.float32
    )

    # Y shape:
    #
    # (time, depth, latitude, longitude)
    #

    print(
        "Target shape:",
        Y.shape
    )

    n_depths = Y.shape[1]

    # --------------------------------------------------------
    # Masks BEFORE filling NaNs
    # --------------------------------------------------------

    print("\nCreating validity masks...")

    input_mask = np.all(
        np.isfinite(X),
        axis=1
    )

    target_mask = np.isfinite(
        Y
    )

    # --------------------------------------------------------
    # Training-only input normalization
    # --------------------------------------------------------

    print("\nCalculating input normalization...")

    X_train_raw = X[
        train_indices
    ]

    input_means = np.nanmean(
        X_train_raw,
        axis=(0, 2, 3)
    )

    input_stds = np.nanstd(
        X_train_raw,
        axis=(0, 2, 3)
    )

    input_stds = np.maximum(
        input_stds,
        1e-6
    )

    print("\nInput means:")

    for name, value in zip(
        INPUT_VARIABLES,
        input_means
    ):
        print(
            f"{name:15s}: {value:.6f}"
        )

    print("\nInput standard deviations:")

    for name, value in zip(
        INPUT_VARIABLES,
        input_stds
    ):
        print(
            f"{name:15s}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Normalize inputs
    # --------------------------------------------------------

    X_normalized = (
        X
        - input_means[
            None,
            :,
            None,
            None
        ]
    ) / input_stds[
        None,
        :,
        None,
        None
    ]

    # Fill invalid inputs AFTER mask creation

    X_normalized = np.nan_to_num(
        X_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Training-only target statistics
    #
    # Statistics are calculated separately for every depth.
    # --------------------------------------------------------

    print(
        "\nCalculating depth-wise target statistics..."
    )

    Y_train_raw = Y[
        train_indices
    ]

    target_means = np.nanmean(
        Y_train_raw,
        axis=(0, 2, 3)
    )

    target_stds = np.nanstd(
        Y_train_raw,
        axis=(0, 2, 3)
    )

    target_stds = np.maximum(
        target_stds,
        1e-6
    )

    depths = ds.depth.values

    print("\nDepth statistics:")

    for i in range(
        n_depths
    ):

        print(
            f"{float(depths[i]):7.1f} m"
            f"  mean={target_means[i]:8.3f}"
            f"  std={target_stds[i]:8.3f}"
        )

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

    Y_normalized = (
        Y
        - target_means[
            None,
            :,
            None,
            None
        ]
    ) / target_stds[
        None,
        :,
        None,
        None
    ]

    Y_normalized = np.nan_to_num(
        Y_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Depth-dependent baseline
    #
    # Deep levels use the training-period climatological
    # temperature profile.
    #
    # Since the target is normalized using the same training
    # statistics, the climatological baseline is approximately
    # zero in normalized space.
    #
    # At 0 m, however, we use the actual SST.
    # --------------------------------------------------------

    print(
        "\nBuilding depth-dependent baseline..."
    )

    raw_sst = X[
        :,
        0,
        :,
        :
    ]

    # Start with climatological baseline.
    #
    # Shape:
    # (time, depth, latitude, longitude)

    baseline = np.broadcast_to(
        target_means[
            None,
            :,
            None,
            None
        ],
        (
            X.shape[0],
            n_depths,
            X.shape[2],
            X.shape[3]
        )
    ).copy()

    # Replace 0 m baseline with observed SST

    baseline[
        :,
        0,
        :,
        :
    ] = raw_sst

    # Convert baseline into normalized target space

    baseline_normalized = (
        baseline
        - target_means[
            None,
            :,
            None,
            None
        ]
    ) / target_stds[
        None,
        :,
        None,
        None
    ]

    baseline_normalized = np.nan_to_num(
        baseline_normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tensor = torch.from_numpy(
        X_normalized
    )

    Y_tensor = torch.from_numpy(
        Y_normalized
    )

    baseline_tensor = torch.from_numpy(
        baseline_normalized
    )

    input_mask_tensor = torch.from_numpy(
        input_mask
    )

    target_mask_tensor = torch.from_numpy(
        target_mask
    )

    # --------------------------------------------------------
    # Split tensors
    # --------------------------------------------------------

    X_train = X_tensor[
        train_indices
    ]

    X_val = X_tensor[
        val_indices
    ]

    X_test = X_tensor[
        test_indices
    ]

    Y_train = Y_tensor[
        train_indices
    ]

    Y_val = Y_tensor[
        val_indices
    ]

    Y_test = Y_tensor[
        test_indices
    ]

    baseline_train = baseline_tensor[
        train_indices
    ]

    baseline_val = baseline_tensor[
        val_indices
    ]

    baseline_test = baseline_tensor[
        test_indices
    ]

    mask_train = input_mask_tensor[
        train_indices
    ]

    mask_val = input_mask_tensor[
        val_indices
    ]

    mask_test = input_mask_tensor[
        test_indices
    ]

    target_mask_train = target_mask_tensor[
        train_indices
    ]

    target_mask_val = target_mask_tensor[
        val_indices
    ]

    target_mask_test = target_mask_tensor[
        test_indices
    ]

    print("\nTensor shapes:")

    print(
        "X train:",
        X_train.shape
    )

    print(
        "Y train:",
        Y_train.shape
    )

    print(
        "X validation:",
        X_val.shape
    )

    print(
        "X test:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print(
        "\nCreating OceanEmbed model..."
    )

    model = OceanEmbedCNN(
        in_channels=7,
        out_channels=n_depths
    )

    print(model)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Depth loss weights
    #
    # Deeper levels receive more importance.
    # We normalize the weights so their mean is 1.
    # --------------------------------------------------------

    depth_weights = torch.linspace(
        1.0,
        4.0,
        n_depths
    )

    depth_weights = (
        depth_weights
        / depth_weights.mean()
    )

    depth_weights = depth_weights.view(
        1,
        n_depths,
        1,
        1
    )

    print(
        "\nDepth loss weights:"
    )

    for depth, weight in zip(
        depths,
        depth_weights.flatten()
    ):

        print(
            f"{float(depth):7.1f} m"
            f" -> {weight.item():.3f}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "Starting training..."
    )
    print(
        "-" * 60
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        optimizer.zero_grad()

        prediction = model(
            X_train,
            baseline_train
        )

        valid_mask = (
            mask_train.unsqueeze(1)
            & target_mask_train
        )

        squared_error = (
            prediction
            - Y_train
        ) ** 2

        # Apply depth weighting

        weighted_error = (
            squared_error
            * depth_weights
        )

        loss = weighted_error[
            valid_mask
        ].mean()

        loss.backward()

        # Prevent unstable gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_val,
                baseline_val
            )

            val_mask = (
                mask_val.unsqueeze(1)
                & target_mask_val
            )

            val_squared_error = (
                val_prediction
                - Y_val
            ) ** 2

            val_weighted_error = (
                val_squared_error
                * depth_weights
            )

            val_loss = val_weighted_error[
                val_mask
            ].mean()

        # ----------------------------------------------------
        # Save best model in memory
        # ----------------------------------------------------

        if val_loss.item() < best_val_loss:

            best_val_loss = (
                val_loss.item()
            )

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        model.train()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(
                f"Epoch {epoch:3d}/{EPOCHS}"
                f" | Train Loss: {loss.item():.6f}"
                f" | Val Loss: {val_loss.item():.6f}"
            )

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------

    print()
    print(
        "Restoring best validation model..."
    )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Test evaluation in normalized space
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        test_prediction = model(
            X_test,
            baseline_test
        )

        test_mask = (
            mask_test.unsqueeze(1)
            & target_mask_test
        )

        test_squared_error = (
            test_prediction
            - Y_test
        ) ** 2

        test_weighted_error = (
            test_squared_error
            * depth_weights
        )

        test_loss = test_weighted_error[
            test_mask
        ].mean()

    print()
    print(
        "========================================"
    )
    print(
        "TRAINING COMPLETE"
    )
    print(
        "========================================"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Test weighted loss: "
        f"{test_loss.item():.6f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_variables":
                INPUT_VARIABLES,

            "target_variable":
                TARGET_VARIABLE,

            "input_means":
                input_means.tolist(),

            "input_stds":
                input_stds.tolist(),

            "target_means":
                target_means.tolist(),

            "target_stds":
                target_stds.tolist(),

            "target_depths":
                depths.tolist(),

            "depth_loss_weights":
                depth_weights.flatten().tolist(),

            "train_dates":
                [
                    str(dates[i])[:10]
                    for i in train_indices
                ],

            "validation_dates":
                [
                    str(dates[i])[:10]
                    for i in val_indices
                ],

            "test_dates":
                [
                    str(dates[i])[:10]
                    for i in test_indices
                ],

            "best_epoch":
                best_epoch,

            "best_validation_loss":
                best_val_loss,
        },
        MODEL_FILE
    )

    print()
    print(
        "Model saved to:",
        MODEL_FILE
    )

    ds.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()