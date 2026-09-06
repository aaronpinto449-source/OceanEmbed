import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual convolution block used by OceanEmbed.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels
            ),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels
            ),
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(
            x + self.block(x)
        )


class OceanEmbedCNN(nn.Module):
    """
    OceanEmbed residual CNN.

    Input:
        7 surface ocean variables
        Shape:
            (batch, 7, latitude, longitude)

    Output:
        Temperature at 15 target depths
        Shape:
            (batch, 15, latitude, longitude)
    """

    def __init__(
        self,
        in_channels=7,
        out_channels=15
    ):
        super().__init__()

        # -------------------------------------------------
        # Surface observation encoder
        # -------------------------------------------------

        self.input_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=64
            ),
            nn.ReLU(),
        )

        # -------------------------------------------------
        # Learned ocean embedding
        # -------------------------------------------------

        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )

        # -------------------------------------------------
        # Depth-aware decoder
        # -------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=32
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                out_channels,
                kernel_size=1
            )
        )

    def forward(self, x, baseline=None):

        # -------------------------------------------------
        # Encode surface observations
        # -------------------------------------------------

        embedding = self.input_encoder(x)

        # -------------------------------------------------
        # Learn spatial ocean representation
        # -------------------------------------------------

        embedding = self.residual_blocks(
            embedding
        )

        # -------------------------------------------------
        # Predict depth-wise temperature corrections
        # -------------------------------------------------

        correction = self.decoder(
            embedding
        )

        # -------------------------------------------------
        # Residual baseline
        #
        # The training pipeline supplies SST converted into
        # the target-normalized temperature scale.
        # -------------------------------------------------

        if baseline is None:

            sst = x[:, 0:1, :, :]

            baseline = sst.expand(
                -1,
                correction.shape[1],
                -1,
                -1
            )

        temperature = (
            baseline + correction
        )

        # -------------------------------------------------
        # Physical surface constraint
        #
        # At 0 m, reconstructed temperature must equal
        # the observed SST baseline.
        # -------------------------------------------------

        temperature = temperature.clone()

        temperature[:, 0, :, :] = (
            baseline[:, 0, :, :]
        )

        return temperature


if __name__ == "__main__":

    print("Testing OceanEmbed residual CNN...")

    model = OceanEmbedCNN()

    x = torch.randn(
        2,
        7,
        61,
        41
    )

    y = model(x)

    print()
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    print()
    print("Surface difference:")

    surface_difference = (
        y[:, 0] - x[:, 0]
    ).abs().max()

    print(
        surface_difference.item()
    )

    print()
    print(model)