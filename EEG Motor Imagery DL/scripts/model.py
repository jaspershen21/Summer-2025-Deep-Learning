import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels = 22, n_temporal_filters = 8, depth_multiplier = 2, n_pointwise_filters = 16, n_classes = 4, dropout_rate = 0.5):
        super(EEGNet, self).__init__()

        # Hyperparameters
        # n_temporal_filters: Number of temporal convolution filters
        # depth_multiplier: Depth multiplier for spatial filters
        # n_pointwise_filters: Number of pointwise convolution filters
        # dropout_rate: Dropout rate for regularization

        # Block 1
        self.block1 = nn.Sequential(
            # Temporal Convolution
            nn.Conv2d(1, n_temporal_filters, kernel_size = (1, 125), padding = "same", bias = False),
            nn.BatchNorm2d(n_temporal_filters),

            # Depthwise Convolution
            nn.Conv2d(n_temporal_filters, depth_multiplier * n_temporal_filters, kernel_size = (n_channels, 1), groups = n_temporal_filters, bias = False),
            nn.BatchNorm2d(depth_multiplier * n_temporal_filters),
            nn.ELU(),

            # Pooling and Dropout
            nn.AvgPool2d(kernel_size = (1, 4)),
            nn.Dropout(dropout_rate)
        )

        # Block 2
        self.block2 = nn.Sequential(
            # Separable Convolution (Depthwise and Pointwise)
            nn.Conv2d(depth_multiplier * n_temporal_filters, n_pointwise_filters, kernel_size = (1, 31), padding = "same", groups = depth_multiplier * n_temporal_filters, bias = False),
            nn.Conv2d(n_pointwise_filters, n_pointwise_filters, kernel_size = (1, 1), bias = False),
            nn.BatchNorm2d(n_pointwise_filters),
            nn.ELU(),

            # Pooling and Dropout
            nn.AvgPool2d(kernel_size = (1, 8)),
            nn.Dropout(dropout_rate)
        )

        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input EEG data of shape (batch_size, 1, n_channels, n_time_points)
        
        Returns:
            torch.Tensor: Raw logits of shape (batch, n_classes)
        """

        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x