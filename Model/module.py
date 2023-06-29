import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 16,
            embedding_dim: int = 768
            ):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(
            in_channels = in_channels,
            out_channels = embedding_dim,
            kernel_size = patch_size,
            stride = patch_size,
            padding = 0
            )
        self.flattener = nn.Flatten(start_dim = 2, end_dim = 3)

    def forward(self, x):
        image_size = x.shape[-1]
        assert image_size % self.patch_size == 0, f"Input image size must be divisible by patch " \
                                                  f"size, please re-check the patch size:" \
                                                  f"{self.patch_size} and the image size: " \
                                                  f"{image_size}."

        return self.flattener(self.patcher(x)).permute(0, 2, 1)


class MSA(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            h = 12,
            dropout: float = .0
            ):
        super().__init__()
        self.norm_layer = nn.LayerNorm(normalized_shape = embedding_dim)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim = embedding_dim,
            num_heads = h,
            dropout = dropout,
            batch_first = True
            )

    def forward(self, x):
        x = self.norm_layer(x)
        # we don't need the attention weights but just the layer output, so need_weights = Flase
        x, _ = self.multi_head_attention(query = x, key = x, value = x, need_weights = False)
        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim: int = 768, mlp_size: int = 3072, dropout: float = .1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape = embedding_dim)
        self.mlp_body = nn.Sequential(
                nn.Linear(in_features = embedding_dim, out_features = mlp_size),
                nn.GELU(),
                nn.Dropout(p = dropout),
                nn.Linear(in_features = mlp_size, out_features = embedding_dim),
                nn.Dropout(p = dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp_body(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self, embedding_dim: int = 768, h: int = 12, mlp_size: int = 3072, mlp_dropout:
            float = .1, msa_dropout: float = .0
            ):
        super().__init__()
        self.msa = MSA(embedding_dim = embedding_dim, h = h, dropout = msa_dropout)
        self.mlp = MLP(embedding_dim = embedding_dim, mlp_size = mlp_size, dropout = mlp_dropout)

    def forward(self, x):
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x

class StandardVit(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 transformer_layers_num: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 h: int = 12,
                 msa_dropout: float = .0,
                 mlp_dropout: float = .1,
                 embedding_dropout: float = .1,
                 num_classes: int = 1000):
        super().__init__()
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, " \
                                           f"image size: {img_size}, patch size: {patch_size}."

        self.patch_num = (img_size * img_size) // (patch_size ** 2)

        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim),
                                            requires_grad = True)

        self.position_embedding = nn.Parameter(
            data = torch.randn(1, self.patch_num + 1, embedding_dim), requires_grad = True)

        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels = in_channels, patch_size = patch_size,
                                              embedding_dim = embedding_dim)

        self.transformer_encoder = nn.Sequential(*[Encoder(embedding_dim = embedding_dim, h = h,
                                                           mlp_size = mlp_size, mlp_dropout =
                                                           mlp_dropout, msa_dropout =
                                                           msa_dropout) for _ in range(transformer_layers_num)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = embedding_dim),
                                        nn.Linear(in_features = embedding_dim, out_features = num_classes))


    def forward(self, x):
        batch_size = x.shape[0]

        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim = 1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x


class TinyVGG(nn.Module):
    def __init__(self, in_channel: int, hidden_unit: int, classes_num:int) -> None:
        self.block1 = nn.Sequential(
                nn.Conv2d(in_channels = in_channel, out_channels = hidden_unit, kernel_size = 3,
                          stride = 1, padding = 0),
                nn.ReLU,
                nn.Conv2d(in_channels = hidden_unit, out_channels = hidden_unit, kernel_size = 3,
                          stride = 1, padding = 0),
                nn.ReLU,
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )

        self.block2 = nn.Sequential(
                nn.nn.Conv2d(in_channels = hidden_unit, out_channels = hidden_unit, kernel_size = 3,
                          stride = 1, padding = 0),
                nn.ReLU,
                nn.Conv2d(in_channels = hidden_unit, out_channels = hidden_unit, kernel_size = 3,
                          stride = 1, padding = 0),
                nn.ReLU,
                nn.MaxPool2d(kernel_size = 2, stride = 2)
                )

        self.block3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features = hidden_unit * 13 ** 2, out_features = classes_num)
                )

    def forward(self, x):
        return self.block3(self.block2(self.block1(x)))
















