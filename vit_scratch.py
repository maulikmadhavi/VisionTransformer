# Vision Transformer in PyTorch
# https://www.youtube.com/watch?v=ovB0ddFtzzA


import torch
import torch.nn as nn
from torchinfo import summary

# Input creation
X = torch.randn(10, 3, 256, 256)

# Embedding layer
# Embedding layer
class PatchEmbed(nn.Module):
    """Split image into patches and then embed them

    Parameters
    ----------
    img_size : int
        Size of the image (it is a square).

    patch_size: int
        Size of the patch (it is a square).

    in chans: int
        Number of inptu channels.

    embed_dim: int
        The embedding dimension.

    Attributes
    ----------
    n_patches : int
        Number of patches inside of our image.

    proj : nn.Conv2d
        Convolutional layer that does both the splitting into patches and their embedding.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.n_patches = (self.img_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`

        Returns
        ---------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`
        """
        x = self.proj(x)  # (n_samples, embed_dim, n_patches**0.5, n_patches**0.5)

        n_samples, _, _, _ = x.shape
        x = x.view(n_samples, self.embed_dim, -1).transpose(
            1, 2
        )  # (n_samples, n_patches, embed_dim)
        return x


# Attention Module
class Attention(nn.Module):
    """Attention mechanism

    Parameters
    ----------
    dim : int
        The input and output dimension of per token features

    n_heads: int
        Number of attention heads

    qkv_bias: bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.

    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    attn_drop, proj_drop: nn.Dropout
        Dropout layers.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_p: float = 0.0,
        proj_p: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim should be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_p = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_p = nn.Dropout(proj_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        --------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_p(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_p(x)
        return x


class _Mlp(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features: int
        Number of hidden features.

    out_features: int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches+1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches+1, out_features)`.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block

    Parameters
    ----------
    dim: int
        Embedding dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio: float
        Determine the hidden dimension side of the `MLP` module wrt `dim`.

    qkv_bias: bool
        If True then we include bias to the query, key and value projectsions.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2: nn.LayerNorm
        Layer normalization.

    attn: Attention
        Attention module.

    mlp: MLP
        MLP module.
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0.0, attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_size = int(dim * mlp_ratio)
        self.mlp = _Mlp(
            in_features=dim,
            hidden_features=hidden_size,
            out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches+1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches+1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Simplified implementation of the vision transformer.

    Parameters
    ----------
    img_size: int
        Both height and width of the image (it is a square).

    patch_size: int
        Both height and width of the image (it is a square).

    in_chans : int
        Number of input channels.

    n_classes: int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth: int
        Number of blocks.

    n_heads: int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed: PatchEmbed
        Instance of `PatchEmbed` layer.

    cls_token: nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.

    pos_emb: nn.Parameter
        Learnable positional embedding parametrs of cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.

    """

    def __init__(
        self,
        img_size=384,
        patch_size=16,
        in_chans=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        p=0.0,
        attn_p=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embd = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        print(x.shape)
        x = self.patch_embed(x)  # (n_samples, n_patches, embed_dim)

        print(x.shape)
        cls_tokens = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, embed_dim)

        x = torch.cat([cls_tokens, x], 1)  # (n_samples, n_patches, embed_dim)

        x = x + self.pos_embd  # (n_samples, n_patches, embed_dim)

        x = self.pos_drop(x)  # (n_samples, n_patches, embed_dim)

        # iteratively applied the blocks of transformer layers
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)  # (n_samples, n_patches, embed_dim)  layer normalization

        cls_token_final = x[:, 0]  # Just the CLS token (shape: (n_samples, embed_dim))

        x = self.head(cls_token_final)  # (n_samples, n_classes)

        return x


X = torch.randn(10, 3, 224, 224)

M = VisionTransformer(
    img_size=224,
    patch_size=16,
    embed_dim=192,
    depth=12,
    n_heads=3,
    mlp_ratio=4,
    qkv_bias=True,
)

Y = M(X)
print(X.shape)
print(Y.shape)

print(summary(M, (2, 3, 224, 224)))