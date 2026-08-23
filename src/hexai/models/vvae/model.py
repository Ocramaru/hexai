"""Pure visual producer for the Neural Hub. Two families share the blocks and the config shape:

VVAE            encodes a frame stack and reconstructs it, latent axes are (time, row, col)
VVAEGeometric   unprojects posed frames into a metric box and renders a query pose, latent axes
                are space

Neither carries task heads, so adapters interpret the latent without shaping the producer.

Author: Marco Cassar (@Ocramaru)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from math import log

import torch
import torch.nn.functional as F
from torch import nn

from hexai.models.vvae.geometry import carve_window, lift_window, render_view


@dataclass
class VVAEConfig:
  cube_size: int = 7  # cube size of latent 7x7x7 and such
  latent_channels: int = 4  # think number of free vars in each latent space coordinate
  in_channels: int = 2  # number of channels in the input frames (e.g., grayscale=1, RGB=3) or 2 for greyscale + depth approximate
  act: type[nn.Module] = nn.ReLU  # assign a default activation function (e.g., ReLU)

  base_channels: int = 32  # first layer channels, doubled each downsample
  n_layers: int = 4  # number of passes to collapse to latent space
  input_frame_hw: tuple[int, int] = (162, 216)  # input frame height and width
  num_frames: int = 5  # depth axis size -- frames stacked as the 3rd spatial dim
  kernel_size: int = 5  # kernel size for conv layers
  downsample_scale: int = 2  # spatial reduction factor per level -- higher means
  # fewer levels to reach cube_size, which caps max channel width sooner (channels
  # double per level, so fewer levels matters a lot at large input_frame_hw)

  use_skips: bool = True  # enable skipping connections between encoder and decoder
  norm: Callable[[int], nn.Module] | None = partial(nn.GroupNorm, 8)  # or nn.BatchNorm3d, None
  dropout: float = 0.0  # Try out some levels of dropout | 0 equals no dropout
  groups: int = 1  # nn.Conv3d groups, 1 is none

  blocks_per_level: int = 2  # number of blocks per level in the encoder/decoder
  levels_per_block: int = 2  # number of conv layers in each block
  pose_dim: int = 0  # query pose handed to the decoder, 0 disables it

  # calculated fields
  latent_shape: tuple[int, int, int, int] = field(init=False)
  n_downsamples: int = field(init=False)
  padding: int = field(init=False)
  pre_pool_shape: tuple[int, int, int] = field(init=False)

  # TODO: test out different types of grouping / groupnorm

  def __post_init__(self):
    self.latent_shape = (self.latent_channels, self.cube_size, self.cube_size, self.cube_size)  # shape of latent space [C, D, H, W]
    self.n_downsamples = int(
      min(  # choose min downsampling to reach approximate cube size
        log(self.input_frame_hw[0] // self.cube_size, self.downsample_scale),
        log(self.input_frame_hw[1] // self.cube_size, self.downsample_scale),
      )
    )
    self.padding = (self.kernel_size - 1) // 2  # padding for conv layers to maintain spatial dimensions

    # final state before encoder's AdaptiveAvgPool3d (depth, H, W)
    self.pre_pool_shape = (self.num_frames, self.input_frame_hw[0] // self.downsample_scale**self.n_downsamples,
                           self.input_frame_hw[1] // self.downsample_scale**self.n_downsamples)


@dataclass
class VVAEGeometricConfig:
  """Works in metres rather than pixels: frames are unprojected by their pose into one canonical
  box anchored on the window's middle camera, so the latent's axes are space.

  Volumes are [B, C, X, Y, Z] -- x right, y down, z forward. A Conv3d reads those three as D, H, W,
  which only agrees on the middle one, and x and z share a length so a mix-up stays silent."""

  latent_channels: int = 4
  n_downsamples: int = 2  # latent voxels are this many halvings coarser than the grid's
  act: type[nn.Module] = nn.ReLU

  target: str = "depth"  # rgb, gray_depth or depth; sets how many appearance channels
  predict_unknown: bool = False
  """One more volume channel, [B, C, X, Y, Z] -> [B, C + 1, X, Y, Z], recording whether the model
  believes it saw that region. It trails density, which the renderer reads last, so renders skip
  it."""

  grid_bounds: tuple[tuple[float, float], ...] = ((-2.0, 2.0), (-1.5, 1.5), (0.0, 4.0))  # metres
  grid_resolution: tuple[int, int, int] = (32, 24, 32)  # 12.5 cm cubic voxels at that box
  source_fovy_deg: float = 90.0  # camera the frames were taken with
  depth_max: float = 4.0
  render_hw: tuple[int, int] = (160, 160)  # what the decoder renders a query pose into
  density_bias: float = -6.0  # decoder's initial density, low enough that rays cross the volume

  base_channels: int = 32
  kernel_size: int = 3
  downsample_scale: int = 2
  use_skips: bool = True
  norm: Callable[[int], nn.Module] | None = partial(nn.GroupNorm, 8)
  dropout: float = 0.0
  groups: int = 1
  blocks_per_level: int = 2
  levels_per_block: int = 2

  # calculated fields
  in_channels: int = field(init=False)  # rgb plus density, what the lift produces
  density_index: int = field(init=False)  # where density sits, since unknown trails it
  volume_channels: int = field(init=False)  # appearance plus density, what the decoder produces
  render_channels: int = field(init=False)
  latent_shape: tuple[int, int, int, int] = field(init=False)
  latent_grid: tuple[int, int, int] = field(init=False)
  padding: int = field(init=False)
  bounds: torch.Tensor = field(init=False)

  def __post_init__(self):
    self.in_channels = 4
    self.density_index = -2 if self.predict_unknown else -1
    appearance = {"rgb": 3, "gray_depth": 1, "depth": 0}[self.target]
    self.volume_channels = appearance + 1 + self.predict_unknown
    self.render_channels = appearance + (self.target != "rgb")
    self.bounds = torch.tensor(self.grid_bounds, dtype=torch.float32).T  # [2, 3], lower then upper
    self.padding = (self.kernel_size - 1) // 2

    shrink = self.downsample_scale**self.n_downsamples
    if any(size % shrink for size in self.grid_resolution):
      raise ValueError(f"grid {self.grid_resolution} is not divisible by {shrink}, so the latent "
                       "would not be a whole number of voxels on every axis")
    self.latent_grid = tuple(size // shrink for size in self.grid_resolution)
    self.latent_shape = (self.latent_channels, *self.latent_grid)


@dataclass
class VVAEOutput:
  """Outputs of the pure visual variational autoencoder."""

  z_hat: torch.Tensor  # sampled latent
  mu: torch.Tensor  # mean
  logvar: torch.Tensor  # variance
  x_hat: torch.Tensor  # reconstructed frames
  volume: torch.Tensor | None = None  # the decoded grid, kept so a volume loss need not redecode


class VVAEBlock(nn.Module):
  """A single block of the Visual Variational Autoencoder."""

  def __init__(self, channel: int, config: VVAEConfig | VVAEGeometricConfig) -> None:
    super().__init__()
    self.config = config
    self.final_act = config.act()

    self.levels = nn.ModuleList()
    for i in range(config.levels_per_block):
      layers = [
        nn.Conv3d(channel, channel, config.kernel_size, padding=config.padding,
                  bias=config.norm is None, groups=config.groups),
        config.norm(channel) if config.norm else nn.Identity(),
        self.config.act(),
      ]
      if i < self.config.levels_per_block // 2:
        layers.append(nn.Dropout3d(p=config.dropout))

      self.levels.append(nn.Sequential(*layers))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.config.use_skips:
      x0 = x
    for layer in self.levels:
      x = layer(x)
    if self.config.use_skips:
      x = x + x0  # pyright: ignore[reportPossiblyUnboundVariable]
    return self.final_act(x)


class VVAE(nn.Module):
  """A Visual Variational Autoencoder."""

  def __init__(self, config: VVAEConfig) -> None:
    super().__init__()

    self.encoder = VVAEEncoder(config)
    self.decoder = VVAEDecoder(config)

  def forward(self, x: torch.Tensor) -> VVAEOutput:
    mu, logvar = self.encoder(x)
    z_hat = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # sampled latent
    x_hat = self.decoder(z_hat)  # decoded output from sampled latent

    return VVAEOutput(z_hat=z_hat, mu=mu, logvar=logvar, x_hat=x_hat)


class VVAEEncoder(nn.Module):
  """Converts a 2d set of frames into a 3d latent."""

  def __init__(self, config: VVAEConfig) -> None:
    super().__init__()
    self.config = config

    self.conv = nn.Conv3d(config.in_channels, config.base_channels, config.kernel_size,
                          padding=config.padding, bias=config.norm is None)
    self.norm = config.norm(config.base_channels) if config.norm else nn.Identity()

    channels = [config.base_channels * (2**i) for i in range(config.n_downsamples + 1)]

    self.levels = nn.ModuleList(
      [ nn.Sequential(*[VVAEBlock(ch, config) for _ in range(config.blocks_per_level)]) for ch in channels])
    
    self.transitions = nn.ModuleList(
      [nn.Conv3d(channels[i], channels[i + 1], 1, bias=config.norm is None)for i in range(len(channels) - 1)])

    self.final_pool = nn.AdaptiveAvgPool3d(output_size=config.latent_shape[1:])
    self.channel_proj = nn.Conv3d(channels[-1], 2 * config.latent_channels, 1)

  def forward(self, x):
    """[B, C, T, H, W] -- channels are greyscale plus depth, T is the frame count."""

    x = self.config.act()(self.norm(self.conv(x)))
    for i in range(len(self.levels)):
      if (i > 0):  # shrink H, W only, number of frames starts near latent's cube_size already
        x = F.avg_pool3d(x, (1, self.config.downsample_scale, self.config.downsample_scale))
        x = self.transitions[i - 1](x)
      x = self.levels[i](x)
    x = self.final_pool(x)
    x = self.channel_proj(x)
    mean, logvar = x.chunk(2, dim=1)  # mean and log variance
    return mean, logvar

class VVAEDecoder(nn.Module):
  """Converts a 3d latent back into a set of frames."""

  def __init__(self, config: VVAEConfig) -> None:
    super().__init__()
    self.config = config

    channels = [config.base_channels * (2**i) for i in range(config.n_downsamples + 1)][::-1]  # reversed from encoder
    self.channel_proj = nn.Conv3d(config.latent_channels, channels[0], 1)  # for vae (mu, logvar)

    self.levels = nn.ModuleList(
      [ nn.Sequential(*[VVAEBlock(ch, config) for _ in range(config.blocks_per_level)]) for ch in channels])

    self.transitions = nn.ModuleList(
      [nn.Conv3d(channels[i], channels[i + 1], 1, bias=config.norm is None)for i in range(len(channels) - 1)])

    self.final_conv = nn.Conv3d(config.base_channels, config.in_channels, config.kernel_size,
                                padding=config.padding)  # mirror the encoder's first conv's input channel count

  def forward(self, z):
    x = self.channel_proj(z)
    x = F.interpolate(x, size=self.config.pre_pool_shape, mode="trilinear", align_corners=False)
    for i in range(len(self.levels)):
      x = self.levels[i](x)
      if i < len(self.levels) - 1:  # not last level
        x = F.interpolate(
          x, scale_factor=(1, self.config.downsample_scale, self.config.downsample_scale),
          mode="trilinear", align_corners=False,
        )
        x = self.transitions[i](x)
    return self.final_conv(x)


class VVAEGeometricEncoder(nn.Module):
  """Posed frames in, latent out. The frames are unprojected into one canonical volume before any
  convolution, so the ladder runs over metres and pooling is symmetric across all three axes."""

  def __init__(self, config: VVAEGeometricConfig) -> None:
    super().__init__()
    self.config = config
    self.conv = nn.Conv3d(config.in_channels, config.base_channels, config.kernel_size,
                          padding=config.padding, bias=config.norm is None)
    self.norm = config.norm(config.base_channels) if config.norm else nn.Identity()

    channels = [config.base_channels * (2**i) for i in range(config.n_downsamples + 1)]
    self.levels = nn.ModuleList(
      [nn.Sequential(*[VVAEBlock(ch, config) for _ in range(config.blocks_per_level)]) for ch in channels])
    self.transitions = nn.ModuleList(
      [nn.Conv3d(channels[i], channels[i + 1], 1, bias=config.norm is None) for i in range(len(channels) - 1)])

    self.channel_proj = nn.Conv3d(channels[-1], 2 * config.latent_channels, 1)

  def lift(self, frames: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """[B, 4, T, H, W] rgb+depth and [B, T, 7] relative poses -> [B, 4, X, Y, Z] canonical grids."""
    return torch.stack([
      lift_window(f, p, self.config.bounds.to(f), self.config.grid_resolution, self.config.source_fovy_deg,
                  self.config.depth_max)
      for f, p in zip(frames, poses, strict=True)
    ])

  def carve(self, frames: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    """[B, 4, T, H, W] and [B, T, 7] -> [B, 2, X, Y, Z] of (occupied, observed). Not part of the
    forward pass: it is supervision, saying which parts of the box any camera actually saw."""
    return torch.stack([
      carve_window(f, p, self.config.bounds.to(f), self.config.grid_resolution,
                   self.config.source_fovy_deg, self.config.depth_max)
      for f, p in zip(frames, poses, strict=True)
    ])

  def forward(self, frames: torch.Tensor, poses: torch.Tensor):
    x = self.config.act()(self.norm(self.conv(self.lift(frames, poses))))
    for i, blocks in enumerate(self.levels):
      if i > 0:  # every axis is space now, so this shrinks symmetrically
        x = self.transitions[i - 1](F.avg_pool3d(x, self.config.downsample_scale))
      x = blocks(x)
    return self.channel_proj(x).chunk(2, dim=1)


class VVAEGeometricDecoder(nn.Module):
  """Latent out to a canonical volume, then rendered from whatever pose it is asked for. Being told
  the viewpoint is what stops the latent storing a view: it has to hold the scene instead."""

  def __init__(self, config: VVAEGeometricConfig) -> None:
    super().__init__()
    self.config = config
    channels = [config.base_channels * (2**i) for i in range(config.n_downsamples + 1)][::-1]
    self.channel_proj = nn.Conv3d(config.latent_channels, channels[0], 1)
    self.levels = nn.ModuleList(
      [nn.Sequential(*[VVAEBlock(ch, config) for _ in range(config.blocks_per_level)]) for ch in channels])
    self.transitions = nn.ModuleList(
      [nn.Conv3d(channels[i], channels[i + 1], 1, bias=config.norm is None) for i in range(len(channels) - 1)])
    self.final_conv = nn.Conv3d(channels[-1], config.volume_channels, config.kernel_size, padding=config.padding)
    bias = torch.zeros(config.volume_channels)
    bias[-1] = config.density_bias  # start nearly transparent
    self.final_conv.bias = nn.Parameter(bias)

  def volume(self, z: torch.Tensor) -> torch.Tensor:
    """[B, C, *latent_grid] latent -> [B, volume_channels, X, Y, Z] in [0, 1], appearance then
    density then, with predict_unknown, observedness. Density sits at config.density_index."""
    x = self.channel_proj(z)
    for i, blocks in enumerate(self.levels):
      x = blocks(x)
      if i < len(self.levels) - 1:
        x = F.interpolate(x, scale_factor=self.config.downsample_scale, mode="trilinear",
                          align_corners=False)
        x = self.transitions[i](x)
    return self.final_conv(x).sigmoid()  # sigmoid not clamp, so gradients survive saturation

  def forward(self, z: torch.Tensor, pose: torch.Tensor):
    """Renders each latent from its query pose, returning the [B, render_channels, H, W] image and
    the decoded grid it came from."""
    config = self.config
    volume = self.volume(z)
    renderable = volume[:, :-1] if config.predict_unknown else volume  # density has to be last
    return torch.stack([
      render_view(v, p, config.bounds.to(v), config.render_hw, config.source_fovy_deg,
                  depth_max=config.depth_max, include_depth=config.target != "rgb")
      for v, p in zip(renderable, pose, strict=True)
    ]), volume


class VVAEGeometric(nn.Module):
  """Encodes posed frames into a scene latent and renders it back from a query pose."""

  def __init__(self, config: VVAEGeometricConfig) -> None:
    super().__init__()
    self.config = config
    self.encoder = VVAEGeometricEncoder(config)
    self.decoder = VVAEGeometricDecoder(config)

  @property
  def camera(self) -> tuple[tuple[int, int], float, float]:
    """Matches the dataset property of the same name, so a mismatch can be caught before training
    rather than showing up as a model that will not converge."""
    return self.config.render_hw, self.config.source_fovy_deg, self.config.depth_max

  def forward(self, frames: torch.Tensor, poses: torch.Tensor, query: torch.Tensor) -> VVAEOutput:
    mu, logvar = self.encoder(frames, poses)
    z_hat = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    x_hat, volume = self.decoder(z_hat, query)
    return VVAEOutput(z_hat=z_hat, mu=mu, logvar=logvar, x_hat=x_hat, volume=volume)
