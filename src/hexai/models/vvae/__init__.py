"""The geometric VVAE: posed rgb+depth frames in, a 3D scene latent out, a held-out view rendered
back. Only the encoder ships to the robot; the decoder is training scaffolding.

Agentic info lives in `.agents/` Look at it for info regarding previous experiments or needed context regarding the vvae.

Author: Marco Cassar (@Ocramaru)
"""

from hexai.models.vvae.model import (
  VVAE,
  VVAEConfig,
  VVAEGeometric,
  VVAEGeometricConfig,
  VVAEGeometricDecoder,
  VVAEGeometricEncoder,
  VVAEOutput,
)

__all__ = [
  "VVAE",
  "VVAEConfig",
  "VVAEGeometric",
  "VVAEGeometricConfig",
  "VVAEGeometricDecoder",
  "VVAEGeometricEncoder",
  "VVAEOutput",
]
