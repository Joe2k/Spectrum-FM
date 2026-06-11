"""
Spectrum Tokenizer
==================
ConvNeXt-V2 based autoencoder with LFQ quantization for DESI spectra.

Architecture (based on AION paper):
- Input: 2 channels (flux, ivar/istd) interpolated to fixed 8704-pixel grid
- Encoder: 4-stage ConvNeXt-V2 with progressive downsampling
  - Stem: 4x4 conv, stride 4 → 2176
  - Stage 1: 3 ConvNeXt blocks (dim 96)
  - Stage 2: downsample 2x + 3 blocks (dim 192) → 1088
  - Stage 3: downsample 2x + 9 blocks (dim 384) → 544
  - Stage 4: downsample 2x + 3 blocks (dim 512) → 272
- Latent: 272 tokens (we pad to 273)
- Quantizer: Look-up-Free Quantizer (LFQ) with codebook size 1024
- Decoder: mirror of encoder with upsampling
- Output: reconstructed flux (+ optional mask)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# Fixed latent grid (matches AION paper)
LATENT_GRID_SIZE = 8704
N_TOKENS = 273

# Fixed WAVELENGTH grid the latent grid corresponds to (AION convention):
# 3500.0 .. 10462.4 Angstrom at 0.8 A spacing — exactly 8704 points.
# DESI's native stitched grid (3600–9824 A @ 0.8 A, 7781 px) aligns with
# this grid sample-for-sample at offset (3600-3500)/0.8 = 125.
GRID_WAVE_MIN = 3500.0
GRID_WAVE_STEP = 0.8
GRID_WAVE_MAX = GRID_WAVE_MIN + GRID_WAVE_STEP * (LATENT_GRID_SIZE - 1)  # 10462.4


class LayerNorm1d(nn.Module):
    """LayerNorm for 1D conv features (B, C, L)."""
    
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        # x: (B, C, L)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt V2 block for 1D sequences.
    
    Depthwise conv -> LayerNorm -> pointwise (expand) -> GELU -> pointwise (project) -> residual
    """
    
    def __init__(self, dim, kernel_size=7, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm1d(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv1d(dim, mlp_hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(mlp_hidden_dim, dim, kernel_size=1)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input_x + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class DownsampleBlock(nn.Module):
    """Downsampling block: LayerNorm -> Conv1d(stride=2)."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm1d(in_dim)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    """Upsampling block: ConvTranspose1d(stride=2) -> LayerNorm."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = LayerNorm1d(out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class LookUpFreeQuantizer(nn.Module):
    """Look-up-Free Quantizer (LFQ) - binary quantization.

    Each dimension is quantized to {-1, +1}, giving exactly 2^dim codes.
    With dim=10, codebook_size = 2^10 = 1024.

    Uses straight-through estimator with commitment loss, plus the
    MAGVIT-v2 entropy objective: minimize per-sample assignment entropy
    (confident assignments) while maximizing batch codebook entropy
    (all codes used) — without it, commitment alone invites dead codes.

    The entropy is computed EXACTLY over all 2^dim codes via
    p(code k | z) = softmax(2 z·c_k / temperature). A per-bit factorized
    version was tried first and is blind to bit *correlation*: the
    tokenizer_v2_trial run showed near-ceiling per-bit marginal entropy
    while only ~30/1024 joint codes were alive. At dim=10 the exact
    1024-way distribution costs a (B·L, 1024) softmax — trivial.
    """

    def __init__(self, dim=10, codebook_size=1024, commitment_weight=0.25,
                 entropy_weight=0.02, entropy_temperature=1.0,
                 entropy_target_frac=0.75):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        self.entropy_temperature = entropy_temperature
        # Diversity reward saturates at this fraction of ln(codebook_size):
        # past it, further uniformity earns nothing. Run tokenizer_v2_3k_ddp
        # (weight 0.1, no cap) drove codebook_use to exactly 1.0 while
        # reconstruction collapsed 15x — the uncapped uniformity pressure
        # kept reassigning code meanings until the decoder broke.
        self.entropy_target = entropy_target_frac * math.log(codebook_size)

        self.project_in = nn.Conv1d(dim, dim, kernel_size=1)

        # All 2^dim codewords in {-1,+1}^dim. Bit order matches the index
        # encoding below (bit i = 2^i). Non-persistent: not in state_dict,
        # so existing (v1) checkpoints load unchanged.
        idx = torch.arange(codebook_size)
        bits = ((idx.unsqueeze(1) >> torch.arange(dim)) & 1) * 2 - 1
        self.register_buffer("codewords", bits.float(), persistent=False)

    def forward(self, z):
        """Quantize latents.

        Args:
            z: (B, dim, L) continuous latent

        Returns:
            z_q: (B, dim, L) quantized latent
            losses: dict with "quant" (total, backprop this), "commit",
                "entropy_sample", "entropy_codebook"
            indices: (B, L) token indices
        """
        z = self.project_in(z)

        # Binary quantization: sign(z) -> {-1, +1}
        z_q = torch.sign(z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Commitment loss (pull z toward +/-1)
        commit = F.mse_loss(z.float(), torch.sign(z).detach().float()) * self.commitment_weight

        # Exact joint entropy objective over all 2^dim codes (fp32 for AMP
        # stability). For binary codewords, -||z - c_k||^2 = 2 z·c_k + const,
        # so p(code k | z) = softmax(2 z·c_k / tau). Entropies are in nats
        # (max = ln codebook_size, e.g. 6.93 for 1024).
        zf = z.float()
        logits = torch.einsum("bdl,kd->bkl", zf, self.codewords) \
            * (2.0 / self.entropy_temperature)              # (B, K, L)
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ent_sample = -(p * logp).sum(dim=1).mean()
        p_bar = p.mean(dim=(0, 2))                           # (K,) batch marginal
        ent_codebook = -(p_bar * p_bar.clamp_min(1e-12).log()).sum()
        # Saturating diversity reward: -min(H_codebook, target). Once the
        # marginal entropy reaches the target the gradient is zero — the
        # term fights collapse but cannot keep scrambling code assignments
        # toward perfect uniformity at reconstruction's expense.
        entropy_aux = ent_sample - ent_codebook.clamp(max=self.entropy_target)

        losses = {
            "quant": commit + self.entropy_weight * entropy_aux,
            "commit": commit,
            "entropy_sample": ent_sample,
            "entropy_codebook": ent_codebook,
        }

        # Compute indices: binary to integer
        bits = ((z_q + 1) / 2).long()  # (B, dim, L) with values {0, 1}
        powers = torch.arange(self.dim, device=z.device).view(1, -1, 1)
        indices = (bits * (2 ** powers)).sum(dim=1)  # (B, L)

        return z_q, losses, indices

    def encode(self, z):
        """Encode to discrete indices."""
        z = self.project_in(z)
        z_q = torch.sign(z)

        bits = ((z_q + 1) / 2).long()
        powers = torch.arange(self.dim, device=z.device).view(1, -1, 1)
        indices = (bits * (2 ** powers)).sum(dim=1)
        return indices

    def decode(self, indices):
        """Decode from discrete indices."""
        # Convert integer to binary
        B, L = indices.shape
        z = torch.zeros(B, self.dim, L, device=indices.device)

        for i in range(self.dim):
            z[:, i, :] = ((indices // (2 ** i)) % 2).float() * 2 - 1

        return z



class SpectrumTokenizer(nn.Module):
    """Spectrum tokenizer: ConvNeXt-V2 autoencoder + LFQ quantization.
    
    Interpolates input to fixed 8704-pixel grid before encoding.
    
    Args:
        in_channels: Input channels (2 for flux+ivar)
        latent_channels: Latent dimension before quantization (default: 512)
        embedding_dim: Quantization dimension (default: 10)
        codebook_size: Number of discrete codes (default: 1024)
        encoder_depths: Number of ConvNeXt blocks per stage
        encoder_dims: Channel dimensions per stage
        decoder_depths: Number of ConvNeXt blocks per stage
        decoder_dims: Channel dimensions per stage
        recon_loss: "nll" (default) = inverse-variance-weighted Gaussian
            NLL on the flux channel (AION Eq. 1) + a small normalized-space
            MSE on the istd channel. "mse" = legacy plain MSE on the
            denormalized 2-channel output (v1 behavior; bright spectra and
            noisy pixels dominate — kept for A/B comparison only).
        entropy_weight: weight of the LFQ entropy objective (0 disables).
            The codebook-diversity reward saturates at 90% of ln(K) (see
            LookUpFreeQuantizer.entropy_target), so this only needs to be
            large enough to escape collapse, not to dominate.
        istd_loss_weight: weight of the istd-channel auxiliary MSE in
            "nll" mode (keeps decode()'s channel 1 meaningful).
    """

    def __init__(
        self,
        in_channels=2,
        latent_channels=512,
        embedding_dim=10,
        codebook_size=1024,
        encoder_depths=(3, 3, 9, 3),
        encoder_dims=(96, 192, 384, 512),
        decoder_depths=(3, 3, 9, 3),
        decoder_dims=(384, 192, 96, 96),
        commitment_weight=0.25,
        recon_loss="nll",
        entropy_weight=0.02,
        entropy_target_frac=0.75,
        istd_loss_weight=0.1,
    ):
        super().__init__()

        if recon_loss not in ("nll", "mse"):
            raise ValueError(f"recon_loss must be 'nll' or 'mse', got {recon_loss!r}")
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.recon_loss = recon_loss
        self.istd_loss_weight = istd_loss_weight
        
        # === ENCODER ===
        self.encoder_stem = nn.Sequential(
            nn.Conv1d(in_channels, encoder_dims[0], kernel_size=4, stride=4, padding=0),
            LayerNorm1d(encoder_dims[0]),
        )
        
        self.encoder_stages = nn.ModuleList()
        for i in range(len(encoder_depths)):
            stage = nn.ModuleList()
            # Downsampling (except first stage which uses stem)
            if i > 0:
                stage.append(DownsampleBlock(encoder_dims[i-1], encoder_dims[i]))
            # ConvNeXt blocks
            for _ in range(encoder_depths[i]):
                stage.append(ConvNeXtBlock1D(encoder_dims[i]))
            self.encoder_stages.append(stage)
        
        # Pre-quantization projection
        self.pre_quant_norm = LayerNorm1d(encoder_dims[-1])
        self.quant_conv = nn.Conv1d(encoder_dims[-1], embedding_dim, kernel_size=1)
        
        # Quantizer
        self.quantizer = LookUpFreeQuantizer(
            dim=embedding_dim,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            entropy_weight=entropy_weight,
            entropy_target_frac=entropy_target_frac,
        )
        
        # Post-quantization projection
        self.post_quant_conv = nn.Conv1d(embedding_dim, decoder_dims[0], kernel_size=1)
        
        # === DECODER ===
        self.decoder_stages = nn.ModuleList()
        for i in range(len(decoder_depths)):
            stage = nn.ModuleList()
            # ConvNeXt blocks
            for _ in range(decoder_depths[i]):
                stage.append(ConvNeXtBlock1D(decoder_dims[i]))
            # Upsampling (except last stage)
            if i < len(decoder_depths) - 1:
                stage.append(UpsampleBlock(decoder_dims[i], decoder_dims[i+1]))
            self.decoder_stages.append(stage)
        
        # Output head: upsample by 4 to match stem stride
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose1d(decoder_dims[-1], decoder_dims[-1], kernel_size=4, stride=4),
            LayerNorm1d(decoder_dims[-1]),
            nn.Conv1d(decoder_dims[-1], in_channels, kernel_size=1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def interpolate_to_grid(self, x, target_length=LATENT_GRID_SIZE):
        """LEGACY length stretch: interpolate to a fixed number of samples,
        ignoring the wavelength solution. Self-consistent for DESI-only
        training (a fixed monotone remap of its 7781-px grid) and what
        spectrum_tokenizer_v1 was trained with — kept as the default so v1
        checkpoints keep working. Physically wrong for any other survey
        (e.g. SDSS's log-lambda spacing); use `resample_to_grid` instead.
        """
        if x.shape[-1] != target_length:
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        return x

    def resample_to_grid(self, x, wavelength):
        """Wavelength-aware resampling onto the fixed grid (AION convention):
        3500–10462.4 A at 0.8 A. Linear interpolation in wavelength; grid
        points outside the input's coverage get flux = 0 AND istd = 0, so
        ivar weighting (the 'nll' loss) excludes them automatically.

        Args:
            x: (B, 2, L) [flux, istd] on the native wavelength grid
            wavelength: (L,) or (B, L) ascending wavelengths in Angstrom

        Returns:
            (B, 2, LATENT_GRID_SIZE) resampled spectrum
        """
        B, C, L = x.shape
        device = x.device
        grid = GRID_WAVE_MIN + GRID_WAVE_STEP * torch.arange(
            LATENT_GRID_SIZE, device=device, dtype=torch.float32)  # (G,)
        if wavelength.dim() == 1:
            wavelength = wavelength.unsqueeze(0).expand(B, -1)
        wave = wavelength.to(device=device, dtype=torch.float32).contiguous()

        # Bracketing input indices for every grid point, per batch row.
        gridB = grid.unsqueeze(0).expand(B, -1).contiguous()
        idx_hi = torch.searchsorted(wave, gridB).clamp(1, L - 1)  # (B, G)
        idx_lo = idx_hi - 1
        w_lo = torch.gather(wave, 1, idx_lo)
        w_hi = torch.gather(wave, 1, idx_hi)
        t = ((gridB - w_lo) / (w_hi - w_lo).clamp(min=1e-6)).clamp(0.0, 1.0)

        x_f = x.float()
        lo = torch.gather(x_f, 2, idx_lo.unsqueeze(1).expand(B, C, -1))
        hi = torch.gather(x_f, 2, idx_hi.unsqueeze(1).expand(B, C, -1))
        out = lo + (hi - lo) * t.unsqueeze(1)

        # Zero everything outside the input's wavelength coverage.
        in_cov = (gridB >= wave[:, :1]) & (gridB <= wave[:, -1:])
        return out * in_cov.unsqueeze(1).to(out.dtype)

    def _to_grid(self, x, wavelength=None):
        """Wavelength-aware resample when wavelengths are given, else the
        legacy length stretch (v1-compatible)."""
        if wavelength is not None:
            return self.resample_to_grid(x, wavelength)
        return self.interpolate_to_grid(x)
    
    def encode(self, x, wavelength=None):
        """Encode spectrum to quantized tokens.

        Args:
            x: (B, in_channels, L) input spectrum
            wavelength: optional (L,) or (B, L) wavelengths in Angstrom.
                When given, the spectrum is resampled onto the fixed
                wavelength grid (physically correct, survey-agnostic).
                When None, the legacy v1 length stretch is used.

        Returns:
            indices: (B, dim, n_tokens) discrete token indices
            denorm: (B,) normalization factor for denormalization
        """
        x = self._to_grid(x, wavelength)
        x_norm, denorm = self.normalize(x)
        x = self.encoder_stem(x_norm)
        
        for stage in self.encoder_stages:
            for block in stage:
                x = block(x)
        
        x = self.pre_quant_norm(x)
        x = self.quant_conv(x)
        
        indices = self.quantizer.encode(x)
        
        return indices, denorm
    
    def decode(self, indices, denorm):
        """Decode tokens to spectrum.
        
        Args:
            indices: (B, dim, n_tokens) discrete token indices
            denorm: (B,) normalization factor
            
        Returns:
            x: (B, in_channels, LATENT_GRID_SIZE) reconstructed spectrum
        """
        x = self.quantizer.decode(indices)
        x = self.post_quant_conv(x)
        
        for stage in self.decoder_stages:
            for block in stage:
                x = block(x)
        
        x = self.decoder_head(x)
        
        # Denormalize
        x = self.denormalize(x, denorm)
        
        return x
    
    def normalize(self, x):
        """Normalize spectrum to zero-mean, unit-variance-like range.
        
        AION-style normalization:
        1. Compute robust median flux (mean of unmasked pixels, clamped)
        2. log10 compress the normalization factor
        3. Normalize flux: (flux / norm - 1) * input_scaling
        4. Normalize istd similarly
        5. Stack and apply arcsinh for additional range compression
        
        Args:
            x: (B, 2, L) where x[:,0] = flux, x[:,1] = istd
            
        Returns:
            x_norm: (B, 2, L) normalized spectrum
            norm_factor: (B,) normalization factor for denormalization
        """
        flux = x[:, 0:1, :]  # (B, 1, L)
        istd = x[:, 1:2, :]  # (B, 1, L)
        
        # Compute robust median (mean of positive flux)
        positive_mask = flux > 0
        norm = (flux * positive_mask.float()).sum(dim=-1) / (positive_mask.sum(dim=-1).float() + 1.0)
        # Denormalization factor. (A previous version round-tripped this
        # through 10^log10(norm+1) - 1, which is exactly `norm` — removed.)
        denorm = torch.clamp(norm, min=0.1)
        
        # Normalize flux and istd
        flux_norm = (flux / denorm.unsqueeze(-1) - 1.0) * 0.2
        istd_norm = (istd / denorm.unsqueeze(-1)) * 0.2
        
        # Stack and apply arcsinh for range compression
        x_norm = torch.arcsinh(torch.cat([flux_norm, istd_norm], dim=1))
        
        return x_norm, denorm
    
    def denormalize(self, x_norm, denorm):
        """Denormalize reconstructed spectrum.
        
        Args:
            x_norm: (B, 2, L) normalized spectrum (after inverse arcsinh)
            denorm: (B,) normalization factor
            
        Returns:
            x: (B, 2, L) denormalized spectrum
        """
        # Inverse arcsinh
        x = torch.sinh(x_norm)
        
        flux_norm = x[:, 0:1, :]
        istd_norm = x[:, 1:2, :]
        
        # Denormalize
        flux = (flux_norm / 0.2 + 1.0) * denorm.unsqueeze(-1)
        istd = (istd_norm / 0.2) * denorm.unsqueeze(-1)
        
        x = torch.cat([flux, istd], dim=1)
        
        return x
    
    def forward(self, x, wavelength=None):
        """Full forward pass: encode + quantize + decode.

        Args:
            x: (B, in_channels, L) input spectrum
            wavelength: optional (L,) or (B, L) wavelengths in Angstrom;
                see `encode`. With the 'nll' loss, out-of-coverage grid
                pixels carry ivar = 0 and are excluded automatically.

        Returns:
            recon: (B, in_channels, LATENT_GRID_SIZE) reconstructed spectrum
            loss: dict with quantization and reconstruction losses
            indices: (B, n_tokens) discrete token indices
        """
        # Map onto the fixed grid
        x_grid = self._to_grid(x, wavelength)
        
        # Normalize
        x_norm, denorm = self.normalize(x_grid)
        
        # Encode
        h = self.encoder_stem(x_norm)
        for stage in self.encoder_stages:
            for block in stage:
                h = block(h)
        
        h = self.pre_quant_norm(h)
        h = self.quant_conv(h)
        
        # Quantize
        h_q, quant_losses, indices = self.quantizer(h)

        # Decode
        h = self.post_quant_conv(h_q)
        for stage in self.decoder_stages:
            for block in stage:
                h = block(h)

        recon_norm = self.decoder_head(h)

        # Denormalize
        recon = self.denormalize(recon_norm, denorm)

        # Reconstruction loss (fp32 for AMP stability).
        loss = {"quant": quant_losses["quant"], **{k: v for k, v in quant_losses.items() if k != "quant"}}
        if self.recon_loss == "nll":
            # Inverse-variance-weighted Gaussian NLL on flux (AION Eq. 1):
            # 0.5 * ivar * (flux - flux_hat)^2, averaged over pixels with
            # ivar > 0. ivar weighting makes the loss scale-invariant
            # (bright spectra don't dominate) and zeroes out padded /
            # fully-masked pixels (collate sets their ivar to 0).
            flux_true = x_grid[:, 0].float()
            istd_true = x_grid[:, 1].float()
            ivar = istd_true.square()
            flux_rec = recon[:, 0].float()
            valid = (ivar > 0).float()
            n_valid = valid.sum().clamp(min=1.0)
            nll = (0.5 * ivar * (flux_true - flux_rec).square() * valid).sum() / n_valid
            # Auxiliary istd-channel MSE in normalized space, so the
            # decoder's channel 1 stays meaningful for decode() callers.
            istd_mse = F.mse_loss(recon_norm[:, 1].float(), x_norm[:, 1].float())
            recon_loss = nll + self.istd_loss_weight * istd_mse
            loss["nll_flux"] = nll
            loss["istd_mse"] = istd_mse
        else:
            # Legacy v1 objective: plain MSE on the denormalized 2-channel
            # output. Scales with brightness^2 and weights noisy pixels
            # equally — kept only for A/B comparison.
            recon_loss = F.mse_loss(recon.float(), x_grid.float())

        loss["recon"] = recon_loss
        loss["total"] = recon_loss + loss["quant"]

        return recon, loss, indices


def flux_r2(flux_true: torch.Tensor, flux_recon: torch.Tensor,
            ivar: torch.Tensor = None, reduce: bool = True) -> torch.Tensor:
    """Mean per-spectrum R^2 of the flux reconstruction.

    The tokenizer-quality gating metric (AION's spectrum tokenizer
    reports R^2 = 0.994). Computed per spectrum over pixels with
    ivar > 0 (all pixels if ivar is None), then averaged over the batch.

    Args:
        flux_true: (B, L) input flux on the model grid
        flux_recon: (B, L) reconstructed flux
        ivar: (B, L) optional inverse variance; pixels with ivar == 0
            (padding / fully masked) are excluded
        reduce: if True (default) return the batch mean; if False return
            the per-spectrum (B,) R^2 values.

    Caveat: R^2 is computed against the NOISY input flux, so for low-SNR
    spectra even a perfect denoised reconstruction caps out near
    signal_var / (signal_var + noise_var). Slice by SNR (or use the
    ivar-weighted NLL) when judging tokenizer quality on faint targets.

    Returns:
        scalar tensor (reduce=True) or (B,) tensor (reduce=False)
    """
    flux_true = flux_true.float()
    flux_recon = flux_recon.float()
    if ivar is None:
        valid = torch.ones_like(flux_true)
    else:
        valid = (ivar > 0).float()
    n = valid.sum(dim=-1).clamp(min=1.0)
    mean = (flux_true * valid).sum(dim=-1, keepdim=True) / n.unsqueeze(-1)
    ss_res = ((flux_true - flux_recon).square() * valid).sum(dim=-1)
    ss_tot = ((flux_true - mean).square() * valid).sum(dim=-1).clamp(min=1e-12)
    r2 = 1.0 - ss_res / ss_tot
    return r2.mean() if reduce else r2


def test_tokenizer_shapes():
    """Quick shape test."""
    batch_size = 2
    seq_len = 7781  # Our DESI data length
    
    model = SpectrumTokenizer()
    x = torch.randn(batch_size, 2, seq_len)
    
    # Forward
    recon, loss, indices = model(x)
    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {recon.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"N tokens:      {indices.shape[1]}")
    print(f"Losses:        { {k: f'{v.item():.4f}' for k, v in loss.items()} }")
    
    # Encode/decode round-trip
    indices2, denorm = model.encode(x)
    recon2 = model.decode(indices2, denorm)
    print(f"\nEncode -> Decode shape: {recon2.shape}")
    print(f"Max reconstruction diff: {(recon - recon2).abs().max().item():.6f}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    return model


if __name__ == "__main__":
    test_tokenizer_shapes()
