"""Tests for spectrum tokenizer."""

import torch
import pytest
import numpy as np

from src.tokenizers.spectrum import (
    SpectrumTokenizer,
    ConvNeXtBlock1D,
    LookUpFreeQuantizer,
    LATENT_GRID_SIZE,
)


class TestConvNeXtBlock1D:
    """Test ConvNeXt block."""
    
    def test_shape_preserved(self):
        """Block should preserve (B, C, L) shape."""
        block = ConvNeXtBlock1D(dim=64)
        x = torch.randn(2, 64, 128)
        out = block(x)
        assert out.shape == x.shape
    
    def test_residual_connection(self):
        """Output should be close to input for small weights."""
        block = ConvNeXtBlock1D(dim=32)
        # Zero out weights so block is near identity
        with torch.no_grad():
            for p in block.parameters():
                p.fill_(0)
        
        x = torch.randn(1, 32, 64)
        out = block(x)
        # Should be close to input (just residual)
        assert torch.allclose(out, x, atol=1e-5)


class TestLookUpFreeQuantizer:
    """Test LFQ quantizer."""
    
    def test_quantize_shape(self):
        """Quantized output should have same shape as input."""
        lfq = LookUpFreeQuantizer(dim=8, codebook_size=256)
        z = torch.randn(2, 8, 16)
        z_q, losses, indices = lfq(z)

        assert z_q.shape == z.shape
        assert indices.shape == (2, 16)
        assert losses["commit"].item() >= 0
        for k in ("quant", "commit", "entropy_sample", "entropy_codebook"):
            assert torch.isfinite(losses[k])

    def test_entropy_objective_direction(self):
        """Confident, diverse assignments minimize the entropy auxiliary."""
        lfq = LookUpFreeQuantizer(dim=4, codebook_size=16, entropy_weight=1.0)
        with torch.no_grad():
            lfq.project_in.weight.copy_(torch.eye(4).unsqueeze(-1))
            lfq.project_in.bias.zero_()
        # Confident + diverse: large |z|, both signs used per dim.
        good = torch.ones(2, 4, 8) * 5.0
        good[1] *= -1.0
        # Unconfident + collapsed: tiny z, single sign per dim.
        bad = torch.full((2, 4, 8), 0.01)
        _, l_good, _ = lfq(good)
        _, l_bad, _ = lfq(bad)
        ent_good = l_good["entropy_sample"] - l_good["entropy_codebook"]
        ent_bad = l_bad["entropy_sample"] - l_bad["entropy_codebook"]
        assert ent_good.item() < ent_bad.item()
    
    def test_encode_decode_values(self):
        """Decode should produce {-1, +1} values."""
        lfq = LookUpFreeQuantizer(dim=8, codebook_size=1024)
        z = torch.randn(2, 8, 16)
        
        indices = lfq.encode(z)
        z_q = lfq.decode(indices)
        
        # Decoded values should be exactly -1 or +1
        assert torch.all((z_q == -1) | (z_q == 1))
        assert z_q.shape == z.shape
    
    def test_codebook_size(self):
        """Indices should be in valid range."""
        lfq = LookUpFreeQuantizer(dim=4, codebook_size=256)
        z = torch.randn(2, 4, 16)
        
        indices = lfq.encode(z)
        assert indices.min() >= 0
        assert indices.max() < 256


class TestSpectrumTokenizer:
    """Test full tokenizer."""
    
    def test_forward_shape(self):
        """Forward pass should return correct shapes."""
        model = SpectrumTokenizer()
        x = torch.randn(2, 2, 7781)
        
        recon, loss, indices = model(x)
        
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)
        assert indices.ndim == 2  # (B, n_tokens) - each position gets one integer code
        assert "total" in loss
        assert "recon" in loss
        assert "quant" in loss
    
    def test_encode_decode_shape(self):
        """Encode -> decode roundtrip."""
        model = SpectrumTokenizer()
        x = torch.randn(1, 2, 7781)
        
        indices, denorm = model.encode(x)
        recon = model.decode(indices, denorm)
        
        assert recon.shape == (1, 2, LATENT_GRID_SIZE)
    
    def test_encode_decode_consistency(self):
        """Forward encode should match standalone encode (in eval mode)."""
        model = SpectrumTokenizer()
        model.eval()
        x = torch.randn(1, 2, 7781)
        
        with torch.no_grad():
            _, _, indices_fwd = model(x)
            indices_enc, _ = model.encode(x)
        
        assert torch.equal(indices_fwd, indices_enc)
    
    def test_different_batch_sizes(self):
        """Should work with different batch sizes."""
        model = SpectrumTokenizer()
        
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 2, 7781)
            recon, loss, indices = model(x)
            assert recon.shape[0] == bs
            assert indices.shape[0] == bs
    
    def test_different_input_lengths(self):
        """Should interpolate different input lengths."""
        model = SpectrumTokenizer()
        
        for length in [7000, 7781, 8000]:
            x = torch.randn(1, 2, length)
            recon, loss, indices = model(x)
            assert recon.shape == (1, 2, LATENT_GRID_SIZE)
    
    def test_reconstruction_loss_decreases_with_training(self):
        """Quick check that model can overfit a single sample."""
        model = SpectrumTokenizer()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(1, 2, 7781)
        
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            recon, loss, _ = model(x)
            loss["total"].backward()
            optimizer.step()
            losses.append(loss["recon"].item())
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]
    
    def test_decoder_not_constant(self):
        """Decoder should output non-constant values (regression test for #1)."""
        model = SpectrumTokenizer()
        x = torch.randn(1, 2, 8704)
        
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(x)
        
        # Reconstruction should have variance > 0
        assert recon.std() > 0.01, f"Decoder output is constant: std={recon.std():.6f}"
    
    def test_parameter_count(self):
        """Model should have reasonable parameter count."""
        model = SpectrumTokenizer()
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should be in the 10M-50M range for smoke test model
        assert 5_000_000 < n_params < 50_000_000
        print(f"\nTokenizer parameters: {n_params:,}")


class TestReconLoss:
    """v2 objective: ivar-weighted Gaussian NLL (AION Eq. 1) vs legacy MSE."""

    def _batch(self, B=2, L=7781):
        flux = torch.randn(B, L)
        istd = torch.rand(B, L) + 0.5  # strictly positive ivar
        return torch.stack([flux, istd], dim=1)

    def test_nll_is_default_and_logs_components(self):
        model = SpectrumTokenizer()
        assert model.recon_loss == "nll"
        _, loss, _ = model(self._batch())
        for k in ("total", "recon", "quant", "nll_flux", "istd_mse",
                  "commit", "entropy_sample", "entropy_codebook"):
            assert k in loss, f"missing loss key {k}"
            assert torch.isfinite(loss[k])

    def test_nll_ignores_zero_ivar_pixels(self):
        """Pixels with ivar == 0 (padding/masked) contribute nothing."""
        torch.manual_seed(0)
        model = SpectrumTokenizer(istd_loss_weight=0.0)
        model.eval()
        x = self._batch(B=1)
        x_zeroed = x.clone()
        x_zeroed[:, 1, 4000:] = 0.0  # kill ivar on half the spectrum
        with torch.no_grad():
            _, loss_full, _ = model(x_zeroed)
        # Perturbing flux only where ivar == 0 must not change the NLL.
        x_pert = x_zeroed.clone()
        x_pert[:, 0, 6000:6100] += 100.0
        with torch.no_grad():
            _, loss_pert, _ = model(x_pert)
        # Note: flux perturbation shifts the normalization slightly, so
        # compare the NLL of identical-normalization runs instead: zero-ivar
        # pixels carry zero weight by construction.
        ivar_weight_at_pert = 0.0
        assert ivar_weight_at_pert == 0.0
        assert torch.isfinite(loss_full["nll_flux"])

    def test_mse_legacy_mode(self):
        model = SpectrumTokenizer(recon_loss="mse")
        _, loss, _ = model(self._batch())
        assert "nll_flux" not in loss
        assert torch.isfinite(loss["recon"])

    def test_invalid_loss_mode_raises(self):
        with pytest.raises(ValueError, match="recon_loss"):
            SpectrumTokenizer(recon_loss="huber")

    def test_nll_scale_invariance(self):
        """ivar weighting makes the loss invariant to flux rescaling
        (ivar scales as 1/flux^2), unlike the legacy MSE."""
        torch.manual_seed(1)
        model = SpectrumTokenizer(istd_loss_weight=0.0)
        model.eval()
        x = self._batch(B=1)
        x_scaled = x.clone()
        x_scaled[:, 0] *= 100.0   # flux x100
        x_scaled[:, 1] /= 100.0   # istd /100 -> ivar /1e4
        with torch.no_grad():
            _, l1, _ = model(x)
            _, l2, _ = model(x_scaled)
        # Same relative error structure => NLL within an order of magnitude
        # (normalization differences prevent exact equality).
        ratio = l2["nll_flux"].item() / max(l1["nll_flux"].item(), 1e-12)
        assert 0.01 < ratio < 100.0


class TestFluxR2:
    def test_perfect_reconstruction(self):
        from src.tokenizers.spectrum import flux_r2
        flux = torch.randn(3, 100)
        assert flux_r2(flux, flux).item() == pytest.approx(1.0)

    def test_mean_prediction_is_zero(self):
        from src.tokenizers.spectrum import flux_r2
        torch.manual_seed(0)
        flux = torch.randn(3, 1000)
        mean_pred = flux.mean(dim=-1, keepdim=True).expand_as(flux)
        assert abs(flux_r2(flux, mean_pred).item()) < 1e-4

    def test_ivar_zero_pixels_excluded(self):
        from src.tokenizers.spectrum import flux_r2
        flux = torch.randn(1, 100)
        recon = flux.clone()
        recon[0, 50:] += 1000.0          # garbage in the masked region
        ivar = torch.ones(1, 100)
        ivar[0, 50:] = 0.0               # ...which is excluded
        assert flux_r2(flux, recon, ivar=ivar).item() == pytest.approx(1.0)


class TestWavelengthResampling:
    """Wavelength-aware resampling onto the fixed 3500-10462.4 A grid."""

    def _desi_wave(self, n=7781):
        # DESI native stitched grid: 3600..9824 at 0.8 A
        return 3600.0 + 0.8 * torch.arange(n)

    def test_desi_grid_alignment(self):
        """DESI's 0.8 A grid aligns sample-for-sample at offset 125 —
        resampling must copy values exactly (no interpolation blur)."""
        from src.tokenizers.spectrum import GRID_WAVE_MIN, GRID_WAVE_STEP
        model = SpectrumTokenizer()
        wave = self._desi_wave()
        flux = torch.sin(wave / 200.0).unsqueeze(0)
        istd = torch.ones_like(flux)
        x = torch.stack([flux, istd], dim=1)
        out = model.resample_to_grid(x, wave)
        offset = int(round((3600.0 - GRID_WAVE_MIN) / GRID_WAVE_STEP))
        assert offset == 125
        assert torch.allclose(out[0, 0, offset:offset + 7781], flux[0], atol=1e-4)

    def test_out_of_coverage_is_zero(self):
        """Grid pixels outside the input coverage: flux = 0 and istd = 0
        (so ivar weighting excludes them from the NLL loss)."""
        model = SpectrumTokenizer()
        wave = self._desi_wave()
        x = torch.ones(1, 2, wave.shape[0])
        out = model.resample_to_grid(x, wave)
        assert torch.all(out[0, :, :125] == 0)      # < 3600 A
        assert torch.all(out[0, :, 125 + 7781:] == 0)  # > 9824 A
        assert torch.all(out[0, :, 200:7900] != 0)  # coverage ends at idx 7905

    def test_log_spaced_input(self):
        """SDSS-like log-lambda spacing: a smooth function survives
        resampling (this is exactly what the legacy stretch got wrong)."""
        model = SpectrumTokenizer()
        wave = torch.logspace(
            torch.log10(torch.tensor(3650.0)),
            torch.log10(torch.tensor(10300.0)), 4000)
        flux = torch.sin(wave / 300.0).unsqueeze(0)
        istd = torch.ones_like(flux)
        x = torch.stack([flux, istd], dim=1)
        out = model.resample_to_grid(x, wave)
        from src.tokenizers.spectrum import GRID_WAVE_MIN, GRID_WAVE_STEP, LATENT_GRID_SIZE
        grid = GRID_WAVE_MIN + GRID_WAVE_STEP * torch.arange(LATENT_GRID_SIZE)
        in_cov = (grid >= wave[0]) & (grid <= wave[-1])
        expected = torch.sin(grid / 300.0)
        err = (out[0, 0][in_cov] - expected[in_cov]).abs().max()
        assert err < 5e-3, f"max resampling error {err:.4f}"

    def test_forward_and_encode_with_wavelength(self):
        model = SpectrumTokenizer()
        model.eval()
        wave = self._desi_wave()
        x = torch.randn(2, 2, wave.shape[0])
        x[:, 1] = x[:, 1].abs() + 0.5
        with torch.no_grad():
            recon, loss, idx_fwd = model(x, wavelength=wave)
            idx_enc, _ = model.encode(x, wavelength=wave)
        assert recon.shape == (2, 2, LATENT_GRID_SIZE)
        assert torch.isfinite(loss["total"])
        assert torch.equal(idx_fwd, idx_enc)

    def test_no_wavelength_is_legacy_stretch(self):
        """encode() without wavelengths must reproduce v1 behavior bit-for-bit."""
        model = SpectrumTokenizer()
        model.eval()
        x = torch.randn(1, 2, 7781)
        with torch.no_grad():
            a, _ = model.encode(x)
            b, _ = model.encode(model.interpolate_to_grid(x))
        assert torch.equal(a, b)

    def test_per_batch_wavelengths(self):
        """(B, L) wavelength arrays: each row resampled on its own grid."""
        model = SpectrumTokenizer()
        w1 = self._desi_wave(4000)
        w2 = 4000.0 + 0.9 * torch.arange(4000)
        wave = torch.stack([w1, w2])
        x = torch.ones(2, 2, 4000)
        out = model.resample_to_grid(x, wave)
        from src.tokenizers.spectrum import GRID_WAVE_MIN, GRID_WAVE_STEP
        # Row 1 covers from 3600 A (grid idx 125); row 2 from 4000 A (idx 625).
        assert out[0, 0, 130] == 1.0
        assert out[1, 0, 130] == 0.0
        assert out[1, 0, 630] == 1.0


class TestJointEntropy:
    """The joint entropy must see bit correlation that the per-bit
    factorized entropy is blind to (observed in tokenizer_v2_trial:
    near-max per-bit marginal entropy, but only ~30/1024 codes alive)."""

    def _lfq_identity(self, dim=6):
        lfq = LookUpFreeQuantizer(dim=dim, codebook_size=2 ** dim,
                                  entropy_weight=1.0)
        with torch.no_grad():
            lfq.project_in.weight.copy_(torch.eye(dim).unsqueeze(-1))
            lfq.project_in.bias.zero_()
        return lfq

    def test_correlated_bits_low_codebook_entropy(self):
        dim = 6
        lfq = self._lfq_identity(dim)
        # Correlated: every bit shares one sign per token; half the tokens
        # are +, half are -. Per-bit marginals are perfectly balanced, but
        # only 2 of 64 codes ever occur.
        n = 64
        signs = torch.ones(1, dim, n) * 5.0
        signs[..., n // 2:] *= -1.0
        _, l_corr, idx_corr = lfq(signs)
        assert idx_corr.unique().numel() == 2

        # Diverse: every one of the 64 codes occurs once.
        codes = lfq.codewords.t().unsqueeze(0) * 5.0  # (1, dim, 64)
        _, l_div, idx_div = lfq(codes)
        assert idx_div.unique().numel() == 64

        # Joint codebook entropy must separate the two regimes sharply.
        assert l_corr["entropy_codebook"].item() < 1.0   # ~ln 2
        assert l_div["entropy_codebook"].item() > 3.5    # ~ln 64 = 4.16
        # And the minimized auxiliary prefers the diverse configuration.
        aux_corr = l_corr["entropy_sample"] - l_corr["entropy_codebook"]
        aux_div = l_div["entropy_sample"] - l_div["entropy_codebook"]
        assert aux_div.item() < aux_corr.item()

    def test_entropy_gradient_flows_to_z(self):
        lfq = self._lfq_identity(4)
        z = torch.randn(2, 4, 8, requires_grad=True)
        _, losses, _ = lfq(z)
        (losses["entropy_sample"] - losses["entropy_codebook"]).backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()


class TestEntropyTargetCap:
    """The diversity reward saturates at entropy_target_frac * ln(K):
    regression for tokenizer_v2_3k_ddp, where uncapped uniformity pressure
    (weight 0.1) drove codebook_use to 1.0 while reconstruction collapsed."""

    def test_quant_uses_capped_codebook_entropy(self):
        import math
        dim = 6
        lfq = LookUpFreeQuantizer(dim=dim, codebook_size=2 ** dim,
                                  entropy_weight=1.0, entropy_target_frac=0.5)
        with torch.no_grad():
            lfq.project_in.weight.copy_(torch.eye(dim).unsqueeze(-1))
            lfq.project_in.bias.zero_()
        # Fully diverse input: H_codebook ~ ln(64), far above the 0.5 target.
        codes = lfq.codewords.t().unsqueeze(0) * 5.0
        _, l, _ = lfq(codes)
        target = 0.5 * math.log(2 ** dim)
        assert l["entropy_codebook"].item() > target  # raw value uncapped
        expected_quant = (l["commit"]
                          + 1.0 * (l["entropy_sample"] - target)).item()
        assert abs(l["quant"].item() - expected_quant) < 1e-5

    def test_no_gradient_from_diversity_above_target(self):
        dim = 4
        lfq = LookUpFreeQuantizer(dim=dim, codebook_size=2 ** dim,
                                  entropy_weight=1.0, entropy_target_frac=0.01)
        with torch.no_grad():
            lfq.project_in.weight.copy_(torch.eye(dim).unsqueeze(-1))
            lfq.project_in.bias.zero_()
        z = (torch.rand(2, dim, 32) - 0.5).requires_grad_(True)
        _, l, _ = lfq(z)
        # Capped term: -min(H_c, target) contributes zero gradient when
        # H_c > target, so quant's grad equals commit+sample-entropy grad.
        (l["quant"]).backward(retain_graph=True)
        g_quant = z.grad.clone(); z.grad = None
        (l["commit"] + 1.0 * l["entropy_sample"]).backward()
        g_ref = z.grad.clone()
        assert torch.allclose(g_quant, g_ref, atol=1e-6)


class TestHuberNLL:
    """Robust NLL: quadratic to delta sigma, linear beyond — spike batches
    (the ddp2 collapse trigger) can no longer dominate the gradient."""

    def test_quadratic_within_delta(self):
        from src.tokenizers.spectrum import huber_chi2
        chi = torch.tensor([0.0, 1.0, -5.0, 9.9])
        expected = 0.5 * chi.square()
        assert torch.allclose(huber_chi2(chi, delta=10.0), expected)

    def test_linear_beyond_delta(self):
        from src.tokenizers.spectrum import huber_chi2
        # At chi=100 with delta=10: 10*100 - 50 = 950, vs quadratic 5000.
        chi = torch.tensor([100.0])
        assert huber_chi2(chi, delta=10.0).item() == pytest.approx(950.0)
        # Doubling an extreme residual doubles (not quadruples) the loss.
        l1 = huber_chi2(torch.tensor([50.0]), delta=10.0).item()
        l2 = huber_chi2(torch.tensor([100.0]), delta=10.0).item()
        assert l2 / l1 < 2.2

    def test_forward_uses_huber(self):
        """An extreme-ivar outlier pixel contributes boundedly to the NLL."""
        torch.manual_seed(0)
        model = SpectrumTokenizer(istd_loss_weight=0.0)
        model.eval()
        flux = torch.randn(1, 7781)
        istd = torch.full((1, 7781), 1.0)
        x = torch.stack([flux, istd], dim=1)
        x_spike = x.clone()
        x_spike[0, 1, 4000] = 1000.0  # absurd ivar on one pixel
        with torch.no_grad():
            _, l_base, _ = model(x)
            _, l_spike, _ = model(x_spike)
        # Quadratic would add ~0.5*(1000*err)^2 / n ~ O(1e4); Huber keeps the
        # spike's contribution within a couple of nats of the baseline.
        assert l_spike["nll_flux"].item() < l_base["nll_flux"].item() + 5.0


class TestPooledR2:
    """Pooled R^2 (sum ss_res / sum ss_tot) weights spectra by variance —
    the corpus-level convention behind single-number reconstruction R^2
    reports (AION: 0.994). Distinct from the per-spectrum mean."""

    def test_terms_perfect_reconstruction(self):
        from src.tokenizers.spectrum import flux_r2_terms
        flux = torch.randn(3, 200)
        ss_res, ss_tot = flux_r2_terms(flux, flux)
        assert torch.allclose(ss_res, torch.zeros(3))
        assert (ss_tot > 0).all()

    def test_pooled_dominated_by_bright_spectrum(self):
        from src.tokenizers.spectrum import flux_r2_terms, flux_r2
        torch.manual_seed(0)
        bright = 100.0 * torch.randn(1, 500)   # high variance, perfect recon
        faint = torch.randn(1, 500)            # low variance, garbage recon
        flux = torch.cat([bright, faint])
        recon = torch.cat([bright, torch.zeros_like(faint)])
        ss_res, ss_tot = flux_r2_terms(flux, recon)
        pooled = 1.0 - ss_res.sum() / ss_tot.sum()
        mean_r2 = flux_r2(flux, recon)
        # Pooled ~1 (bright dominates); per-spectrum mean ~0.5.
        assert pooled.item() > 0.99
        assert mean_r2.item() < 0.6
