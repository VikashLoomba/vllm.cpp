#!/usr/bin/env python3
"""`LTX25-ADHERENCE-DETAIL-LOSS` (#2513): the instrument that decides whether the
render's smoothness CAUSES its adherence gap has to be able to give the wrong
answer, and these cases prove each of its parts can.

WHY A SUITE AT ALL FOR A DIAGNOSIS. This row publishes a NEGATIVE result: the
spectrum says our render is not smoother, the within-render correlation points
the wrong way, and blurring the reference does not reproduce the gap. A negative
result is exactly the shape a broken instrument produces for free. A radial
profile that returns noise, a correlation that returns zero because one input is
constant, and a blur that does not blur all read as "no effect". So each case
below feeds the function a signal whose answer is known in closed form, and
several of them assert the instrument moves in the direction the falsified
hypothesis would have needed. An instrument that cannot produce the positive
answer has not refuted it.

THE DIGEST GUARD IS THE FIRST CASE, not a footnote. This repository has gated
against the wrong checkpoint before, and the frames live on a soft-mounted CIFS
share where a read can truncate. `verify_ours` refuses a set whose concatenated
digest is not the one `ltx25-prompt-adherence.md` records for the frames W3
scored, so a measurement taken on some other render cannot be compared with that
verdict by accident.

Needs numpy and nothing else: no checkpoint, no frames, no network, no GPU. Every
signal is synthesised here.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - the lane installs it
    print("SKIP: numpy is not importable, and every case here computes over arrays")
    raise SystemExit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_cmp = _load("ltx25_render_compare_for_detail",
             os.path.join("scripts", "ltx25-render-compare.py"))
D = _load("ltx25_detail_loss", os.path.join("scripts", "ltx25-adherence-detail-loss.py"))


def write_ppm(path: str, a: np.ndarray) -> None:
    h, w, _ = a.shape
    with open(path, "wb") as fh:
        fh.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        fh.write(np.ascontiguousarray(a.astype(np.uint8)).tobytes())


class IdentityRefusals(unittest.TestCase):
    """The set has to be the set W3 scored, or nothing may be compared to it."""

    def _dir(self, n: int = 3, seed: int = 0):
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(seed)
        for i in range(n):
            write_ppm(os.path.join(d, f"frame_{i:06d}.ppm"),
                      rng.integers(0, 256, (8, 8, 3), dtype=np.uint8))
        return d

    def test_a_wrong_set_is_refused_and_names_the_recorded_digest(self):
        with self.assertRaises(D.UnreadableInput) as cm:
            D.verify_ours(self._dir())
        self.assertIn(D.OURS_SET_DIGEST, str(cm.exception))

    def test_the_recorded_digest_is_the_concatenated_form_and_not_a_hash_of_bytes(self):
        # The guard is only meaningful if it recomputes the SAME reduction the
        # spec published. Build a directory, compute the reduction by hand, and
        # require the checker to accept exactly that value -- so a checker that
        # hashed the frames some other way would fail here rather than silently
        # guard a different quantity.
        d = self._dir(seed=7)
        paths = D.frame_paths(d)
        lines = "".join(f"{D.sha256_file(p)}  {os.path.basename(p)}\n" for p in paths)
        want = hashlib.sha256(lines.encode("ascii")).hexdigest()
        saved = D.OURS_SET_DIGEST
        try:
            D.OURS_SET_DIGEST = want
            self.assertEqual(D.verify_ours(d)["set_digest"], want)
        finally:
            D.OURS_SET_DIGEST = saved

    def test_one_flipped_byte_in_one_frame_is_refused(self):
        d = self._dir(seed=11)
        paths = D.frame_paths(d)
        lines = "".join(f"{D.sha256_file(p)}  {os.path.basename(p)}\n" for p in paths)
        saved = D.OURS_SET_DIGEST
        try:
            D.OURS_SET_DIGEST = hashlib.sha256(lines.encode("ascii")).hexdigest()
            self.assertTrue(D.verify_ours(d)["verified"])
            with open(paths[-1], "r+b") as fh:
                fh.seek(-1, os.SEEK_END)
                b = fh.read(1)
                fh.seek(-1, os.SEEK_END)
                fh.write(bytes([b[0] ^ 0x01]))
            with self.assertRaises(D.UnreadableInput):
                D.verify_ours(d)
        finally:
            D.OURS_SET_DIGEST = saved

    def test_a_reference_frame_whose_digest_is_absent_is_unidentified(self):
        d = self._dir(n=2, seed=3)
        sums = os.path.join(d, "SUMS")
        with open(sums, "w", encoding="utf-8") as fh:
            fh.write("# only one of the two\n")
            p = D.frame_paths(d)[0]
            fh.write(f"{D.sha256_file(p)}  {os.path.basename(p)}\n")
        with self.assertRaises(D.UnreadableInput):
            D.verify_reference(d, sums)


class ToneFit(unittest.TestCase):
    """A gamma that IS there must be recovered, or the tone arm proves nothing."""

    def test_a_known_gamma_is_recovered_from_the_quantile_curve(self):
        # GREY, deliberately. The fit reads LUMA quantiles, and luma is a linear
        # mix of three channels while a gamma is not, so
        # `mix(c_i ** g) != mix(c_i) ** g` and a colour fixture would measure the
        # mixing bias rather than the recovery. That approximation is a real
        # limit of the tone arm and the report states it; this case isolates the
        # fit itself by handing it an image on which the two agree exactly.
        rng = np.random.default_rng(2)
        g = np.clip(rng.normal(0.45, 0.16, (48, 64, 1)), 0.02, 0.98)
        base = np.repeat(g, 3, axis=2)
        src = (255.0 * base).astype(np.uint8)
        dst = np.clip(np.rint(255.0 * 0.93 * (base ** 1.35)), 0, 255).astype(np.uint8)
        fit = D.fit_gain_gamma(D.luma_quantiles([src]), D.luma_quantiles([dst]))
        self.assertAlmostEqual(fit["gamma"], 1.35, delta=0.06)
        self.assertAlmostEqual(fit["gain"], 0.93, delta=0.05)

    def test_applying_the_fit_moves_the_level_toward_the_target(self):
        rng = np.random.default_rng(4)
        base = np.clip(rng.normal(0.5, 0.15, (48, 64, 3)), 0.02, 0.98)
        src = (255.0 * base).astype(np.uint8)
        dst = np.clip(np.rint(255.0 * 0.85 * (base ** 1.2)), 0, 255).astype(np.uint8)
        fit = D.fit_gain_gamma(D.luma_quantiles([src]), D.luma_quantiles([dst]))
        got = D.apply_gain_gamma(src, fit["gain"], fit["gamma"])
        before = abs(float(_cmp.luma(src).mean()) - float(_cmp.luma(dst).mean()))
        after = abs(float(_cmp.luma(got).mean()) - float(_cmp.luma(dst).mean()))
        self.assertLess(after, before / 4.0)

    def test_a_collapsed_render_cannot_be_fitted_and_says_so(self):
        flat = np.zeros(101)
        with self.assertRaises(D.UnreadableInput):
            D.fit_gain_gamma(flat, flat)


class Spectrum(unittest.TestCase):
    """A profile that cannot locate a frequency cannot localise a deficit."""

    @staticmethod
    def _sine(f: float, h: int = 96, w: int = 128) -> np.ndarray:
        x = np.arange(w)[None, :]
        return 128.0 + 100.0 * np.sin(2.0 * np.pi * f * x) * np.ones((h, 1))

    def test_a_pure_sinusoid_lands_in_the_bin_that_names_its_frequency(self):
        for f in (0.08, 0.19, 0.33):
            c, p = D.radial_power(self._sine(f), None, nbins=64)
            self.assertAlmostEqual(float(c[int(np.nanargmax(p))]), f, delta=0.012,
                                   msg=f"peak misplaced for f={f}")

    def test_the_dc_term_is_not_reported_as_signal(self):
        flat = np.full((96, 128), 200.0)
        _, p = D.radial_power(flat, None, nbins=32)
        self.assertLess(float(np.nansum(p)), 1e-6)

    def test_blurring_moves_high_frequency_energy_DOWN(self):
        # The direction the smoothness hypothesis needed. If this case cannot
        # see it, the measured absence of it downstream means nothing.
        rng = np.random.default_rng(9)
        a = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        b = D.gaussian_blur(a, 1.2)
        c, pa = D.radial_power(_cmp.luma(a), None, nbins=64)
        _, pb = D.radial_power(_cmp.luma(b), None, nbins=64)
        fa = D.hf_fraction(c, np.array([pa]), 0.25)[0]
        fb = D.hf_fraction(c, np.array([pb]), 0.25)[0]
        self.assertLess(fb, fa / 2.0)

    def test_the_crossover_of_a_uniformly_scaled_copy_is_the_first_bin(self):
        # A pure SCALE difference must not read as a rolloff. Halving every
        # amplitude divides every bin by four, so the ratio is flat and below
        # one everywhere, and the crossover belongs at the bottom of the range.
        rng = np.random.default_rng(13)
        l = rng.normal(128.0, 30.0, (96, 128))
        c, p1 = D.radial_power(l, None, nbins=64)
        _, p2 = D.radial_power(l * 0.5, None, nbins=64)
        ratio = p2 / p1
        have = np.isfinite(ratio)
        self.assertTrue(np.all(ratio[have] < 1.0))
        below = have & (ratio < 1.0)
        cross = next(float(c[i]) for i in range(len(ratio))
                     if have[i] and np.all(below[i:][have[i:]]))
        # The first bin that HAS coefficients, not the first bin. At 96x128 the
        # lowest annulus is empty, and an empty bin must not push the crossover
        # up and read as a rolloff -- which is exactly the defect this case
        # found in the first version of the crossover rule.
        self.assertEqual(cross, float(c[int(np.argmax(have))]))


class Bands(unittest.TestCase):
    """Mid against high, and the churn that separates detail from grain."""

    def test_a_mid_band_sinusoid_lands_in_the_mid_share_and_not_the_high_one(self):
        f = 0.09
        x = np.arange(128)[None, :]
        img = np.clip(128.0 + 90.0 * np.sin(2 * np.pi * f * x) * np.ones((96, 1)),
                      0, 255).astype(np.uint8)
        frames = [np.repeat(img[:, :, None], 3, axis=2)] * 3
        t = D.band_terms(frames)
        self.assertGreater(float(t["mid_fraction"].mean()), 0.5)
        self.assertLess(float(t["hi_fraction"].mean()), 0.05)

    def test_static_detail_churns_far_less_than_per_frame_grain(self):
        # THE DISCRIMINATOR. A fixed high-frequency texture repeated across
        # frames must churn near zero; independent noise per frame must churn
        # near or above one. If this case cannot tell them apart, the churn
        # number reported for the two renders decides nothing.
        rng = np.random.default_rng(5)
        tex = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        static = [tex.copy() for _ in range(6)]
        grain = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(6)]
        self.assertLess(D.band_terms(static)["hi_temporal_churn"], 0.05)
        self.assertGreater(D.band_terms(grain)["hi_temporal_churn"], 1.0)

    def test_an_axis_aligned_nyquist_pattern_beats_its_own_radial_ring(self):
        # What a separable upsampling stage leaves behind, and the probe has to
        # see it as an AXIS excess rather than as ordinary fine texture.
        y = np.arange(96)[:, None]
        img = np.clip(128.0 + 60.0 * np.cos(np.pi * y) * np.ones((1, 128)),
                      0, 255).astype(np.uint8)
        frames = [np.repeat(img[:, :, None], 3, axis=2)] * 2
        a = D.nyquist_axes(frames)
        self.assertGreater(a["vertical_nyquist"], 20.0 * a["radial_ring_mean"])

    def test_a_smooth_gradient_has_no_axis_excess(self):
        y = np.linspace(20, 230, 96)[:, None] * np.ones((1, 128))
        img = np.repeat(np.clip(y, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
        a = D.nyquist_axes([img, img])
        self.assertLess(a["vertical_nyquist"], 5.0 * a["radial_ring_mean"] + 1e-6)


class Correlation(unittest.TestCase):
    """The hypothesis test, and the two ways it can be vacuous."""

    def test_a_planted_positive_association_is_recovered(self):
        x = np.linspace(0.0, 1.0, 25)
        y = 3.0 * x + 0.5
        self.assertAlmostEqual(D.pearson(x, y), 1.0, places=9)
        self.assertAlmostEqual(D.spearman(x, y), 1.0, places=9)

    def test_a_planted_negative_association_is_recovered_with_its_SIGN(self):
        # The measured within-render coefficient is negative, which is the
        # opposite of what the hypothesis predicts. A `pearson` that lost the
        # sign would turn that refutation into a confirmation.
        x = np.linspace(0.0, 1.0, 25)
        self.assertAlmostEqual(D.pearson(x, -2.0 * x + 7.0), -1.0, places=9)
        self.assertAlmostEqual(D.spearman(x, -2.0 * x + 7.0), -1.0, places=9)

    def test_a_constant_input_returns_nan_and_never_zero(self):
        # A zero would read as "measured, no association". A constant input is
        # an input on which no association CAN be measured, and the two must not
        # print the same way.
        self.assertTrue(np.isnan(D.pearson(np.ones(25), np.arange(25.0))))

    def test_spearman_sees_a_monotone_relation_pearson_understates(self):
        x = np.linspace(0.1, 1.0, 25)
        y = np.exp(6.0 * x)
        self.assertAlmostEqual(D.spearman(x, y), 1.0, places=9)
        self.assertLess(D.pearson(x, y), 0.95)


class Interventions(unittest.TestCase):
    """The blur has to blur and the unsharp has to sharpen, in the statistic the
    report quotes. A no-op intervention reads as "detail does not matter"."""

    def test_the_blur_lowers_the_sharpness_statistic_monotonically(self):
        rng = np.random.default_rng(17)
        a = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
        vals = [D.mean_sharpness([D.gaussian_blur(a, s)]) for s in (0.0, 0.5, 1.0, 2.0)]
        self.assertTrue(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)),
                        f"not monotone: {vals}")

    def test_a_zero_sigma_blur_is_the_identity_and_not_a_quiet_copy(self):
        rng = np.random.default_rng(19)
        a = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(D.gaussian_blur(a, 0.0), a))

    def test_the_unsharp_raises_the_sharpness_statistic(self):
        rng = np.random.default_rng(23)
        a = D.gaussian_blur(rng.integers(0, 256, (64, 96, 3), dtype=np.uint8), 1.0)
        self.assertGreater(D.mean_sharpness([D.unsharp(a, 1.0, 1.0)]),
                           D.mean_sharpness([a]))

    def test_the_blur_stays_inside_the_representable_range(self):
        rng = np.random.default_rng(29)
        a = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        b = D.gaussian_blur(a, 1.5)
        self.assertEqual(b.dtype, np.uint8)
        self.assertGreaterEqual(int(b.min()), 0)
        self.assertLessEqual(int(b.max()), 255)


if __name__ == "__main__":
    unittest.main(verbosity=2)
