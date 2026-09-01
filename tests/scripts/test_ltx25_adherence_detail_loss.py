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
        # RAW, because this case is about the radial BINNING and nothing else.
        # `periodic_component` deliberately injects a low-frequency counter-term
        # whose size scales with the boundary jump, and a high-frequency sinusoid
        # is almost all boundary jump, so under that convention the counter-term
        # outweighs the signal's own ring mean and the peak lands in bin 0. That
        # is the decomposition behaving correctly on a pathological fixture, not
        # a binning defect, and `test_the_periodic_counter_term_is_real` below
        # pins it as a known property rather than leaving it to be rediscovered.
        for f in (10 / 128, 24 / 128, 42 / 128):
            c, p = D.radial_power(self._sine(f), None, nbins=64, convention="raw")
            self.assertAlmostEqual(float(c[int(np.nanargmax(p))]), f, delta=0.012,
                                   msg=f"peak misplaced for f={f}")

    def test_the_periodic_counter_term_is_real_and_scales_with_the_boundary(self):
        # WHY THE REPORTED ESTIMATOR IS WELCH AND NOT THIS. `p = u - s` removes
        # the wrap step exactly, but `s` carries it into a low-frequency
        # counter-term, and the two renders do not share a boundary jump -- so
        # this convention is not neutral between them either.
        x = np.arange(128)[None, :]
        hi_f = 128.0 + 100.0 * np.sin(2 * np.pi * (24 / 128) * x) * np.ones((96, 1))
        c, p = D.radial_power(hi_f, None, nbins=64, convention="periodic")
        self.assertLess(float(c[int(np.nanargmax(p))]), 0.05,
                        "the counter-term no longer dominates; the note in "
                        "welch_psd's docstring and this row's Outcome would "
                        "need re-deriving")

    def test_the_dc_term_is_not_reported_as_signal(self):
        flat = np.full((96, 128), 200.0)
        _, p = D.radial_power(flat, None, nbins=32, convention="raw")
        self.assertLess(float(np.nansum(p)), 1e-6)

    def test_welch_never_sees_the_frame_border(self):
        # THE PROPERTY THE WHOLE CORRECTION RESTS ON. Paint a cliff into the
        # outermost rows and columns only. A whole-frame estimator's answer moves;
        # Welch's interior tiles must not see it at all.
        rng = np.random.default_rng(31)
        base = rng.integers(60, 190, (192, 320, 3), dtype=np.uint8)
        edged = base.copy()
        edged[0, :, :] = 0; edged[-1, :, :] = 255
        edged[:, 0, :] = 0; edged[:, -1, :] = 255
        r = D.tile_freq_grid(); hi = (r >= 0.20); nz = r > 0
        a = D.welch_psd([base])[0]; b = D.welch_psd([edged])[0]
        sa = float(a[hi].sum() / a[nz].sum()); sb = float(b[hi].sum() / b[nz].sum())
        self.assertAlmostEqual(sa, sb, places=9)
        # and the border DOES move a whole-frame estimator, or the case is vacuous
        pa = D._power(D.luma(base) - D.luma(base).mean())
        pb = D._power(D.luma(edged) - D.luma(edged).mean())
        rr = D._fgrid(192, 320); h2 = (rr >= 0.20); n2 = rr > 0
        self.assertNotAlmostEqual(float(pa[h2].sum() / pa[n2].sum()),
                                  float(pb[h2].sum() / pb[n2].sum()), places=4)

    def test_welch_refuses_a_frame_too_small_to_hold_an_interior_tile(self):
        tiny = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(D.UnreadableInput):
            D.welch_psd([tiny])

    def test_blurring_moves_high_frequency_energy_DOWN(self):
        # The direction the smoothness hypothesis needed. If this case cannot
        # see it, the measured absence of it downstream means nothing.
        rng = np.random.default_rng(9)
        a = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        b = D.gaussian_blur(a, 1.2)
        c, pa = D.radial_power(_cmp.luma(a), None, nbins=64, convention="raw")
        _, pb = D.radial_power(_cmp.luma(b), None, nbins=64, convention="raw")
        fa = D.hf_fraction(c, np.array([pa]), 0.25)[0]
        fb = D.hf_fraction(c, np.array([pb]), 0.25)[0]
        self.assertLess(fb, fa / 2.0)

    def test_the_crossover_of_a_uniformly_scaled_copy_is_the_first_bin(self):
        # A pure SCALE difference must not read as a rolloff. Halving every
        # amplitude divides every bin by four, so the ratio is flat and below
        # one everywhere, and the crossover belongs at the bottom of the range.
        rng = np.random.default_rng(13)
        l = rng.normal(128.0, 30.0, (96, 128))
        c, p1 = D.radial_power(l, None, nbins=64, convention="raw")
        _, p2 = D.radial_power(l * 0.5, None, nbins=64, convention="raw")
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


class PeriodicDecomposition(unittest.TestCase):
    """The repair that changed this row's headline, so it is the most heavily
    guarded thing in the file.

    A reviewer reproduced the high-band comparison with the raw convention and
    got the OPPOSITE SIGN. Both readings were the DFT's wrap step, which the two
    renders do not carry equally. These cases pin that the decomposition removes
    that step, leaves interior content alone, and actually changes the answer on
    a frame whose borders disagree -- because a no-op would read as "the
    convention did not matter" and put the withdrawn headline straight back.
    """

    @staticmethod
    def _ramp(h=64, w=96):
        # Borders disagree by construction: a horizontal ramp wraps as a cliff.
        return np.linspace(10.0, 240.0, w)[None, :] * np.ones((h, 1))

    def test_the_periodic_component_removes_the_wrap_step(self):
        u = self._ramp()
        p = D.periodic_component(u)
        before = float(np.abs(u[:, 0] - u[:, -1]).mean())
        after = float(np.abs(p[:, 0] - p[:, -1]).mean())
        self.assertGreater(before, 100.0)
        self.assertLess(after, before / 10.0)

    def test_an_already_periodic_image_is_left_alone(self):
        # A sinusoid at an integer number of cycles has no wrap DISCONTINUITY,
        # so the decomposition must be near the identity on it. A transform that
        # "cleans" such an image is altering content, not removing an artefact.
        #
        # The bound is ONE ORDINARY INTERIOR STEP and not a constant, because a
        # sampled sinusoid's last-to-first difference is not zero: the wrap runs
        # from sample N-1 to sample 0, which is one step like any other. A
        # tighter bound would be asserting the fixture is continuous when it is
        # merely periodic, and it failed here for exactly that reason.
        x = np.arange(96)[None, :]
        u = 128.0 + 40.0 * np.sin(2 * np.pi * 3 * x / 96) * np.ones((64, 1))
        step = float(np.abs(np.diff(u, axis=1)).mean())
        self.assertLess(float(np.abs(D.periodic_component(u) - u).max()), step)

    def test_it_changes_the_high_band_share_on_a_frame_with_a_wrap_step(self):
        # THE CASE THAT MATTERS. If this passed with a no-op decomposition the
        # withdrawn headline would come back silently.
        u = self._ramp() + np.random.default_rng(3).normal(0, 4.0, (64, 96))
        r = D._fgrid(64, 96)
        hi = (r >= 0.20) & (r < 0.71)
        tot = r > 0
        raw = D._power(u - u.mean())
        per = D._power(D.periodic_component(u))
        s_raw = float(raw[hi].sum() / raw[tot].sum())
        s_per = float(per[hi].sum() / per[tot].sum())
        self.assertGreater(s_per, 1.5 * s_raw,
                           f"decomposition barely moved the share: {s_raw} -> {s_per}")

    def test_the_wrap_statistic_ranks_a_cliff_above_a_seamless_frame(self):
        seam = np.repeat((self._ramp())[:, :, None], 3, axis=2).astype(np.uint8)
        x = np.arange(96)[None, :]
        smooth = 128.0 + 40.0 * np.sin(2 * np.pi * 3 * x / 96) * np.ones((64, 1))
        per = np.repeat(smooth[:, :, None], 3, axis=2).astype(np.uint8)
        a = D.wrap_discontinuity([seam, seam])
        b = D.wrap_discontinuity([per, per])
        self.assertGreater(a["jump_over_interior_step"],
                           10.0 * b["jump_over_interior_step"])

    def test_radial_power_rejects_nothing_and_the_conventions_differ(self):
        u = self._ramp() + np.random.default_rng(5).normal(0, 3.0, (64, 96))
        c, p_raw = D.radial_power(u, None, 32, "raw")
        _, p_per = D.radial_power(u, None, 32, "periodic")
        self.assertFalse(np.allclose(np.nan_to_num(p_raw), np.nan_to_num(p_per)))


class Bands(unittest.TestCase):
    """Mid against high, and the churn that separates detail from grain."""

    def test_a_mid_band_sinusoid_lands_in_the_mid_share_and_not_the_high_one(self):
        f = 0.09
        x = np.arange(320)[None, :]
        img = np.clip(128.0 + 90.0 * np.sin(2 * np.pi * f * x) * np.ones((192, 1)),
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
        tex = rng.integers(0, 256, (192, 320, 3), dtype=np.uint8)
        static = [tex.copy() for _ in range(6)]
        grain = [rng.integers(0, 256, (192, 320, 3), dtype=np.uint8) for _ in range(6)]
        self.assertLess(D.band_terms(static)["hi_temporal_churn"], 0.05)
        self.assertGreater(D.band_terms(grain)["hi_temporal_churn"], 1.0)

    def test_an_axis_aligned_nyquist_pattern_beats_its_own_radial_ring(self):
        # What a separable upsampling stage leaves behind, and the probe has to
        # see it as an AXIS excess rather than as ordinary fine texture.
        y = np.arange(192)[:, None]
        img = np.clip(128.0 + 60.0 * np.cos(np.pi * y) * np.ones((1, 320)),
                      0, 255).astype(np.uint8)
        frames = [np.repeat(img[:, :, None], 3, axis=2)] * 2
        a = D.nyquist_axes(frames)
        self.assertGreater(a["vertical_nyquist"], 20.0 * a["radial_ring_mean"])

    def test_a_smooth_image_with_broadband_content_has_no_axis_excess(self):
        # THE NOISE IS NOT DECORATION. A noiseless smooth fixture has essentially
        # nothing near Nyquist, so its ring mean is floating-point dust and ANY
        # ratio against it is meaningless -- the earlier version of this case
        # failed for that reason and not because the probe was wrong. Every real
        # frame carries broadband content, so the fixture does too, and the
        # assertion is then about the AXES against a populated ring.
        rng = np.random.default_rng(41)
        y = np.arange(192)[:, None]
        smooth = 128.0 + 90.0 * np.cos(2 * np.pi * y / 192) * np.ones((1, 320))
        img8 = np.clip(smooth + rng.normal(0, 6.0, (192, 320)), 0, 255).astype(np.uint8)
        img = np.repeat(img8[:, :, None], 3, axis=2)
        a = D.nyquist_axes([img, img])
        self.assertLess(a["vertical_nyquist"], 5.0 * a["radial_ring_mean"])

    def test_the_axis_probe_SEPARATES_a_lattice_pattern_from_ordinary_texture(self):
        # The comparative form, which is what the probe is actually used for:
        # ours against the reference. A probe that scored both alike would have
        # nothing to say about either.
        rng = np.random.default_rng(43)
        noise = rng.integers(0, 256, (192, 320, 3), dtype=np.uint8)
        y = np.arange(192)[:, None]
        lattice8 = np.clip(128.0 + 50.0 * np.cos(np.pi * y) * np.ones((1, 320))
                           + rng.normal(0, 20.0, (192, 320)), 0, 255).astype(np.uint8)
        lattice = np.repeat(lattice8[:, :, None], 3, axis=2)
        a = D.nyquist_axes([lattice, lattice])
        b = D.nyquist_axes([noise, noise])
        self.assertGreater(a["vertical_nyquist"] / a["radial_ring_mean"],
                           10.0 * b["vertical_nyquist"] / b["radial_ring_mean"])


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
