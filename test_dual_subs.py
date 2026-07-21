"""Tests for dual_subs.py — stdlib unittest, no network/API calls."""

import unittest

import pysubs2

import dual_subs as ds


def _make_subs(cues):
    """cues: list of (start_ms, end_ms, text) -> SSAFile."""
    subs = pysubs2.SSAFile()
    for start, end, text in cues:
        ev = pysubs2.SSAEvent(start=start, end=end)
        ev.plaintext = text
        subs.append(ev)
    return subs


class ParseNumberedTests(unittest.TestCase):
    def test_restores_newlines_from_slash_separator(self):
        out = ds._parse_numbered("000|Hello there / How are you", [0])
        self.assertEqual(out, ["Hello there\nHow are you"])

    def test_multiple_lines_with_and_without_separator(self):
        text = "000|First\n001|Second / part two"
        out = ds._parse_numbered(text, [0, 1])
        self.assertEqual(out, ["First", "Second\npart two"])

    def test_fallback_plain_lines_restore_newlines(self):
        # No index prefixes; falls back to positional parsing.
        text = "Alpha / beta\nGamma"
        out = ds._parse_numbered(text, [0, 1])
        self.assertEqual(out, ["Alpha\nbeta", "Gamma"])


class TimestampTests(unittest.TestCase):
    def test_parse_fmt_roundtrip_minutes(self):
        ms = ds._parse_ts("01:02.500")
        self.assertEqual(ms, 62500)
        self.assertEqual(ds._fmt_ts(ms), "01:02.500")

    def test_parse_hours_and_comma(self):
        self.assertEqual(ds._parse_ts("1:02:03,004"), 3723004)


class LineReTests(unittest.TestCase):
    def test_matches_large_index(self):
        m = ds.LINE_RE.match("10000|some text")
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 10000)
        self.assertEqual(m.group(2), "some text")

    def test_matches_five_plus_digits_with_spaces(self):
        m = ds.LINE_RE.match("123456 | padded text")
        self.assertIsNotNone(m)
        self.assertEqual(int(m.group(1)), 123456)
        self.assertEqual(m.group(2), "padded text")


class EstimateTimeShiftTests(unittest.TestCase):
    def test_unsorted_secondary_still_estimates_offset(self):
        # Primary at 0-1s, 2-3s, 4-5s. Secondary is the same but late by 1000ms,
        # provided in scrambled (unsorted) order to exercise the sort fix.
        primary = _make_subs([(0, 1000, "a"), (2000, 3000, "b"), (4000, 5000, "c")])
        secondary = _make_subs([(5000, 6000, "c"), (1000, 2000, "a"), (3000, 4000, "b")])
        shift = ds.estimate_time_shift(primary, secondary)
        # Secondary is 1000ms late; a shift near -1000ms should best align it.
        self.assertLess(shift, -500)
        self.assertGreater(shift, -1500)

    def test_empty_inputs_return_zero(self):
        self.assertEqual(ds.estimate_time_shift(_make_subs([]), _make_subs([])), 0)


class ShiftAndMergeOrderTests(unittest.TestCase):
    def test_shift_subs_moves_events(self):
        subs = _make_subs([(1000, 2000, "x")])
        shifted = ds._shift_subs(subs, 500)
        self.assertEqual(shifted[0].start, 1500)
        self.assertEqual(shifted[0].end, 2500)
        # Original is untouched (deep copy).
        self.assertEqual(subs[0].start, 1000)

    def test_manual_shift_to_file2_enables_overlap(self):
        # File 1 (Latin spine) at 2000-3000. File 2 (CJK) at 0-1000 — no overlap
        # until we shift file 2 by +2000ms, mimicking process_merge semantics.
        subs_a = _make_subs([(2000, 3000, "Hello")])
        subs_b = _make_subs([(0, 1000, "\u4f60\u597d")])

        shifted_b = ds._shift_subs(subs_b, 2000)
        merged = ds.merge_subs(subs_a, shifted_b, order="source-top", min_overlap_ms=80)
        self.assertEqual(len(merged), 1)
        self.assertIn("\n", merged[0].plaintext)
        self.assertIn("Hello", merged[0].plaintext)
        self.assertIn("\u4f60\u597d", merged[0].plaintext)


class TranslateAllTests(unittest.TestCase):
    def test_empty_cues_pass_through_and_dedupe(self):
        calls = []

        def fake_translate_batch(client, model, chunk, src, tgt, context, start_index=0):
            calls.append(list(chunk))
            return [f"T:{c}" for c in chunk]

        orig = ds.translate_batch
        ds.translate_batch = fake_translate_batch
        try:
            lines = ["Hello", "", "Hello", "World", "   "]
            out = ds.translate_all(None, "m", lines, "en", "zh-CN", "", batch_size=10)
        finally:
            ds.translate_batch = orig

        self.assertEqual(out, ["T:Hello", "", "T:Hello", "T:World", ""])
        # "Hello" is deduped: only unique non-empty texts get translated once.
        flat = [c for batch in calls for c in batch]
        self.assertEqual(sorted(flat), ["Hello", "World"])


if __name__ == "__main__":
    unittest.main()
