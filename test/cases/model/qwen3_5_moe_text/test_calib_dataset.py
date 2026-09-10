#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

"""
Tests for the Qwen3.8 calibration-corpus builder.

The generator is deterministic, so its properties can be asserted directly
rather than on a checked-in output: every assertion below either re-derives the
corpus or recomputes the arithmetic it contains.
"""

import json
import re
import unittest
from fractions import Fraction
from pathlib import Path

from tools.build_calib_dataset import MIX, build, gen_math_cot

REPO_ROOT = Path(__file__).resolve().parents[4]


class TestDeterminism(unittest.TestCase):
    def test_same_seed_produces_the_same_corpus(self):
        self.assertEqual(build(seed=1).samples, build(seed=1).samples)

    def test_different_seeds_produce_different_corpora(self):
        self.assertNotEqual(build(seed=1).samples, build(seed=2).samples)


class TestCorpusProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build()
        cls.samples = cls.result.samples

    def test_domain_mix_is_exactly_as_declared(self):
        for domain, count, _ in MIX:
            with self.subTest(domain=domain):
                self.assertEqual(self.result.provenance[domain]["count"], count)

    def test_sample_count_matches_the_mix(self):
        self.assertEqual(len(self.samples), sum(count for _, count, _ in MIX))

    def test_no_duplicates(self):
        self.assertEqual(len(self.samples), len(set(self.samples)))

    def test_no_empty_samples(self):
        self.assertTrue(all(s.strip() for s in self.samples))

    def test_provenance_records_every_domain(self):
        self.assertEqual(set(self.result.provenance), {d for d, _, _ in MIX})

    def test_provenance_distinguishes_synthesised_from_sampled(self):
        synthesised = {d for d, _, synth in MIX if synth}
        for domain, info in self.result.provenance.items():
            with self.subTest(domain=domain):
                if domain in synthesised:
                    self.assertIn("synthesised", info["origin"])
                else:
                    self.assertIn("sampled", info["origin"])
                    self.assertTrue(info["sources"])


class TestSynthesisedArithmetic(unittest.TestCase):
    """Every equation written into a generated chain of thought must be true."""

    EQUATION = re.compile(r"(-?\d+(?:\.\d+)?)\s*([x+])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")

    def test_every_equation_recomputes(self):
        checked = 0
        for text in build().samples:
            for a, op, b, c in self.EQUATION.findall(text):
                a, b, c = Fraction(a), Fraction(b), Fraction(c)
                expected = a * b if op == "x" else a + b
                with self.subTest(equation=f"{a} {op} {b} = {c}"):
                    self.assertEqual(expected, c)
                checked += 1
        self.assertGreater(checked, 0, "no equations found; the generator changed shape")

    def test_generated_problems_are_not_hardcoded(self):
        """Two seeds must not emit the same arithmetic."""
        import random

        rng_a, rng_b = random.Random(11), random.Random(99)
        self.assertNotEqual(
            [gen_math_cot(rng_a) for _ in range(8)],
            [gen_math_cot(rng_b) for _ in range(8)],
        )


class TestOverlapWithUpstreamCorpus(unittest.TestCase):
    def test_no_content_shared_with_the_upstream_mix(self):
        upstream = REPO_ROOT / "lab_calib" / "mix_calib.jsonl"
        if not upstream.is_file():
            self.skipTest("upstream mix_calib.jsonl not present")
        other = {
            json.loads(line)["inputs_pretokenized"]
            for line in upstream.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        overlap = set(build().samples) & other
        self.assertEqual(overlap, set(), f"{len(overlap)} samples duplicated from the upstream corpus")


class TestLoaderCompatibility(unittest.TestCase):
    def test_written_file_only_uses_the_field_the_loader_reads(self):
        path = REPO_ROOT / "lab_calib" / "qwen38_calib.jsonl"
        if not path.is_file():
            self.skipTest("corpus not built yet")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            self.assertEqual(set(json.loads(line)), {"inputs_pretokenized"})

    def test_corpus_is_referenced_by_a_practice_config(self):
        config = REPO_ROOT / "lab_practice" / "qwen3_5_moe_text" / "qwen3_8_2_4t_a95b_w8a8.yaml"
        self.assertIn("qwen38_calib.jsonl", config.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
