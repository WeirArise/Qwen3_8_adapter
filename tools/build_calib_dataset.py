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

from __future__ import annotations

"""
Qwen3.8 calibration-set builder.

Produces ``lab_calib/qwen38_calib.jsonl`` — the calibration corpus referenced by
``lab_practice/qwen3_8/qwen3_8_w8a8_dynamic.yaml`` and
``lab_practice/qwen3_8/qwen3_8_w4a8_flex_awq.yaml``.

Design notes
------------
Activation quantisation (W8A8 / W4A8 per-token) derives its scales from the
activation statistics observed on this corpus, so the *domain coverage* of the
corpus matters more than its absolute size.  The mix below therefore mirrors the
domain distribution a general-purpose chat model is expected to serve, while
half of the samples are synthesised here from scratch (verified arithmetic,
generated code tasks, generated structured-reasoning and instruction items) so
that the corpus is reproducible and self-contained.

Every sample carries the same single JSON field the framework loader expects
(``inputs_pretokenized``); provenance for each domain is written to
``lab_calib/qwen38_calib.provenance.json``.

Usage::

    python tools/build_calib_dataset.py                 # defaults, seed=20260824
    python tools/build_calib_dataset.py --seed 7 --out lab_calib/qwen38_calib.jsonl
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "lab_calib" / "qwen38_calib.jsonl"
DEFAULT_SEED = 20260824

# Domain mix: (domain id, sample count, synthesised-in-repo?)
MIX: List[tuple[str, int, bool]] = [
    ("zh_mcq", 12, False),
    ("math_cot", 12, True),
    ("prose_zh_en", 12, False),
    ("code", 10, True),
    ("en_reading", 8, False),
    ("structured_reasoning", 6, True),
    ("instruction", 4, True),
]


# --------------------------------------------------------------------------- #
# Synthesised domains
# --------------------------------------------------------------------------- #
def _num(value) -> str:
    """Render a number without a trailing .0 when it is integral."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def gen_math_cot(rng: random.Random) -> str:
    """GSM8K-style word problems whose answers are computed, never hardcoded."""
    kind = rng.choice(["unit_rate", "discount", "work_rate", "distance", "interest", "average", "age"])
    if kind == "unit_rate":
        per_hour, hours, days = rng.randint(12, 60), rng.randint(4, 9), rng.randint(3, 9)
        total = per_hour * hours * days
        body = (
            f"A workshop produces {per_hour} units every hour and runs {hours} hours per day.\n"
            f"Let's think step by step\n"
            f"In one day it makes {per_hour} x {hours} = {per_hour * hours} units.\n"
            f"Over {days} days it makes {per_hour * hours} x {days} = {total} units.\n"
            f"The answer is {total}."
        )
        q = f"A workshop produces {per_hour} units every hour and runs {hours} hours per day. How many units in {days} days?"
    elif kind == "discount":
        price = rng.randrange(40, 400, 10)
        pct = rng.choice([10, 15, 20, 25, 30, 40])
        cut = price * pct / 100
        pay = price - cut
        body = (
            f"Let's think step by step\n"
            f"The discount is {price} x {pct} / 100 = {_num(cut)}.\n"
            f"The sale price is {price} - {_num(cut)} = {_num(pay)}.\n"
            f"The answer is {_num(pay)}."
        )
        q = f"A jacket is listed at ${price}. A store offers {pct}% off. What is the sale price in dollars?"
    elif kind == "work_rate":
        workers, days, more = rng.randint(3, 12), rng.randint(4, 20), rng.randint(2, 6)
        total = workers * days
        if total % more:
            more = rng.choice([d for d in range(2, 7) if total % d == 0] or [workers])
            total = workers * days
        body = (
            f"Let's think step by step\n"
            f"The job needs {workers} x {days} = {total} worker-days.\n"
            f"With {more} workers it takes {total} / {more} = {total // more} days.\n"
            f"The answer is {total // more}."
        )
        q = f"{workers} workers build a wall in {days} days. At the same rate, how many days do {more} workers need?"
    elif kind == "distance":
        speed, hours = rng.randint(40, 120), rng.randint(2, 7)
        body = (
            f"Let's think step by step\n"
            f"Distance = speed x time = {speed} x {hours} = {speed * hours} km.\n"
            f"The answer is {speed * hours}."
        )
        q = f"A train travels at {speed} km/h for {hours} hours. How far does it travel in kilometres?"
    elif kind == "interest":
        principal, rate, years = rng.randrange(1000, 20000, 500), rng.choice([2, 3, 4, 5, 6]), rng.randint(2, 8)
        interest = principal * rate * years / 100
        body = (
            f"Let's think step by step\n"
            f"Simple interest = principal x rate x years / 100.\n"
            f"= {principal} x {rate} x {years} / 100\n"
            f"= {_num(principal * rate * years)} / 100\n"
            f"= {_num(interest)}.\n"
            f"The answer is {_num(interest)}."
        )
        q = (f"A sum of ${principal} is deposited at {rate}% simple interest per year. "
             f"What is the total interest earned after {years} years, in dollars?")
    elif kind == "average":
        count, avg = rng.randint(4, 9), rng.randint(10, 90)
        total = count * avg
        added = rng.randint(50, 100)
        new_avg = round((total + added) / (count + 1), 2)
        body = (
            f"Let's think step by step\n"
            f"The {count} numbers sum to {count} x {avg} = {total}.\n"
            f"Adding {added} gives {total} + {added} = {total + added}.\n"
            f"Dividing by {count + 1} numbers gives {new_avg}.\n"
            f"The answer is {new_avg}."
        )
        q = (f"The average of {count} numbers is {avg}. A new number {added} is added. "
             f"What is the new average, rounded to two decimals?")
    else:  # age
        ann, k, y = rng.randint(8, 20), rng.randint(2, 4), rng.randint(3, 12)
        body = (
            f"Let's think step by step\n"
            f"Ann's brother is {ann} x {k} = {ann * k} years old now.\n"
            f"In {y} years Ann will be {ann} + {y} = {ann + y}.\n"
            f"In {y} years her brother will be {ann * k} + {y} = {ann * k + y}.\n"
            f"The answer is {ann * k + y}."
        )
        q = (f"Ann is {ann} years old. Her brother is {k} times as old as Ann. "
             f"How old will her brother be in {y} years?")
    return f"Question: {q}\n{body}"


_CODE_TASKS = [
    ("count_occurrences", "items: list, target",
     "Return how many times `target` appears in `items`.",
     "count_occurrences([1, 2, 1, 1], 1) == 3", "sum(1 for x in items if x == target)"),
    ("longest_run", "items: list",
     "Return the length of the longest run of equal consecutive elements.",
     "longest_run([1, 1, 2, 2, 2, 3]) == 3",
     "max((len(list(g)) for _, g in __import__('itertools').groupby(items)), default=0)"),
    ("is_balanced_brackets", "s: str",
     "Return True when every bracket in `s` is closed in the correct order.",
     "is_balanced_brackets('([]{})') == True",
     "balanced = {'(': ')', '[': ']', '{': '}'}"),
    ("chunk", "items: list, size: int",
     "Split `items` into consecutive chunks of at most `size` elements.",
     "chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
     "[items[i:i + size] for i in range(0, len(items), size)]"),
    ("normalise_whitespace", "text: str",
     "Collapse every whitespace run into a single space and strip the ends.",
     "normalise_whitespace('  a \\n b ') == 'a b'",
     "' '.join(text.split())"),
]


def gen_code(rng: random.Random) -> str:
    name, sig, desc, example, _impl = rng.choice(_CODE_TASKS)
    style = rng.choice(["docstring", "spec", "review"])
    if style == "docstring":
        return (f"Complete the following Python function.\n\n"
                f"def {name}({sig}):\n"
                f'    """{desc}\n\n'
                f"    >>> {example}\n"
                f'    """\n')
    if style == "spec":
        return (f"Implement a Python function `{name}`.\n\n"
                f"Signature: {name}({sig})\n"
                f"Behaviour: {desc}\n"
                f"Example: {example}\n"
                f"Return the result; do not print it.\n")
    return (f"Review the function below and explain what it returns for the given input.\n\n"
            f"def {name}({sig}):\n"
            f"    # {desc}\n"
            f"    ...\n\n"
            f"Input: {example.split('==')[0].strip()}\n"
            f"State the returned value.\n")


def gen_structured_reasoning(rng: random.Random) -> str:
    regions = rng.sample(["North", "South", "East", "West", "Central"], k=3)
    quarters = [f"Q{q}" for q in range(1, 5)]
    rows = {r: [rng.randint(20, 200) for _ in quarters] for r in regions}
    total = {r: sum(v) for r, v in rows.items()}
    best = max(total, key=lambda r: total[r])
    header = "Region | " + " | ".join(quarters) + " | Total"
    lines = [header, "-" * len(header)]
    for r in regions:
        lines.append(f"{r} | " + " | ".join(str(v) for v in rows[r]) + f" | {total[r]}")
    table = "\n".join(lines)
    style = rng.choice(["argmax", "sum", "share"])
    if style == "argmax":
        q = "Which region has the highest total for the year? Answer with the region name."
        a = f"The totals are " + ", ".join(f"{r}={total[r]}" for r in regions) + f".\nThe highest is {best}.\nThe answer is {best}."
    elif style == "sum":
        grand = sum(total.values())
        q = "What is the combined yearly total across all regions in the table?"
        a = "Adding the region totals " + " + ".join(str(total[r]) for r in regions) + f" gives {grand}.\nThe answer is {grand}."
    else:
        grand = sum(total.values())
        lo = min(total, key=lambda r: total[r])
        q = f"What percentage of the combined total does {lo} contribute, to one decimal place?"
        pct = round(total[lo] * 100 / grand, 1)
        a = f"{lo} totals {total[lo]} out of {grand}, so {total[lo]}/{grand} = {pct}%.\nThe answer is {pct}."
    return (f"The table below reports quarterly revenue (thousands of dollars).\n\n"
            f"{table}\n\nQuestion: {q}\nLet's think step by step\n{a}")


_INSTRUCTION_TEMPLATES = [
    ("summarise", "Summarise the following passage in {n} bullet points. Keep every bullet under 20 words."),
    ("rewrite", "Rewrite the following passage for a technical audience. Preserve all facts; do not add new claims."),
    ("extract", "Extract every {n} mentioned in the passage as a JSON list. Output JSON only."),
    ("compare", "Compare the two described approaches in a markdown table with columns: aspect, approach A, approach B."),
]


def gen_instruction(rng: random.Random) -> str:
    tag, tpl = rng.choice(_INSTRUCTION_TEMPLATES)
    n = rng.choice([3, 4, 5])
    topic = rng.choice([
        "post-training quantisation of a large language model",
        "the trade-off between activation precision and inference throughput",
        "why calibration corpora should match the deployment domain",
        "the role of per-token scaling in W8A8 inference",
    ])
    subject = rng.choice(["quantisation levels", "risk factors", "evaluation metrics", "implementation steps"])
    body = (
        f"A team is preparing to deploy a model and must decide on {topic}. "
        f"They run a small pilot, record the {subject}, and compare the outcome against a "
        f"float baseline. Reviewers ask for a short written recommendation, a list of the "
        f"{subject} that drove the decision, and an explicit note of what would change their mind."
    )
    return f"{tpl.format(n=n)}\n\nPassage:\n{body}\n"


# --------------------------------------------------------------------------- #
# Downstream pools
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path, key: str = "inputs_pretokenized") -> List[str]:
    out: List[str] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get(key), str) and item[key].strip():
                out.append(item[key])
            elif isinstance(item, str) and item.strip():
                out.append(item)
    return out


POOLS: Dict[str, List[Path]] = {
    "zh_mcq": [REPO_ROOT / "example" / "common" / "ceval.jsonl",
               REPO_ROOT / "example" / "common" / "cn_en.jsonl"],
    "en_reading": [REPO_ROOT / "lab_calib" / "boolq.jsonl",
                   REPO_ROOT / "example" / "common" / "boolq.jsonl"],
    "prose_zh_en": [REPO_ROOT / "example" / "common" / "wiki.jsonl",
                    REPO_ROOT / "lab_calib" / "cn_en.jsonl"],
}

SYNTH: Dict[str, Callable[[random.Random], str]] = {
    "math_cot": gen_math_cot,
    "code": gen_code,
    "structured_reasoning": gen_structured_reasoning,
    "instruction": gen_instruction,
}


@dataclass
class Result:
    samples: List[str] = field(default_factory=list)
    provenance: Dict[str, dict] = field(default_factory=dict)


def build(seed: int = DEFAULT_SEED) -> Result:
    rng = random.Random(seed)
    res = Result()
    # global de-duplication: the public pools overlap between domains, so a text
    # picked for one domain must not be picked again for another
    seen_global: set[str] = set()
    for domain, count, synthesised in MIX:
        if synthesised:
            items, guard = [], 0
            while len(items) < count and guard < count * 200:
                cand = SYNTH[domain](rng)
                guard += 1
                if cand not in seen_global:
                    seen_global.add(cand)
                    items.append(cand)
            res.provenance[domain] = {
                "count": len(items),
                "origin": "synthesised in this repository",
                "generator": SYNTH[domain].__name__,
            }
        else:
            pool: List[str] = []
            for p in POOLS[domain]:
                pool.extend(_read_jsonl(p))
            if not pool:
                res.provenance[domain] = {"count": 0, "origin": "pool unavailable", "sources": []}
                continue
            # shuffle with our own seed so the draw differs from any other mix
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            items = []
            for i in indices:
                if len(items) >= count:
                    break
                if pool[i] not in seen_global:
                    seen_global.add(pool[i])
                    items.append(pool[i])
            res.provenance[domain] = {
                "count": len(items),
                "origin": "public corpora sampled locally",
                "sources": [str(p.relative_to(REPO_ROOT)) for p in POOLS[domain] if p.is_file()],
                "pool_size": len(pool),
                "selection": f"seeded shuffle (seed={seed}), first {len(items)} unused",
            }
        res.samples.extend(items)
    rng.shuffle(res.samples)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Qwen3.8 calibration corpus.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    res = build(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for text in res.samples:
            fh.write(json.dumps({"inputs_pretokenized": text}, ensure_ascii=False) + "\n")

    prov_path = args.out.with_suffix(".provenance.json")
    prov_path.write_text(
        json.dumps(
            {"seed": args.seed, "total": len(res.samples), "domains": res.provenance},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(res.samples)} samples -> {args.out}")
    for domain, info in res.provenance.items():
        print(f"  {domain:<22} {info['count']:>3}  {info['origin']}")
    print(f"provenance -> {prov_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
