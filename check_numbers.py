"""Check that the numbers typed into the manuscript still match the data.

The manuscript quotes results as plain numbers. paper/numbers_in_text.tex records, for every
quoted result, the value make_tables.py produced when it was typed in. This script compares that
record with a freshly generated paper/generated/results_macros.tex and lists any value that has
changed, so the corresponding number in FGCS_2026.tex can be updated by hand.
"""
import os
import re
import sys

H = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", re.M)


def load(path):
    return dict(PAT.findall(open(os.path.join(H, path)).read()))


typed = load("paper/numbers_in_text.tex")
fresh = load("paper/generated/results_macros.tex")
bad = [(k, v, fresh.get(k)) for k, v in sorted(typed.items()) if fresh.get(k) != v]
if bad:
    print("Numbers in the manuscript that no longer match the data:")
    for k, old, new in bad:
        print(f"  {k}: text has {old!r}, data give {new!r}")
    sys.exit(1)
print(f"All {len(typed)} quoted results match the data.")
