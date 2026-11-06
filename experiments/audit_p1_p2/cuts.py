"""Reduce a raw model continuation to the response that is clustered and scored.

registered_cut: the earlier runner's rule. It returns '' on a leading newline and cuts at any period, so 'J. K. Rowling' becomes 'J'.
repaired_cut: the same first-line, first-sentence rule, except that leading whitespace and a repeated answer prefix are removed, an invented next question counts as no answer, and a period
after an initial, a listed abbreviation, or inside a number does not end the sentence."""
import re
ABBREV = {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'mt', 'jr', 'sr', 'vs', 'no', 'gen', 'col', 'sgt', 'capt', 'rev', 'hon', 'lt', 'ft', 'sen', 'gov', 'pres', 'inc', 'co', 'ltd'}
def registered_cut(raw):
    return raw.split('\n')[0].split('.')[0].strip()
def repaired_cut(raw):
    s = re.sub(r'^A:\s*', '', raw.lstrip()).split('\n')[0]
    if s.startswith('Q:'): return ''
    for m in re.finditer(r'\.', s):
        before, after = s[:m.start()], s[m.end():]
        tok = re.split(r'[\s\-(]', before)[-1].split('.')[-1] if before else ''
        if len(tok) == 1 and tok.isupper(): continue
        if tok.lower() in ABBREV: continue
        if tok[-1:].isdigit() and after[:1].isdigit(): continue
        return before.strip()
    return s.strip()
