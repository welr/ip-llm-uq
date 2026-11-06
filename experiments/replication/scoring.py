"""Answer scoring for the PopQA replication.

registered_rule: the two-way substring test of the earlier registrations (500-character cap, as in the current runner).
word_bounded_rule: the same test, with matches required to fall on word boundaries and responses made only of stop words or single letters refused.
"""
import re
STOP = {'a','an','the','of','in','is','it','he','she','they','and','or','to','on','at','by','for','was','his','her','its','i','you','we','this','that','unknown','none','no','yes'}
def norm(s, cap=500):
    if not s: return ''
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower().strip()))[:cap]
def registered_rule(gen, golds):
    g = norm(gen)
    if not g: return False
    return any(a and (a in g or g in a) for a in map(norm, golds))
def _within(x, y):
    return re.search(r'(?<!\w)' + re.escape(x) + r'(?!\w)', y) is not None
def word_bounded_rule(gen, golds):
    g = norm(gen)
    if not g or all(t in STOP or (len(t) == 1 and t.isalpha()) for t in g.split()): return False
    return any(a and (_within(a, g) or _within(g, a)) for a in map(norm, golds))
