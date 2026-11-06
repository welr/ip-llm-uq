"""Fetch the remaining study models into the Hugging Face cache before generation.

A download inside regen_instruct.py stalled silently on 2026-11-04: no bytes in, no writes, 0% CPU, and no error. The generation runs have no watchdog, so this step restarts any download whose cache folder stops growing.
Dependency: huggingface_hub (Apache-2.0)."""
import os, subprocess, sys, time
REPOS = ['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-7B', 'google/gemma-2-9b-it', 'google/gemma-2-9b']
PAT = "['*.safetensors', '*.json', 'tokenizer*', '*.model', '*.txt']"
STALL, TRIES = 180, 8
def size(r):
    p = os.path.expanduser('~/.cache/huggingface/hub/models--' + r.replace('/', '--')); n = 0
    for d, _, fs in os.walk(p):
        for f in fs:
            q = os.path.join(d, f)
            if not os.path.islink(q): n += os.path.getsize(q)
    return n
for r in REPOS:
    for t in range(TRIES):
        code = f"from huggingface_hub import snapshot_download; snapshot_download({r!r}, allow_patterns={PAT}, ignore_patterns=['original/*'])"
        P = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); last, since = size(r), time.time()
        while P.poll() is None:
            time.sleep(15); s = size(r)
            if s != last: last, since = s, time.time()
            elif time.time() - since > STALL: P.kill(); P.wait(); print(r, 'stalled at', last, 'bytes, attempt', t + 1, flush=True); break
        if P.returncode == 0: print(r, 'ready', last, 'bytes', flush=True); break
    else: sys.exit(f'{r}: failed after {TRIES} attempts')
