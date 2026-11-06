"""
Independent re-verification of the self-consistency partition baseline
(REBUTTAL_NOTES Q8 / self_consistency_partition_results.json), whose generating
script is not in the tree. Recomputes from source:

  - responses: modal_data/pred1_*.json  (base_responses[:6])
  - per-response correctness: kle_snne_triviaqa_results.json 'accuracies'
    (post-Path-B patched, the source the stored JSON cites)

Partition rule: group the 6 responses by normalize_answer; the "mode cluster"
is the union of all normalized-string groups tied for the largest count.
Label each response mode / non_mode; ratio = mode_acc / non_mode_acc.

Usage: ~/ip_llm/bin/python experiments/verify_self_consistency_partition.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from kle_snne_comparison import normalize_answer

PRED_FILES = {
    "Qwen": "pred1_qwen.json",
    "Mistral": "pred1_mistral_20260110_071956.json",
    "Llama": "pred1_llama8b_20260112_074432.json",
    "Gemma": "pred1_gemma9b_20260112_074222.json",
}


def main():
    sd = Path(__file__).parent
    # correctness source, keyed by (model, str(qid))
    acc_by_key = {}
    for r in json.load(open(sd / "kle_snne_triviaqa_results.json")):
        acc_by_key[(r["model"], str(r["qid"]))] = r["accuracies"]

    per_model = {}
    pool_mode_c = pool_mode_t = pool_non_c = pool_non_t = 0

    for model, fn in PRED_FILES.items():
        pq = json.load(open(sd / "modal_data" / fn))["per_question_results"]
        mode_c = mode_t = non_c = non_t = 0
        used = 0
        for qid, q in pq.items():
            key = (model, str(qid))
            if key not in acc_by_key:
                continue
            responses = q["base_responses"][:6]
            acc = acc_by_key[key]
            if len(responses) != len(acc) or len(responses) < 2:
                continue
            used += 1
            norm = [normalize_answer(x) for x in responses]
            counts = Counter(norm)
            max_count = max(counts.values())
            mode_strings = {s for s, c in counts.items() if c == max_count}
            for i, ns in enumerate(norm):
                if ns in mode_strings:
                    mode_t += 1
                    mode_c += acc[i]
                else:
                    non_t += 1
                    non_c += acc[i]
        mode_acc = mode_c / mode_t if mode_t else float("nan")
        non_acc = non_c / non_t if non_t else float("nan")
        ratio = mode_acc / non_acc if non_acc else float("nan")
        per_model[model] = dict(
            used_q=used, mode_c=mode_c, mode_t=mode_t, non_c=non_c, non_t=non_t,
            mode_acc=round(mode_acc, 4), non_acc=round(non_acc, 4),
            ratio=round(ratio, 3), rej_rate=round(non_t / (mode_t + non_t), 5),
        )
        pool_mode_c += mode_c; pool_mode_t += mode_t
        pool_non_c += non_c; pool_non_t += non_t

    pooled = dict(
        mode_c=pool_mode_c, mode_t=pool_mode_t, non_c=pool_non_c, non_t=pool_non_t,
        mode_acc=round(pool_mode_c / pool_mode_t, 4),
        non_acc=round(pool_non_c / pool_non_t, 4),
        ratio=round((pool_mode_c / pool_mode_t) / (pool_non_c / pool_non_t), 3),
        rej_rate=round(pool_non_t / (pool_mode_t + pool_non_t), 5),
    )

    print(f"{'model':<9} {'q':>4} {'mode_t':>7} {'non_t':>6} {'mode_acc':>9} {'non_acc':>8} {'ratio':>6} {'rej':>7}")
    for m, v in per_model.items():
        print(f"{m:<9} {v['used_q']:>4} {v['mode_t']:>7} {v['non_t']:>6} "
              f"{v['mode_acc']:>9} {v['non_acc']:>8} {v['ratio']:>6} {v['rej_rate']:>7}")
    print(f"{'POOLED':<9} {'':>4} {pooled['mode_t']:>7} {pooled['non_t']:>6} "
          f"{pooled['mode_acc']:>9} {pooled['non_acc']:>8} {pooled['ratio']:>6} {pooled['rej_rate']:>7}")

    print("\nStored JSON claims: pooled ratio 2.367 @ 10.708% rej; "
          "Qwen 3.184 / Mistral 2.618 / Llama 2.183 / Gemma 1.780")


if __name__ == "__main__":
    main()
