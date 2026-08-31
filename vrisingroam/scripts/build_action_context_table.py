"""Build the [33, L, 4096] UMT5 action-sentence table for per-frame CA.

Encodes the 32 combo sentences + neutral with the EXACT same T5 path as the
training cache (encode_prompt_bitexact), truncates to TABLE_LEN tokens after
verifying every sentence's true token count fits, and saves a .pt with a
sentence-list SHA so train/inference can verify byte-stability.

Usage:  . env.sh && python scripts/build_action_context_table.py \
            --model_paths "$DIT_JSON" --tokenizer_path "$TOKENIZER" \
            --out data/processed/action_context_table_v1.pt
"""
import argparse
import hashlib

import torch

from ReactiveGWM_Code.training.data.action_text import (
    ALL_SENTENCES, ALL_SENTENCES_V4, TABLE_LEN,
)
from ReactiveGWM_Code.training.bidirectional.precompute_cache import (
    load_pipeline, encode_prompt_bitexact,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_paths", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--v4", action="store_true", help="encode the 41-sentence v4 set (incl. pushes)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    pipe = load_pipeline(args.model_paths, args.tokenizer_path,
                         args.device, torch.bfloat16)

    sentences = ALL_SENTENCES_V4 if args.v4 else ALL_SENTENCES
    rows = []
    for i, sent in enumerate(sentences):
        torch.cuda.empty_cache()
        ids, mask = pipe.tokenizer(sent, return_mask=True,
                                   add_special_tokens=True)
        true_len = int(mask.gt(0).sum())
        assert true_len <= TABLE_LEN, \
            f"sentence {i} needs {true_len} tokens > TABLE_LEN={TABLE_LEN}: {sent!r}"
        with torch.no_grad():
            emb = encode_prompt_bitexact(pipe, sent)  # [1, 512, 4096], zero-padded
        rows.append(emb[0, :TABLE_LEN].float().cpu())
        print(f"[{i:2d}] len={true_len:2d}  {sent}")

    table = torch.stack(rows)                        # [33, TABLE_LEN, 4096]
    sha = hashlib.sha256("\n".join(sentences).encode()).hexdigest()
    torch.save({"table": table, "sentences": sentences,
                "sentences_sha": sha, "table_len": TABLE_LEN}, args.out)
    print(f"saved {args.out}  shape={tuple(table.shape)}  sha={sha[:12]}")


if __name__ == "__main__":
    main()
