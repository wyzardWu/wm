"""Entry: LTX-2.3 LoRA training with per-frame action context.

Builds the stock ltx_trainer.Trainer from a YAML config, then:
  1. swaps the training strategy for ActionT2VStrategy (caption ⊕ per-frame action)
  2. installs the frame-fold cross-attention patch on the loaded transformer

Usage:
  python train_action.py --config smoke64.yaml \
      [--tables /data/yuzhewu/ltxwm/tables/ltx_action_tables.pt]
"""
import argparse
import sys

sys.path.insert(0, "/data/yuzhewu/ltxwm")

from ltxwm.action_strategy import ACT_LEN, NUM_FRAMES, ActionT2VStrategy
from ltxwm.frame_context_patch import install_frame_context

CAPTION_LEN = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables", default="/data/yuzhewu/ltxwm/tables/ltx_action_tables.pt")
    ap.add_argument("--stock", action="store_true", help="baseline: no strategy swap, no patch")
    ap.add_argument("--inject", choices=["fold", "adaln", "wwfold"], default="fold",
                    help="fold=ABot 逐帧槽; adaln=离散 id AdaLN 臂; wwfold=WildWorld 三槽")
    args = ap.parse_args()

    from ltx_trainer.config import LtxTrainerConfig
    from ltx_trainer.trainer import LtxvTrainer

    import yaml
    with open(args.config) as f:
        cfg = LtxTrainerConfig(**yaml.safe_load(f))
    trainer = LtxvTrainer(cfg)

    if args.stock:
        print("[ltxwm] STOCK baseline mode: no swap, no patch", flush=True)
        trainer.train()
        return

    # 1. strategy swap (keep the stock config object the factory built with)
    stock = trainer._training_strategy
    if args.inject == "adaln":
        from ltxwm.adaln_strategy import AdaLNActionStrategy
        trainer._training_strategy = AdaLNActionStrategy(stock.config)
    elif args.inject == "wwfold":
        from ltxwm.ww_strategy import WWActionStrategy
        trainer._training_strategy = WWActionStrategy(stock.config, tables_path=args.tables)
    else:
        trainer._training_strategy = ActionT2VStrategy(stock.config, tables_path=args.tables)
    print("[ltxwm] scale_factors copied:", hasattr(stock, "video_scale_factors"), flush=True)
    if hasattr(stock, "video_scale_factors"):
        trainer._training_strategy.video_scale_factors = stock.video_scale_factors

    if args.inject == "adaln":
        # 2b. AdaLN 注入:embedder 参数补进优化器、单独 DDP、checkpoint 附带存
        import os
        import torch
        from ltxwm.action_adaln_patch import HOLDER, install_action_adaln
        emb = install_action_adaln(trainer._accelerator.unwrap_model(trainer._transformer))
        emb = emb.to(trainer._accelerator.device)
        emb_prepared = trainer._accelerator.prepare(emb)
        HOLDER["module"] = emb_prepared
        trainer._trainable_params = list(trainer._trainable_params) + [
            p for p in emb.parameters() if p.requires_grad]
        n_emb = sum(p.numel() for p in emb.parameters())
        print(f"[ltxwm] AdaLN embedder params: {n_emb:,} (fp32, in optimizer)", flush=True)

        _orig_save = trainer._save_checkpoint
        def _save_with_embedder():
            path = _orig_save()
            if trainer._accelerator.is_main_process:
                sd = trainer._accelerator.unwrap_model(emb_prepared).state_dict()
                out = os.path.join(cfg.output_dir, "checkpoints",
                                   f"action_adaln_step_{trainer._global_step:05d}.pt")
                torch.save(sd, out)
                print(f"[ltxwm] AdaLN embedder saved: {out}", flush=True)
            return path
        trainer._save_checkpoint = _save_with_embedder
    else:
        # 2. frame-fold patch on the (already loaded) transformer
        act_len = ACT_LEN
        if args.inject == "wwfold":
            from ltxwm.ww_strategy import WW_ACT_LEN
            act_len = WW_ACT_LEN
        n = install_frame_context(trainer._transformer, NUM_FRAMES, CAPTION_LEN + act_len)
        print(f"[ltxwm] frame-fold installed on {n} blocks "
              f"(F={NUM_FRAMES}, ctx_len={CAPTION_LEN + act_len})", flush=True)
        assert n > 0, "no BasicAVTransformerBlock found — check model class"

    # line-logging: rich progress is invisible under nohup; print every step
    _orig_step = trainer._training_step
    _n = [0]
    def _logged_step(batch):
        out = _orig_step(batch)
        _n[0] += 1
        loss = getattr(out, "loss", None)
        val = float(loss) if loss is not None else float("nan")
        print(f"[step {_n[0]}] loss_sum={val:.1f} loss_per_el={val/798720:.4f}", flush=True)
        return out
    trainer._training_step = _logged_step

    trainer.train()


if __name__ == "__main__":
    main()
