"""AdaLN 动作注入(离散 id 版,ABot 验证臂)。

注入点与 AlayaWorld ActionAdaLN 完全同构:在 adaln_single 的输出
(timestep_emb [*,9d], embedded_timestep [*,d]) 上逐潜帧加 (a0, a);
区别仅在动作表征:他们是连续 6D 相对位姿正弦编码,我们是离散
(move_id, view_id) 的 learned embedding(9 移动 x 9 相机)。

零初始化 mlp 末层 => (a, a0) 从 0 起步,训练开始时对底模零扰动;
直接加在 embedded_timestep 上的通路给 mlp 末层供梯度,两三步后整体激活。

HOLDER 由 strategy(训练)或 probe(推理)在每次 forward 前填充:
  HOLDER['ids']    : [B, F, 2] long  (move_id, view_id per latent frame)
  HOLDER['module'] : 实际调用的 embedder(训练时是 DDP 包装后的)
wrapper 按 total % (B*F) == 0 守卫:形状对不上(如全局标量 timestep 或
prompt 路径)则跳过注入。
"""
import torch
import torch.nn as nn

HOLDER = {}


class ActionAdaLNEmbedder(nn.Module):
    def __init__(self, dim: int = 4096, coefficient: int = 9,
                 n_move: int = 9, n_view: int = 9):
        super().__init__()
        self.move_emb = nn.Embedding(n_move, dim)
        self.view_emb = nn.Embedding(n_view, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, coefficient * dim)
        nn.init.normal_(self.move_emb.weight, std=0.02)
        nn.init.normal_(self.view_emb.weight, std=0.02)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.zeros_(self.proj.bias)

    def forward(self, ids: torch.Tensor):
        """ids [B,F,2] -> (a [B,F,d], a0 [B,F,coef*d])"""
        h = self.move_emb(ids[..., 0]) + self.view_emb(ids[..., 1])
        a = self.mlp(h)
        a0 = self.proj(a)
        return a, a0


def install_action_adaln(transformer) -> ActionAdaLNEmbedder:
    """Wrap transformer.adaln_single.forward(视频 token 路径专属实例)。
    返回未包装的 embedder(参数注册/存盘用);实际前向经 HOLDER['module']。"""
    if hasattr(transformer, "adaln_single"):
        adaln = transformer.adaln_single
    else:
        # 推理时模型可能被 X0Model 等包装:按名搜索视频路径的 adaln_single
        cands = [(n, m) for n, m in transformer.named_modules()
                 if n.split(".")[-1] == "adaln_single"
                 and not any(x in n for x in ("prompt", "audio", "av_ca"))]
        assert len(cands) == 1, f"adaln_single 定位失败: {[n for n, _ in cands]}"
        print(f"[ltxwm-AdaLN] found adaln_single at '{cands[0][0]}'", flush=True)
        adaln = cands[0][1]
    dim = adaln.linear.in_features
    coefficient = adaln.linear.out_features // dim
    emb = ActionAdaLNEmbedder(dim, coefficient)
    orig_forward = adaln.forward
    logged = [False]
    ncall = [0]

    def wrapped(timestep, hidden_dtype=None):
        ts_emb, emb_t = orig_forward(timestep, hidden_dtype=hidden_dtype)
        ids = HOLDER.get("ids")
        ncall[0] += 1
        if ncall[0] <= 6 and HOLDER.get("debug"):
            print(f"[ltxwm-AdaLN] call#{ncall[0]} timestep={tuple(timestep.shape)} "
                  f"rows={ts_emb.shape[0]}", flush=True)
        if ids is not None and HOLDER.get("seq_cfg"):
            # 串行 CFG(one-stage 推理管线):每去噪步两次 forward,各 B=1。
            # ids [2,F,2] = [pos, neg];奇数调用 = cond(pos),偶数 = uncond(neg)。
            ids = ids[0:1] if ncall[0] % 2 == 1 else ids[1:2]
        if ids is not None:
            B, F = ids.shape[0], ids.shape[1]
            total = ts_emb.shape[0]
            if total % B == 0 and (total // B) % F == 0 and total // B > 1:
                mod = HOLDER.get("module", emb)
                a, a0 = mod(ids.to(ts_emb.device))
                rep = total // B // F
                a = a.repeat_interleave(rep, dim=1).reshape(total, -1)
                a0 = a0.repeat_interleave(rep, dim=1).reshape(total, -1)
                ts_emb = ts_emb + a0.to(ts_emb.dtype)
                emb_t = emb_t + a.to(emb_t.dtype)
                if not logged[0]:
                    logged[0] = True
                    print(f"[ltxwm-AdaLN] inject: ids={tuple(ids.shape)} rep={rep} "
                          f"ts_emb={tuple(ts_emb.shape)} emb_t={tuple(emb_t.shape)}",
                          flush=True)
        return ts_emb, emb_t

    adaln.forward = wrapped
    print(f"[ltxwm-AdaLN] installed on adaln_single (dim={dim}, coef={coefficient})",
          flush=True)
    return emb
