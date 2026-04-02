"""
训练工具函数集合
"""
import math
import os
import random
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoModel, AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, ResidualMixer


def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, "n_routed_experts", getattr(config, "num_experts", 0))
    n_active = getattr(config, "num_experts_per_tok", 0)
    n_shared = getattr(config, "n_shared_experts", 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.experts.0." in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if "mlp.shared_experts.0." in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total:
        Logger(f"Model Params: {total:.2f}M-A{active:.2f}M")
    else:
        Logger(f"Model Params: {total:.2f}M")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_residual_args(parser):
    parser.add_argument("--residual_mode", default="standard", choices=["standard", "attnres_full", "attnres_block"], help="残差模式")
    parser.add_argument("--attnres_block_size", default=4, type=int, help="Block AttnRes 每组残差分支数")
    parser.add_argument("--attnres_use_output_norm", default=1, type=int, choices=[0, 1], help="AttnRes 最终输出是否做RMSNorm")
    parser.add_argument("--attnres_collect_stats", default=0, type=int, choices=[0, 1], help="是否记录AttnRes统计信息")
    parser.add_argument("--attnres_lr_scale", default=2.0, type=float, help="AttnRes参数组学习率缩放")
    return parser


def build_lm_config(args, **overrides):
    config_kwargs = {
        "hidden_size": getattr(args, "hidden_size", 768),
        "num_hidden_layers": getattr(args, "num_hidden_layers", 8),
        "use_moe": bool(getattr(args, "use_moe", 0)),
        "residual_mode": getattr(args, "residual_mode", "standard"),
        "attnres_block_size": getattr(args, "attnres_block_size", 4),
        "attnres_use_output_norm": bool(getattr(args, "attnres_use_output_norm", 1)),
        "attnres_collect_stats": bool(getattr(args, "attnres_collect_stats", 0)),
    }
    if hasattr(args, "inference_rope_scaling"):
        config_kwargs["inference_rope_scaling"] = getattr(args, "inference_rope_scaling")
    max_position_embeddings = overrides.pop("max_position_embeddings", None)
    if max_position_embeddings is None:
        max_position_embeddings = getattr(args, "max_position_embeddings", None)
    if max_position_embeddings is None and hasattr(args, "max_seq_len"):
        max_position_embeddings = getattr(args, "max_seq_len")
    if max_position_embeddings is not None:
        config_kwargs["max_position_embeddings"] = max_position_embeddings
    config_kwargs.update(overrides)
    return MiniMindConfig(**config_kwargs)


def get_residual_suffix(lm_config):
    mode = getattr(lm_config, "residual_mode", "standard")
    if mode == "standard":
        return ""
    suffix = f"_{mode}"
    if mode == "attnres_block":
        suffix += f"_b{getattr(lm_config, 'attnres_block_size', 4)}"
    return suffix


def get_model_suffix(lm_config):
    moe_suffix = "_moe" if lm_config.use_moe else ""
    return f"{moe_suffix}{get_residual_suffix(lm_config)}"


def get_weight_path(save_dir, weight, lm_config):
    return f"{save_dir}/{weight}_{lm_config.hidden_size}{get_model_suffix(lm_config)}.pth"


def get_resume_path(save_dir, weight, lm_config):
    return f"{save_dir}/{weight}_{lm_config.hidden_size}{get_model_suffix(lm_config)}_resume.pth"


def get_legacy_weight_path(save_dir, weight, lm_config):
    moe_suffix = "_moe" if lm_config.use_moe else ""
    return f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}.pth"


def get_legacy_resume_path(save_dir, weight, lm_config):
    moe_suffix = "_moe" if lm_config.use_moe else ""
    return f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_suffix}_resume.pth"


def resolve_weight_path(save_dir, weight, lm_config):
    weight_path = get_weight_path(save_dir, weight, lm_config)
    if os.path.exists(weight_path):
        return weight_path
    legacy_path = get_legacy_weight_path(save_dir, weight, lm_config)
    if getattr(lm_config, "residual_mode", "standard") == "standard" and os.path.exists(legacy_path):
        return legacy_path
    raise FileNotFoundError(f"Weight file not found: {weight_path}")


def resolve_resume_path(save_dir, weight, lm_config):
    resume_path = get_resume_path(save_dir, weight, lm_config)
    if os.path.exists(resume_path):
        return resume_path
    legacy_path = get_legacy_resume_path(save_dir, weight, lm_config)
    if getattr(lm_config, "residual_mode", "standard") == "standard" and os.path.exists(legacy_path):
        return legacy_path
    return None


def unwrap_model(model):
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(raw_model, "_orig_mod", raw_model)


def get_attnres_param_groups(model, learning_rate, residual_lr_scale=1.0, weight_decay=0.0):
    raw_model = unwrap_model(model)
    residual_param_ids = set()
    for module in raw_model.modules():
        if isinstance(module, ResidualMixer):
            for param in module.parameters(recurse=True):
                if param.requires_grad:
                    residual_param_ids.add(id(param))

    base_params, residual_params = [], []
    for param in raw_model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in residual_param_ids:
            residual_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": learning_rate, "weight_decay": weight_decay})
    if residual_params:
        param_groups.append({"params": residual_params, "lr": learning_rate * residual_lr_scale, "weight_decay": 0.0})
    return param_groups


def build_optimizer(model, learning_rate, residual_lr_scale=1.0, weight_decay=0.0):
    return torch.optim.AdamW(
        get_attnres_param_groups(
            model=model,
            learning_rate=learning_rate,
            residual_lr_scale=residual_lr_scale,
            weight_decay=weight_decay,
        ),
        lr=learning_rate,
    )


def ensure_supported_rollout_engine(lm_config, rollout_engine):
    if rollout_engine == "sglang" and getattr(lm_config, "residual_mode", "standard") != "standard":
        raise ValueError("SGLang rollout 暂不支持 Attention Residuals，请使用 --rollout_engine torch。")


def lm_checkpoint(lm_config, weight="full_sft", model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir="../checkpoints", **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    ckp_path = get_weight_path(save_dir, weight, lm_config)
    resume_path = get_resume_path(save_dir, weight, lm_config)

    if model is not None:
        raw_model = unwrap_model(model)
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + ".tmp"
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        wandb_id = None
        if wandb:
            if hasattr(wandb, "get_run"):
                run = wandb.get_run()
                wandb_id = getattr(run, "id", None) if run else None
            else:
                wandb_id = getattr(wandb, "id", None)

        resume_data = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "wandb_id": wandb_id,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if hasattr(value, "state_dict"):
                raw_value = unwrap_model(value)
                resume_data[key] = raw_value.state_dict()
            else:
                resume_data[key] = value

        resume_tmp = resume_path + ".tmp"
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        resolved_resume_path = resolve_resume_path(save_dir, weight, lm_config)
        if resolved_resume_path and os.path.exists(resolved_resume_path):
            ckp_data = torch.load(resolved_resume_path, map_location="cpu")
            saved_ws = ckp_data.get("world_size", 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data["step"] = ckp_data["step"] * saved_ws // current_ws
                Logger(f"GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data['step']}")
            return ckp_data
        return None


def init_model(lm_config, from_weight="pretrain", tokenizer_path="../model", save_dir="../out", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)
    if from_weight != "none":
        weight_path = resolve_weight_path(save_dir, from_weight, lm_config)
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
    get_model_params(model, lm_config)
    Logger(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]["content"] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
