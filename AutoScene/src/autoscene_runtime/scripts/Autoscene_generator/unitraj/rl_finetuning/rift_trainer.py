
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import AdamW


logger = logging.getLogger(__name__)


class RIFTTrainer(LightningModule):
    """Scorer-only PPO-style fine-tuner for Pluto candidate logits."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.05,
        trainable_layers: Optional[Iterable[str]] = None,
        kl_warmup_steps: int = 20,
        kl_warn: float = 0.15,
        kl_stop: float = 0.30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.trainable_layers = list(trainable_layers or ["planning_decoder.pi_head"])
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.kl_warn = float(kl_warn)
        self.kl_stop = float(kl_stop)
        self._stop_update = False

        self._freeze_parameters()
        self._set_model_mode()

    def _freeze_parameters(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        modules = dict(self.model.named_modules())
        for layer_name in self.trainable_layers:
            layer = modules.get(layer_name, None)
            if layer is None:
                logger.warning("Layer %s not found for fine-tuning", layer_name)
                continue
            for param in layer.parameters():
                param.requires_grad = True

    def _set_model_mode(self) -> None:
        # Keep the frozen backbone in eval mode to avoid rollout/train mismatch.
        self.model.eval()

    def forward(self, pluto_feature):
        self._set_model_mode()
        data = pluto_feature.data if hasattr(pluto_feature, "data") else pluto_feature
        return self.model(data)

    def _compute_policy_loss(self, probability: torch.Tensor, batch: Dict) -> torch.Tensor:
        bs, _, _ = probability.shape
        valid_mask = batch["group_advantage_mask_torch"].bool()
        valid_mask_flat = valid_mask.view(bs, -1)
        num_valid = valid_mask_flat.sum()
        if num_valid <= 0:
            return probability.sum() * 0.0

        new_logits = probability.masked_fill(~valid_mask, -50.0)
        old_logits = batch["old_group_logits_torch"].masked_fill(~valid_mask, -50.0)
        new_log_prob = F.log_softmax(new_logits.view(bs, -1), dim=1)
        old_log_prob = F.log_softmax(old_logits.view(bs, -1), dim=1)
        adv = batch["group_advantage_torch"].view(bs, -1)
        adv_valid = torch.clamp(adv[valid_mask_flat], min=-5.0, max=5.0)

        logp_diff = (new_log_prob - old_log_prob).clamp(-10.0, 10.0)
        ratio = torch.exp(logp_diff)
        ratio_valid = ratio[valid_mask_flat]

        clip_lower = 1.0 - self.clip_ratio
        clip_upper = 1.0 + self.clip_ratio
        surr1 = ratio_valid * adv_valid
        surr2 = torch.clamp(ratio_valid, clip_lower, clip_upper) * adv_valid
        min_surr = torch.min(surr1, surr2)
        dual_clip = 2.0 * adv_valid
        objective = torch.where(
            adv_valid >= 0,
            min_surr,
            torch.max(min_surr, dual_clip),
        )
        policy_loss = -objective.mean()

        prob = torch.softmax(new_logits.view(bs, -1), dim=1)
        entropy = -(prob * new_log_prob).sum(dim=1).mean()
        loss = policy_loss - self.entropy_coef * entropy

        kl = ((ratio - 1.0) - logp_diff)[valid_mask_flat].mean()
        if self.training:
            self.log("train/ratio_mean", ratio_valid.mean(), on_step=True, on_epoch=True)
            self.log("train/entropy", entropy, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/kl_mean", kl, on_step=True, on_epoch=True, prog_bar=True)
            if float(kl.detach().cpu()) > self.kl_warn:
                logger.warning("[RIFT] KL warn: %.3f > %.3f", float(kl.detach().cpu()), self.kl_warn)
            if int(self.global_step) >= self.kl_warmup_steps and float(kl.detach().cpu()) > self.kl_stop:
                logger.warning("[RIFT] KL stop triggered at step %s", int(self.global_step))
                self._stop_update = True
        return loss

    def training_step(self, batch: Dict, batch_idx: int):
        if self._stop_update:
            self.log("train/kl_early_stopped", 1.0)
            return torch.zeros((), device=self.device, requires_grad=True)

        feature = batch["cur_pluto_feature_torch"]
        if hasattr(feature, "data") and "_inference" in feature.data:
            feature.data["_inference"] = False
        output = self(feature)
        probability = output.get("probability", None)
        if probability is None:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            self.log("train/loss", zero, on_step=True, on_epoch=True, prog_bar=True)
            return zero

        loss = self._compute_policy_loss(probability, batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        feature = batch["cur_pluto_feature_torch"]
        output = self(feature)
        probability = output.get("probability", None)
        if probability is None:
            return torch.zeros((), device=self.device)
        loss = self._compute_policy_loss(probability, batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        return AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)




