# ==================================================================
# my_custom_module3.py
# 
# v3 改进：
#   - 去掉 alpha（冗余）
#   - Gate bias=-2.0（保守初始化）
#   - LeakyReLU 替代 ReLU
#   - GroupNorm 替代 BN（encoder 中）
#   - 去掉 clone()
#   - 新增 FreqFusionStatsHook（全图统计）
# ==================================================================

import math
import os
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

from mmcv.transforms import BaseTransform
from mmengine.hooks import Hook
from mmdet.registry import TRANSFORMS, MODELS, HOOKS
from mmdet.models.detectors.grounding_dino import GroundingDINO


# ==================================================================
# 0. LoRA
# ==================================================================

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8,
                 lora_alpha=16, lora_dropout=0.0, bias=True,
                 base_layer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = int(r)
        self.lora_alpha = int(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 1.0
        self.lora_drop = (nn.Dropout(p=lora_dropout)
                          if lora_dropout > 0 else nn.Identity())
        if base_layer is not None:
            self.weight = nn.Parameter(
                base_layer.weight.data.clone(), requires_grad=False)
            self.bias = (nn.Parameter(base_layer.bias.data.clone(),
                                      requires_grad=False)
                         if base_layer.bias is not None else None)
        else:
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features),
                requires_grad=False)
            self.bias = (nn.Parameter(torch.empty(out_features),
                                      requires_grad=False) if bias else None)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                nn.init.uniform_(self.bias, -1/math.sqrt(fan_in),
                                 1/math.sqrt(fan_in))
        if self.r > 0:
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = self.lora_B = None

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            result = result + self.lora_B(
                self.lora_A(self.lora_drop(x))) * self.scaling
        return result


# ==================================================================
# 1. 频域先验模块（v3: LeakyReLU + GroupNorm）
# ==================================================================

class FreqPriorModule(nn.Module):
    def __init__(self, out_channels=256, center_radius=30,
                 point_radius=8, outlier_percentile=0.5,
                 smooth_sigma=20, num_peaks=3,
                 suppress_radius=None, border_ratio=0.03,
                 prior_type='fft'):
        super().__init__()
        if prior_type not in ('fft', 'gray'):
            raise ValueError(f'Unsupported prior_type: {prior_type}')
        self.prior_type = prior_type
        self.center_radius = center_radius
        self.point_radius = point_radius
        self.outlier_percentile = outlier_percentile
        self.smooth_sigma = smooth_sigma
        self.num_peaks = num_peaks
        self.suppress_radius = suppress_radius or (point_radius + 3)
        self.border_ratio = border_ratio

        ks = int(6 * smooth_sigma + 1)
        if ks % 2 == 0:
            ks += 1
        self.smooth_kernel_size = ks
        gk = self._make_gaussian_kernel(ks, smooth_sigma)
        self.register_buffer('gaussian_kernel', gk)

        # v3 改进：LeakyReLU + GroupNorm
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    @staticmethod
    def _make_gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    @staticmethod
    def _build_soft_mask(H, W, margin_h, margin_w, device):
        weight = torch.ones(H, W, device=device)
        for i in range(margin_h):
            t = i / max(margin_h, 1)
            weight[i, :] *= t
            weight[H - 1 - i, :] *= t
        for j in range(margin_w):
            t = j / max(margin_w, 1)
            weight[:, j] *= t
            weight[:, W - 1 - j] *= t
        return weight

    @torch.no_grad()
    def _compute_heatmap_batch(self, images):
        B, _, H, W = images.shape
        device = images.device
        gray = (0.299 * images[:, 0] + 0.587 * images[:, 1]
                + 0.114 * images[:, 2])
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)

        freq = torch.fft.fft2(gray)
        freq_shift = torch.fft.fftshift(freq, dim=(-2, -1))
        magnitude = torch.abs(freq_shift)
        cy, cx = H // 2, W // 2
        yy = torch.arange(H, device=device).view(-1, 1).float()
        xx = torch.arange(W, device=device).view(1, -1).float()
        dc_mask = (yy - cy)**2 + (xx - cx)**2 < self.center_radius**2

        freq_filtered = freq_shift.clone()
        for b in range(B):
            mag_b = magnitude[b].clone()
            mag_b[dc_mask] = 0
            # ``num_peaks`` counts independent Fourier peak pairs. The
            # conjugate-symmetric location is suppressed together with each
            # selected peak.
            for _ in range(self.num_peaks):
                flat_idx = torch.argmax(mag_b)
                py, px = flat_idx // W, flat_idx % W
                pd = (yy - py.float())**2 + (xx - px.float())**2
                freq_filtered[b][pd <= self.point_radius**2] = 0
                sym_py = (2 * cy - py) % H
                sym_px = (2 * cx - px) % W
                spd = (yy - sym_py.float())**2 + (xx - sym_px.float())**2
                freq_filtered[b][spd <= self.point_radius**2] = 0
                mag_b[pd <= self.suppress_radius**2] = 0
                mag_b[spd <= self.suppress_radius**2] = 0

        freq_ishift = torch.fft.ifftshift(freq_filtered, dim=(-2, -1))
        residual = torch.abs(torch.fft.ifft2(freq_ishift))

        mh = max(int(H * self.border_ratio), 1)
        mw = max(int(W * self.border_ratio), 1)
        pixel_scores = torch.zeros_like(residual)
        for b in range(B):
            res_b = residual[b]
            interior = res_b[mh:H-mh, mw:W-mw].flatten()
            lower = torch.quantile(interior, self.outlier_percentile/100.0)
            upper = torch.quantile(interior, 1.0 - self.outlier_percentile/100.0)
            pixel_scores[b][res_b > upper] = res_b[res_b > upper] - upper
            pixel_scores[b][res_b < lower] = lower - res_b[res_b < lower]

        soft_mask = self._build_soft_mask(H, W, mh, mw, device)
        pixel_scores = pixel_scores * soft_mask.unsqueeze(0)
        for b in range(B):
            mv = pixel_scores[b].max()
            if mv > 1e-8:
                pixel_scores[b] = pixel_scores[b] / mv

        heatmaps = pixel_scores.unsqueeze(1)
        pad = self.smooth_kernel_size // 2
        heatmaps = F.pad(heatmaps, (pad, pad, pad, pad), mode='reflect')
        heatmaps = F.conv2d(heatmaps, self.gaussian_kernel)
        for b in range(B):
            mv = heatmaps[b].max()
            if mv > 1e-8:
                heatmaps[b] = heatmaps[b] / mv
        return heatmaps

    @torch.no_grad()
    def _compute_grayscale_prior_batch(self, images):
        gray = (0.299 * images[:, 0] + 0.587 * images[:, 1]
                + 0.114 * images[:, 2])
        gray = gray.unsqueeze(1)
        B = gray.shape[0]
        gray_flat = gray.flatten(1)
        lo = gray_flat.min(dim=1)[0].view(B, 1, 1, 1)
        hi = gray_flat.max(dim=1)[0].view(B, 1, 1, 1)
        return (gray - lo) / (hi - lo + 1e-8)

    @torch.no_grad()
    def _compute_prior_batch(self, images):
        if self.prior_type == 'gray':
            return self._compute_grayscale_prior_batch(images)
        return self._compute_heatmap_batch(images)

    def forward(self, images, target_sizes):
        priors = self._compute_prior_batch(images)
        feat = self.encoder(priors)
        multi_scale = []
        for (th, tw) in target_sizes:
            f = F.interpolate(feat, size=(th, tw),
                              mode='bilinear', align_corners=False)
            multi_scale.append(f)
        return multi_scale


# ==================================================================
# 2. 门控融合模块（v3: 去掉 alpha，bias=-2.0）
# ==================================================================

class GatedFusionModule(nn.Module):
    """
    v3: F_out = F_v + gate × GroupNorm(F_f)
    去掉 alpha，gate 自身控制融合强度
    bias=-2.0 → sigmoid(-2)≈0.12，保守起步
    """
    def __init__(self, channels=256, num_levels=4):
        super().__init__()
        self.freq_norms = nn.ModuleList([
            nn.GroupNorm(32, channels) for _ in range(num_levels)
        ])
        self.gates = nn.ModuleList()
        for _ in range(num_levels):
            gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 1, 1, bias=True),
                nn.Sigmoid(),
            )
            # v3: 保守初始化
            nn.init.zeros_(gate[3].weight)
            nn.init.constant_(gate[3].bias, -2.0)
            self.gates.append(gate)

    def forward(self, visual_feats, freq_feats):
        fused = []
        for v, f, gate_layer, freq_norm in zip(
                visual_feats, freq_feats, self.gates, self.freq_norms):
            f = freq_norm(f)
            gate = gate_layer(torch.cat([v, f], dim=1))
            fused.append(v + gate * f)
        return tuple(fused)


class ResidualFusionModule(nn.Module):
    """Ablation: inject normalized prior features without a learned gate."""

    def __init__(self, channels=256, num_levels=4):
        super().__init__()
        self.freq_norms = nn.ModuleList([
            nn.GroupNorm(32, channels) for _ in range(num_levels)
        ])

    def forward(self, visual_feats, freq_feats):
        fused = []
        for v, f, freq_norm in zip(visual_feats, freq_feats, self.freq_norms):
            fused.append(v + freq_norm(f))
        return tuple(fused)


# ==================================================================
# 3. Transforms
# ==================================================================

@TRANSFORMS.register_module()
class FixTextForVal(BaseTransform):
    def __init__(self, text='defect .'):
        self.text = text
    def transform(self, results):
        if 'text' not in results or not results['text']:
            results['text'] = self.text
        return results

@TRANSFORMS.register_module(force=True)
class LoadTokensPositive(BaseTransform):
    def transform(self, results):
        if 'phrases' in results:
            phrases = results['phrases']
            if isinstance(phrases, dict) and len(phrases) > 0:
                max_label = max(int(k) for k in phrases.keys())
                tokens_list = [[] for _ in range(max_label + 1)]
                for k, v in phrases.items():
                    tokens_list[int(k)] = v.get('tokens_positive', [])
                results['tokens_positive'] = tokens_list
                return results
        tokens_list = []
        if 'instances' in results:
            for inst in results['instances']:
                tokens_list.append(inst.get('tokens_positive', []))
        if not tokens_list:
            num = len(results.get('gt_bboxes', [])) if 'gt_bboxes' in results else 0
            tokens_list = [[] for _ in range(num)]
        results['tokens_positive'] = tokens_list
        return results


# ==================================================================
# 4. GroundingDINOLoRA（v3: 去掉 clone，去掉 alpha）
# ==================================================================

@MODELS.register_module()
class GroundingDINOLoRA(GroundingDINO):

    def __init__(self, *args, **kwargs):
        # pop 自定义参数
        lora_r = kwargs.pop('lora_r', 8)
        lora_alpha = kwargs.pop('lora_alpha', 16)
        lora_dropout = kwargs.pop('lora_dropout', 0.0)
        lora_target_modules = kwargs.pop('lora_target_modules', None)
        lora_exclude_modules = kwargs.pop('lora_exclude_modules', None)
        lora_backbone_stages = kwargs.pop('lora_backbone_stages', None)
        freeze_non_lora = kwargs.pop('freeze_non_lora', True)
        trainable_patterns = kwargs.pop('trainable_patterns', None)

        use_freq_prior = kwargs.pop('use_freq_prior', False)
        freq_channels = kwargs.pop('freq_channels', 256)
        freq_num_levels = kwargs.pop('freq_num_levels', 4)
        freq_center_radius = kwargs.pop('freq_center_radius', 30)
        freq_point_radius = kwargs.pop('freq_point_radius', 8)
        freq_outlier_percentile = kwargs.pop('freq_outlier_percentile', 0.5)
        freq_smooth_sigma = kwargs.pop('freq_smooth_sigma', 20)
        freq_num_peaks = kwargs.pop('freq_num_peaks', 3)
        freq_border_ratio = kwargs.pop('freq_border_ratio', 0.03)
        freq_prior_type = kwargs.pop('freq_prior_type', 'fft')
        freq_fusion_type = kwargs.pop('freq_fusion_type', 'gated')
        if freq_fusion_type not in ('gated', 'residual'):
            raise ValueError(f'Unsupported freq_fusion_type: {freq_fusion_type}')

        # pop 掉旧版的 alpha 参数（兼容）
        kwargs.pop('freq_init_alpha', None)
        kwargs.pop('freeze_patterns', None)

        self._use_freq_prior = use_freq_prior
        self._freq_cfg = dict(
            out_channels=freq_channels,
            center_radius=freq_center_radius,
            point_radius=freq_point_radius,
            outlier_percentile=freq_outlier_percentile,
            smooth_sigma=freq_smooth_sigma,
            num_peaks=freq_num_peaks,
            border_ratio=freq_border_ratio,
            prior_type=freq_prior_type,
        )
        self._freq_num_levels = freq_num_levels
        self._freq_fusion_type = freq_fusion_type

        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout_rate = float(lora_dropout)
        self.lora_target_modules = lora_target_modules or []
        self.lora_exclude_modules = lora_exclude_modules or ['language_model']
        self.lora_backbone_stages = lora_backbone_stages
        self.freeze_non_lora = bool(freeze_non_lora)

        self.trainable_patterns = trainable_patterns or [
            'encoder.layers', 'encoder.text_layers',
            'encoder.fusion_layers', 'neck', 'decoder',
            'bbox_head', 'query_embedding', 'text_feat_map',
            'memory_trans', 'dn_query_generator', 'level_embed',
            'backbone.norm1', 'backbone.norm2', 'backbone.norm3',
            'freq_prior', 'gated_fusion',
        ]

        super().__init__(*args, **kwargs)

        self.use_freq_prior = self._use_freq_prior
        if self.use_freq_prior:
            self.freq_prior = FreqPriorModule(**self._freq_cfg)
            if self._freq_fusion_type == 'gated':
                self.gated_fusion = GatedFusionModule(
                    channels=self._freq_cfg['out_channels'],
                    num_levels=self._freq_num_levels)
            else:
                self.gated_fusion = ResidualFusionModule(
                    channels=self._freq_cfg['out_channels'],
                    num_levels=self._freq_num_levels)

        self._inject_lora()
        self._apply_freeze_policy()
        self._freeze_applied = True

    def extract_feat(self, batch_inputs):
        # v3: 不用 clone
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        if self.use_freq_prior:
            target_sizes = [feat.shape[-2:] for feat in x]
            freq_feats = self.freq_prior(batch_inputs, target_sizes)
            x = self.gated_fusion(x, freq_feats)
        return x

    def compute_freq_fusion_debug(self, images):
        """Return intermediate tensors for visualization hooks."""
        priors = self.freq_prior._compute_prior_batch(images)
        backbone_feats = self.backbone(images)
        visual_feats = self.neck(backbone_feats) if self.with_neck else backbone_feats
        target_sizes = [f.shape[-2:] for f in visual_feats]
        freq_feats = self.freq_prior(images, target_sizes)

        gates, fused_feats = [], []
        if hasattr(self.gated_fusion, 'gates'):
            for i, (v, f, gl) in enumerate(zip(
                    visual_feats, freq_feats, self.gated_fusion.gates)):
                fn = self.gated_fusion.freq_norms[i](f)
                gate = gl(torch.cat([v, fn], dim=1))
                gates.append(gate)
                fused_feats.append(v + gate * fn)
        else:
            for i, (v, f) in enumerate(zip(visual_feats, freq_feats)):
                fn = self.gated_fusion.freq_norms[i](f)
                gates.append(torch.ones(
                    v.shape[0], 1, v.shape[-2], v.shape[-1],
                    device=v.device, dtype=v.dtype))
                fused_feats.append(v + fn)
        return priors, visual_feats, freq_feats, gates, fused_feats

    def _is_lora_target(self, full_name):
        if not self.lora_target_modules:
            return False
        if any(e in full_name for e in self.lora_exclude_modules):
            return False
        if self.lora_backbone_stages is not None:
            if full_name.startswith('backbone.stages.'):
                parts = full_name.split('.')
                if len(parts) >= 3:
                    try:
                        if int(parts[2]) not in self.lora_backbone_stages:
                            return False
                    except ValueError:
                        return False
        return any(t in full_name for t in self.lora_target_modules)

    def _inject_lora(self):
        if self.lora_r <= 0 or not self.lora_target_modules:
            return
        to_replace = []
        for mn, m in self.named_modules():
            for cn, c in m.named_children():
                if isinstance(c, LoRALinear) or not isinstance(c, nn.Linear):
                    continue
                fn = f"{mn}.{cn}" if mn else cn
                if self._is_lora_target(fn):
                    to_replace.append((m, cn, c))
        for parent, cn, c in to_replace:
            setattr(parent, cn, LoRALinear(
                c.in_features, c.out_features,
                r=self.lora_r, lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout_rate,
                bias=(c.bias is not None), base_layer=c))

    def _apply_freeze_policy(self):
        for p in self.parameters():
            p.requires_grad = False
        for n, p in self.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                p.requires_grad = True
        for n, p in self.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                continue
            if any(pat in n for pat in self.trainable_patterns):
                p.requires_grad = True


# ==================================================================
# 5. Hooks
# ==================================================================

@HOOKS.register_module()
class CheckFreezeHook(Hook):
    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        total, trainable, lora = 0, 0, 0
        for n, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
            if ('lora_A' in n or 'lora_B' in n) and p.requires_grad:
                lora += p.numel()
        runner.logger.info(
            f'[CheckFreeze] Total: {total:,} | '
            f'Trainable: {trainable:,} ({trainable/total:.2%}) | '
            f'LoRA: {lora:,}')
        if getattr(model, 'use_freq_prior', False):
            fp = sum(p.numel() for n, p in model.named_parameters()
                     if ('freq_prior' in n or 'gated_fusion' in n) and p.requires_grad)
            runner.logger.info(f'[CheckFreeze] FreqPrior trainable: {fp:,}')


@HOOKS.register_module()
class FreqFusionStatsHook(Hook):
    """
    每次验证后，在全部验证集上统计融合参数
    输出：
      - 每层 gate 的全局分布（mean, std, min, max, 中位数）
      - 每层融合影响比例
      - 特征尺度比
      - 热力图覆盖率
      - 保存统计图 + JSON
    """

    def __init__(self, save_dir='stats_fusion', stats_interval=1):
        self.save_dir = save_dir
        self.stats_interval = stats_interval

    def after_val_epoch(self, runner, metrics=None):
        self._do_stats(runner, 'val')

    def after_test_epoch(self, runner, metrics=None):
        self._do_stats(runner, 'test')

    def _do_stats(self, runner, mode='val'):
        if not hasattr(self, '_count'):
            self._count = 0
        self._count += 1
        if self._count % self.stats_interval != 0:
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if not getattr(model, 'use_freq_prior', False):
            return

        epoch = getattr(runner, 'epoch', 0)
        save_dir = os.path.join(self.save_dir, f'{mode}_epoch_{epoch+1}')
        os.makedirs(save_dir, exist_ok=True)

        dl = runner.val_dataloader if mode == 'val' else runner.test_dataloader
        logger = runner.logger
        model.eval()

        # 收集统计量
        num_levels = model._freq_num_levels
        all_gate_vals = [[] for _ in range(num_levels)]
        all_fv_std = [[] for _ in range(num_levels)]
        all_ff_std = [[] for _ in range(num_levels)]
        all_impact = [[] for _ in range(num_levels)]
        all_hm_mean = []
        all_hm_max = []
        all_hm_coverage = []
        n_images = 0

        try:
            for batch in dl:
                batch = model.data_preprocessor(batch, False)
                images = batch['inputs']
                B = images.shape[0]

                with torch.no_grad():
                    heatmaps, visual_feats, freq_feats, gates, fused_feats = (
                        model.compute_freq_fusion_debug(images))

                    for lv in range(num_levels):
                        v = visual_feats[lv]
                        f = freq_feats[lv]
                        gate = gates[lv]
                        fused = fused_feats[lv]
                        diff = (fused - v).abs()

                        for b in range(B):
                            all_gate_vals[lv].append(
                                gate[b, 0].cpu().numpy().flatten())
                            all_fv_std[lv].append(v[b].std().item())
                            all_ff_std[lv].append(f[b].std().item())
                            vn = v[b].abs().mean().item()
                            dn = diff[b].mean().item()
                            all_impact[lv].append(
                                dn / (vn + 1e-8) * 100)

                    for b in range(B):
                        hm = heatmaps[b, 0].cpu().numpy()
                        all_hm_mean.append(hm.mean())
                        all_hm_max.append(hm.max())
                        all_hm_coverage.append(
                            (hm > 0.1).sum() / hm.size * 100)

                n_images += B

        except Exception as e:
            logger.error(f'[FusionStats] ERROR: {e}')
            import traceback
            logger.error(traceback.format_exc())
            return

        # ========== 汇总 ==========
        logger.info('=' * 70)
        logger.info(f'[FusionStats] {mode} epoch {epoch+1}: '
                    f'{n_images} images analyzed')
        logger.info('=' * 70)

        stats = {
            'epoch': epoch + 1,
            'mode': mode,
            'n_images': n_images,
            'levels': {},
            'heatmap': {},
        }

        # 每层统计
        for lv in range(num_levels):
            gate_all = np.concatenate(all_gate_vals[lv])
            fv_stds = np.array(all_fv_std[lv])
            ff_stds = np.array(all_ff_std[lv])
            impacts = np.array(all_impact[lv])
            ratios = fv_stds / (ff_stds + 1e-8)

            level_stats = {
                'gate_mean': float(gate_all.mean()),
                'gate_std': float(gate_all.std()),
                'gate_median': float(np.median(gate_all)),
                'gate_min': float(gate_all.min()),
                'gate_max': float(gate_all.max()),
                'gate_q25': float(np.percentile(gate_all, 25)),
                'gate_q75': float(np.percentile(gate_all, 75)),
                'fv_std_mean': float(fv_stds.mean()),
                'ff_std_mean': float(ff_stds.mean()),
                'scale_ratio_mean': float(ratios.mean()),
                'impact_mean': float(impacts.mean()),
                'impact_std': float(impacts.std()),
            }
            stats['levels'][f'P{lv+3}'] = level_stats

            logger.info(
                f'  P{lv+3}: gate={level_stats["gate_mean"]:.4f}±'
                f'{level_stats["gate_std"]:.4f} '
                f'[{level_stats["gate_q25"]:.3f}, '
                f'{level_stats["gate_median"]:.3f}, '
                f'{level_stats["gate_q75"]:.3f}] | '
                f'ratio={level_stats["scale_ratio_mean"]:.1f}x | '
                f'impact={level_stats["impact_mean"]:.2f}%')

        # 热力图统计
        hm_stats = {
            'mean': float(np.mean(all_hm_mean)),
            'max_mean': float(np.mean(all_hm_max)),
            'coverage_mean': float(np.mean(all_hm_coverage)),
            'coverage_std': float(np.std(all_hm_coverage)),
        }
        stats['heatmap'] = hm_stats

        logger.info(
            f'  Heatmap: mean={hm_stats["mean"]:.4f} '
            f'coverage={hm_stats["coverage_mean"]:.1f}±'
            f'{hm_stats["coverage_std"]:.1f}%')
        logger.info('=' * 70)

        # ========== 保存 JSON ==========
        json_path = os.path.join(save_dir, 'stats.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)

        # ========== 画统计图 ==========
        self._plot_stats(stats, all_gate_vals, all_impact,
                         all_hm_coverage, num_levels, save_dir)

        logger.info(f'[FusionStats] 保存到 {save_dir}/')

    def _plot_stats(self, stats, all_gate_vals, all_impact,
                    all_hm_coverage, num_levels, save_dir):

        fig, axes = plt.subplots(2, 3, figsize=(24, 14))

        # (a) 各层 gate 分布箱线图
        gate_data = [np.concatenate(all_gate_vals[lv])
                     for lv in range(num_levels)]
        bp = axes[0, 0].boxplot(
            gate_data,
            labels=[f'P{lv+3}' for lv in range(num_levels)],
            patch_artist=True,
            showfliers=False)
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        for patch, color in zip(bp['boxes'], colors[:num_levels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0, 0].set_ylabel('Gate Value')
        axes[0, 0].set_title('(a) Gate Distribution / Level',
                             fontsize=13)
        axes[0, 0].axhline(y=0.5, color='red', ls='--', alpha=0.5)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(alpha=0.3)

        # (b) 各层 gate 均值柱状图
        means = [stats['levels'][f'P{lv+3}']['gate_mean']
                 for lv in range(num_levels)]
        stds = [stats['levels'][f'P{lv+3}']['gate_std']
                for lv in range(num_levels)]
        bars = axes[0, 1].bar(
            [f'P{lv+3}' for lv in range(num_levels)],
            means, yerr=stds, capsize=5,
            color=colors[:num_levels], alpha=0.8,
            edgecolor='black')
        for bar, val in zip(bars, means):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2,
                            val + 0.03, f'{val:.3f}',
                            ha='center', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_ylabel('Gate Mean ± Std')
        axes[0, 1].set_title('(b) Gate Mean / Level', fontsize=13)
        axes[0, 1].grid(alpha=0.3, axis='y')

        # (c) 融合影响柱状图
        impact_means = [stats['levels'][f'P{lv+3}']['impact_mean']
                        for lv in range(num_levels)]
        impact_stds = [stats['levels'][f'P{lv+3}']['impact_std']
                       for lv in range(num_levels)]
        bars2 = axes[0, 2].bar(
            [f'P{lv+3}' for lv in range(num_levels)],
            impact_means, yerr=impact_stds, capsize=5,
            color=colors[:num_levels], alpha=0.8,
            edgecolor='black')
        for bar, val in zip(bars2, impact_means):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2,
                            val + 0.5, f'{val:.1f}%',
                            ha='center', fontsize=11, fontweight='bold')
        axes[0, 2].set_ylabel('Impact %')
        axes[0, 2].set_title('(c) Fusion Impact / Level', fontsize=13)
        axes[0, 2].grid(alpha=0.3, axis='y')

        # (d) Gate 直方图（P3层）
        axes[1, 0].hist(gate_data[0], bins=50,
                        color='steelblue', alpha=0.7, edgecolor='white')
        axes[1, 0].axvline(x=np.mean(gate_data[0]), color='red',
                           lw=2, label=f'mean={np.mean(gate_data[0]):.3f}')
        axes[1, 0].set_xlabel('Gate Value')
        axes[1, 0].set_ylabel('Pixel Count')
        axes[1, 0].set_title('(d) P3 Gate Histogram', fontsize=13)
        axes[1, 0].legend(fontsize=11)

        # (e) 热力图覆盖率分布
        axes[1, 1].hist(all_hm_coverage, bins=30,
                        color='coral', alpha=0.7, edgecolor='white')
        axes[1, 1].axvline(x=np.mean(all_hm_coverage), color='red',
                           lw=2,
                           label=f'mean={np.mean(all_hm_coverage):.1f}%')
        axes[1, 1].set_xlabel('Coverage %')
        axes[1, 1].set_ylabel('Image Count')
        axes[1, 1].set_title('(e) Heatmap Coverage Distribution',
                             fontsize=13)
        axes[1, 1].legend(fontsize=11)

        # (f) 尺度比
        ratios = [stats['levels'][f'P{lv+3}']['scale_ratio_mean']
                  for lv in range(num_levels)]
        axes[1, 2].bar(
            [f'P{lv+3}' for lv in range(num_levels)],
            ratios, color=colors[:num_levels], alpha=0.8,
            edgecolor='black')
        axes[1, 2].axhline(y=1.0, color='red', ls='--', alpha=0.5,
                           label='ratio=1 (matched)')
        axes[1, 2].set_ylabel('F_v std / F_f std')
        axes[1, 2].set_title('(f) Feature Scale Ratio', fontsize=13)
        axes[1, 2].legend(fontsize=11)
        axes[1, 2].grid(alpha=0.3, axis='y')

        plt.suptitle(
            f'Fusion Statistics - Epoch {stats["epoch"]} '
            f'({stats["n_images"]} images)',
            fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'stats_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


@HOOKS.register_module()
class FreqPriorVisualizeHook(Hook):
    """可视化个别图片的中间结果"""

    def __init__(self, save_dir='vis_freq', vis_interval=1,
                 num_images=8, feat_level=0):
        self.save_dir = save_dir
        self.vis_interval = vis_interval
        self.num_images = num_images
        self.feat_level = feat_level

    def after_val_epoch(self, runner, metrics=None):
        self._do(runner, 'val')

    def after_test_epoch(self, runner, metrics=None):
        self._do(runner, 'test')

    def _do(self, runner, mode):
        if not hasattr(self, '_c'):
            self._c = 0
        self._c += 1
        if self._c % self.vis_interval != 0:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if not getattr(model, 'use_freq_prior', False):
            return
        epoch = getattr(runner, 'epoch', 0)
        save_dir = os.path.join(self.save_dir, f'{mode}_epoch_{epoch+1}')
        os.makedirs(save_dir, exist_ok=True)
        dl = runner.val_dataloader if mode == 'val' else runner.test_dataloader
        model.eval()
        count = 0
        try:
            for batch in dl:
                if count >= self.num_images:
                    break
                batch = model.data_preprocessor(batch, False)
                images = batch['inputs']
                with torch.no_grad():
                    self._vis(model, images, save_dir, count)
                count += images.shape[0]
        except Exception as e:
            runner.logger.error(f'[FreqVis] {e}')
        runner.logger.info(f'[FreqVis] {count} images → {save_dir}')

    @torch.no_grad()
    def _vis(self, model, images, save_dir, start_idx):
        B = images.shape[0]
        lv = self.feat_level

        heatmaps, visual_feats, freq_feats, gates, fused_feats = (
            model.compute_freq_fusion_debug(images))

        for b in range(B):
            fig, axes = plt.subplots(2, 4, figsize=(28, 14))
            img_np = images[b].cpu().float()
            mean_t = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1)
            std_t = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1)
            img_show = (img_np * std_t + mean_t).clamp(0,255).byte().permute(1,2,0).numpy()
            ih, iw = img_show.shape[:2]

            hm = heatmaps[b, 0].cpu().numpy()
            gate_np = gates[lv][b, 0].cpu().numpy()
            v_feat = visual_feats[lv][b].cpu().numpy()
            f_feat = freq_feats[lv][b].cpu().numpy()
            fused_np = fused_feats[lv][b].cpu().numpy()

            axes[0,0].imshow(img_show); axes[0,0].set_title('Input'); axes[0,0].axis('off')
            axes[0,1].imshow(hm, cmap='jet', vmin=0, vmax=1)
            axes[0,1].set_title(f'Prior max={hm.max():.3f}'); axes[0,1].axis('off')
            axes[0,2].imshow(img_show)
            axes[0,2].imshow(cv2.resize(hm,(iw,ih)), cmap='jet', alpha=0.5)
            axes[0,2].set_title('Overlay'); axes[0,2].axis('off')
            im = axes[0,3].imshow(gate_np, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[0,3].set_title(f'Gate mean={gate_np.mean():.3f}'); axes[0,3].axis('off')
            plt.colorbar(im, ax=axes[0,3], fraction=0.046)

            axes[1,0].imshow(np.mean(v_feat,0), cmap='viridis')
            axes[1,0].set_title(f'F_v std={v_feat.std():.3f}'); axes[1,0].axis('off')
            axes[1,1].imshow(np.mean(f_feat,0), cmap='viridis')
            axes[1,1].set_title(f'F_f std={f_feat.std():.3f}'); axes[1,1].axis('off')
            diff = np.abs(np.mean(fused_np,0) - np.mean(v_feat,0))
            axes[1,2].imshow(diff/(diff.max()+1e-8), cmap='hot')
            axes[1,2].set_title(f'|Diff| max={diff.max():.4f}'); axes[1,2].axis('off')

            axes[1,3].hist(gate_np.flatten(), bins=50, color='steelblue', alpha=0.7)
            axes[1,3].axvline(x=gate_np.mean(), color='red', lw=2)
            axes[1,3].set_title('Gate Dist')

            gate_means = [g[b,0].cpu().numpy().mean() for g in gates]
            plt.suptitle(
                f'Sample {start_idx+b} | '
                f'Gate P3~P6: {[f"{m:.3f}" for m in gate_means]}',
                fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'vis_{start_idx+b:04d}.png'),
                        dpi=120, bbox_inches='tight')
            plt.close()

            self._save_paper_style_panels(
                img_show, hm, gates[lv][b, 0].cpu().numpy(),
                lv, iw, ih, save_dir, start_idx + b)

    def _save_paper_style_panels(self, img_show, heatmap, gate_np, lv,
                                 iw, ih, save_dir, idx):
        """Save compact panels that can be used for paper Figure 6 drafts."""
        heat_resized = cv2.resize(heatmap, (iw, ih))
        gate_resized = cv2.resize(gate_np, (iw, ih))

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_show)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='magma', vmin=0, vmax=1)
        axes[1].set_title('Frequency Prior')
        axes[1].axis('off')

        axes[2].imshow(img_show)
        heat_overlay = np.ma.masked_where(heat_resized < 0.20, heat_resized)
        axes[2].imshow(heat_overlay, cmap='magma', alpha=0.62, vmin=0, vmax=1)
        axes[2].set_title('Heatmap Overlay')
        axes[2].axis('off')

        g_low, g_high = np.percentile(gate_resized, [5, 95])
        g_mid = float(np.median(gate_resized))
        g_span = max(float(g_high - g_low), 0.04)
        axes[3].imshow(
            gate_resized, cmap='cividis',
            vmin=g_mid - g_span / 2, vmax=g_mid + g_span / 2)
        axes[3].set_title(f'Gate Map P{lv+3}')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'paper_gate_{idx:04d}.png'),
                    dpi=180, bbox_inches='tight')
        plt.close()
