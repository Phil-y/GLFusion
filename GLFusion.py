"""Vision Mamba (VIM) for chest radiographs"""
import mscmamba_adapter  # noqa: F401 - registers mscmamba_dual with timm
import torch.nn as nn
import torch.nn.functional as F
import torch   #负责张量计算。
import torch.nn as nn #基础张量和模块，后面所有网络层，比如 Linear、Conv2d、LayerNorm，都从这里来
from torch import Tensor #只是类型标注，让人知道某些参数应该是张量
from typing import Any, Dict, Optional
# timm 里的 _cfg、register_model、DropPath、初始化函数：为了把这个模型注册成 timm 可创建的模型。
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model   
from timm.models.layers import trunc_normal_, lecun_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from collections import namedtuple
# Mamba：真正的状态空间模块，来自底层 mamba_ssm。
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
# rope：可选的 rotary position embedding。
from rope import *  #导入 rope.py 里的旋转位置编码。这个项目默认注册的模型里 if_rope=False，所以通常不会用到。
import random
# RMSNorm / layer_norm_fn / rms_norm_fn：为了走 fused add + norm 的高效实现。
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
# __all__ 只是声明这个文件对外暴露哪些模型名，本身不影响训练逻辑。
__all__ = [
    'vim_tiny_patch16_224', 'vim_small_patch16_224', 'vim_base_patch16_224',
    'vim_tiny_patch16_384', 'vim_small_patch16_384', 'vim_base_patch16_384',
]

# PatchEmbed：把图像变成 token 序列 ，这是视觉模型最前面的 patch 切分模块。   
class PatchEmbed(nn.Module):
    """ 
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size) #把输入尺寸统一成 (H, W)
        patch_size = to_2tuple(patch_size) #patch 大小也统一成二元组。
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1) #计算切完 patch 后得到多少个网格。
        self.num_patches = self.grid_size[0] * self.grid_size[1]  #总 patch 数，也就是后面 token 序列长度的主体部分。
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  #这行最关键。它不是手工切 patch，而是直接用一个 kernel_size=patch_size, stride=stride 的卷积，把每个 patch 投影到 embed_dim 维。
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape   #取 batch 和图像尺寸。
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert H % self.img_size[0] == 0 and W % self.img_size[1] == 0, \
            f"Input image size ({H}*{W}) dose not match model ({self.img_size[0]}*{self.img_size[1]})." # 这里不是要求输入和训练尺寸完全相等，而是要求输入尺寸能被当前配置整齐处理。这个项目支持双视角上下拼接后高度变成 2*img_size，所以这里写成“整除关系”。

        x = self.proj(x) #卷积得到 patch feature map。
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  #从 [B, C, H', W'] 变成 [B, N, C]，也就是 token 序列。
        x = self.norm(x) #对 token 做归一化。
        return x
'''
你可以把它理解成“Transformer block 的 Mamba 版本”：
    有残差
    有归一化
    里面的核心算子不是 self-attention，而是 Mamba
标准逻辑大致是：
    把当前输入和残差相加
    做 norm
    送进 self.mixer
    返回新的 hidden_states 和 residual
'''
class TokenSpatialMixer(nn.Module):
    """Grid-aware token refinement for CXR cardiovascular risk cues."""

    def __init__(
            self,
            dim: int,
            drop: float = 0.0,
            use_multiscale: bool = True,
            use_edge: bool = True,
            use_cardiac_prior: bool = True,
            use_spatial_gate: bool = True,
            use_channel_gate: bool = True,
    ):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_edge = use_edge
        self.use_cardiac_prior = use_cardiac_prior
        self.use_spatial_gate = use_spatial_gate
        self.use_channel_gate = use_channel_gate
        self.norm = nn.LayerNorm(dim)
        if use_multiscale:
            self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
            self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
            self.dilated = nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False)
        if use_edge:
            self.edge = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        if use_spatial_gate:
            self.gate = nn.Conv2d(3, 1, 7, padding=3, bias=True)
        hidden = max(dim // 8, 16)
        if use_channel_gate:
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, dim, 1),
                nn.Sigmoid(),
            )
        self.out = nn.Conv2d(dim, dim, 1, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-3)

    def _token_keep_mask(self, token_count, token_position, device):
        keep = torch.ones(token_count, dtype=torch.bool, device=device)
        if token_position is None:
            return keep
        if isinstance(token_position, list):
            for pos in token_position:
                if 0 <= pos < token_count:
                    keep[pos] = False
        elif 0 <= token_position < token_count:
            keep[token_position] = False
        return keep

    def _cardiac_prior(self, h, w, device, dtype):
        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
        heart = torch.exp(-((x / 0.58) ** 2 + ((y - 0.25) / 0.72) ** 2))
        mediastinum = torch.exp(-((x / 0.32) ** 2 + ((y + 0.05) / 1.15) ** 2))
        return torch.clamp(0.7 * heart + 0.3 * mediastinum, 0.0, 1.0)

    def forward(self, hidden_states, token_position=None, grid_size=None):
        if grid_size is None:
            return hidden_states
        h, w = grid_size
        keep = self._token_keep_mask(hidden_states.shape[1], token_position, hidden_states.device)
        patch_tokens = hidden_states[:, keep, :]
        if patch_tokens.shape[1] != h * w:
            return hidden_states

        b, _, c = patch_tokens.shape
        patch_tokens = self.norm(patch_tokens)
        spatial = patch_tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        local = torch.zeros_like(spatial)
        if self.use_multiscale:
            local = self.dw3(spatial) + self.dw5(spatial) + self.dilated(spatial)
        edge = torch.zeros_like(spatial)
        if self.use_edge:
            edge = self.edge(spatial - F.avg_pool2d(spatial, kernel_size=3, stride=1, padding=1))
        prior = torch.zeros((b, 1, h, w), device=spatial.device, dtype=spatial.dtype)
        if self.use_cardiac_prior:
            prior = self._cardiac_prior(h, w, spatial.device, spatial.dtype).expand(b, -1, -1, -1)
        spatial_gate = 1.0
        if self.use_spatial_gate:
            spatial_summary = torch.cat(
                [spatial.mean(1, keepdim=True), spatial.amax(1, keepdim=True), prior], dim=1
            )
            spatial_gate = torch.sigmoid(self.gate(spatial_summary))
        channel_gate = 1.0
        if self.use_channel_gate:
            channel_gate = self.channel_gate(spatial)
        enhanced = self.out((local + edge) * spatial_gate) * channel_gate
        enhanced = enhanced.flatten(2).transpose(1, 2).contiguous()
        enhanced = self.drop(enhanced) * self.layer_scale

        output = hidden_states.clone()
        output[:, keep, :] = hidden_states[:, keep, :] + enhanced
        return output


# 一个标准的 Mamba 残差块    这个类相当于 Transformer 里的一个 encoder block，只不过把注意力/MLP 换成了 Mamba mixer。
class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
            cvd_spatial_mixer=False, cvd_spatial_mixer_cfg=None,
    ):
        """
        dim 是 token 维度，比如 small 模型里是 384。
        mixer_cls 是一个“还没真正实例化的 Mamba 类”。后面会执行 mixer_cls(dim) 生成真正的 Mamba 层。
        norm_cls 是归一化层，可能是 LayerNorm，也可能是 RMSNorm。
        fused_add_norm 表示是否用融合版的“残差相加 + norm”，速度更快。
        residual_in_fp32 表示残差分支是否用 float32 保存，更稳定。
        drop_path 是 stochastic depth，训练时随机跳过某些残差路径，减少过拟合。

        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)    #LayerNorm 或 RMSNorm。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()    #随机深度，训练时做正则化。 
        spatial_mixer_cfg = cvd_spatial_mixer_cfg or {}
        self.cvd_spatial_mixer = (
            TokenSpatialMixer(dim, drop=drop_path, **spatial_mixer_cfg)
            if cvd_spatial_mixer else None
        )
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
            token_position=None, grid_size=None
    ):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None: #说明这是第一层，直接把当前输入作为残差起点
                residual = hidden_states
            else:
                #如果不是第一层，就做残差相加：
                residual = residual + self.drop_path(hidden_states) 
            #然后做归一化：
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:   #残差分支提升到 fp32，提高数值稳定性。
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:   #把“残差加法 + norm”一次性做掉，逻辑和上面等价，只是更高效，是 fused 版本。它把“残差相加 + norm”合成一个底层函数，逻辑基本一样，只是更快。
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        '''
        这里最关键的一句是 models_mamba.py (line 115)，
        它表明真正的时序/序列建模发生在 self.mixer(...)，
        而这个 mixer 就是底层 mamba_ssm 里的 Mamba 模块。
        也就是说：归一化后的 token 序列被送进 Mamba，Mamba 负责序列建模。
        '''
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        # 注意它返回两个值。hidden_states 是当前 Mamba 层输出，residual 是残差分支，下一层还要继续用。
        if self.cvd_spatial_mixer is not None:
            hidden_states = self.cvd_spatial_mixer(hidden_states, token_position=token_position, grid_size=grid_size)
        return hidden_states, residual   

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# 构造 block 的工厂函数   这个函数本身不复杂，但很重要，因为它把“普通 Mamba”还是“Bi-Mamba”决定了。
# 它的意义是：把 Mamba 的各种配置提前包装好，然后交给 Block。
def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
        cvd_spatial_mixer=False,
        cvd_spatial_mixer_cfg=None,
):
    if if_bimamba:
        bimamba_type = "v1"   #老接口兼容。如果显式传 if_bimamba=True，就当成 v1。
    if ssm_cfg is None:
        ssm_cfg = {}  #防空。
    factory_kwargs = {"device": device, "dtype": dtype} #把 dtype/device 传进底层模块。
    # partial 可以理解成“先把一部分参数填好，剩下的以后再填”。比如这里先填好 layer_idx、bimamba_type、if_devide_out 等，后面 Block 里再调用：  这样每一层都能创建对应配置的 Mamba。
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out,
                        init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)   #关键行。这里把参数“预灌”进 Mamba，后面 Block 再调用 mixer_cls(dim) 就能生成对应的 mixer。
    norm_cls = partial(# 创建 norm 类。如果 rms_norm=True，就用 RMSNorm；否则用 LayerNorm。
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    # 创建并返回 Block。
    block = Block(   #生成一个完整 block
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        cvd_spatial_mixer=cvd_spatial_mixer,
        cvd_spatial_mixer_cfg=cvd_spatial_mixer_cfg,
    )
    block.layer_idx = layer_idx  #给 block 记住层编号，调试和内部行为可能会用到。
    return block

# 初始化策略    一个更偏 Mamba/GPT 风格，一个更偏视觉 backbone 风格，作者两套都用了
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:   # 对 Linear 的 bias 清零
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):  
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# 整个模型是怎么搭起来的
class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,  #输入图像大小。
                 patch_size=16,
                 stride=16,   #patch embedding 的卷积步长。
                 depth=24,   # Mamba block 层数
                 embed_dim=192,  #网络深度和通道维度 token 维度。
                 channels=3,
                 num_classes=1000,  #分类输出维度。你的任务里通常是 1。
                 ssm_cfg=None,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 if_bidirectional=False,
                 final_pool_type='none',
                 if_abs_pos_embed=False, #是否使用绝对位置编码。
                 if_rope=False,  #是否使用旋转位置编码。
                 if_rope_residual=False,
                 flip_img_sequences_ratio=-1.,
                 if_bimamba=False,  
                 bimamba_type="none", #传给底层 Mamba 的 Bi-Mamba 类型。
                 if_cls_token=False,   #是否使用 cls token。
                 if_devide_out=False,
                 init_layer_scale=None,
                 use_double_cls_token=False,  
                 use_middle_cls_token=False,  #是否把 cls token 放在 token 序列中间。
                 dual_mode=False, # 是否双视角。
                 lin4=False,
                 cat_cls=False,   #双视角时，是拼图输入，还是分别提特征后拼 cls。
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.dual_mode = dual_mode
        self.lin4 = lin4
        self.cat_cls = cat_cls

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        '''
        这里完成整网搭建：
            如果是双视角且不是 cat_cls 模式，输入高变成 2 * img_size，见 models_mamba.py (line 261)
            建 patch_embed
            建 cls_token
            建位置编码 pos_embed
            堆叠很多个 Block 到 self.layers
            最后接一个线性分类头 self.head
        '''
        if self.dual_mode and not self.cat_cls:
            img_size = (2*img_size, img_size)
        # 建 patch_embed
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        # 拿到 patch 数量。后面位置编码需要知道 token 个数。
        num_patches = self.patch_embed.num_patches
        # 建 cls_token cls token 是一个可学习向量，它会跟图像 token 放在一起经过 Mamba，最后用它代表整张图的特征。
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1
        # 创建绝对位置编码 pos_embed 位置编码也是一个可学习参数，形状是 [1, num_patches + num_tokens, embed_dim]，它会加到 token 上，告诉模型每个 token 的位置信息。
        if if_abs_pos_embed:  #位置编码：
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:  # 给序列 token 加旋转位置编码
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len
            )

        if self.lin4:   #如果最后要拼接最后 4 层的 cls feature，分类头输入维度要扩大 4 倍。
            self.num_features *= 4

        if self.cat_cls:  #如果前后位/侧位分别提 cls 再拼起来，维度扩大 2 倍。
            self.num_features *= 2

        # 创建分类头：你的二分类风险预测里，num_classes=1，所以最后输出一个 logit。
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() #最终二分类时，num_classes 在主训练脚本里会被设成 1。

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        cvd_spatial_blocks = kwargs.pop("cvd_spatial_blocks", False)
        cvd_spatial_mixer_cfg = kwargs.pop("cvd_spatial_mixer_cfg", None)
        if cvd_spatial_blocks == "last_half":
            cvd_spatial_layer_ids = set(range(depth // 2, depth))
        elif cvd_spatial_blocks == "late":
            cvd_spatial_layer_ids = set(range(3 * depth // 4, depth))
        elif cvd_spatial_blocks is True:
            cvd_spatial_layer_ids = set(range(depth))
        elif isinstance(cvd_spatial_blocks, (list, tuple, set)):
            cvd_spatial_layer_ids = set(cvd_spatial_blocks)
        else:
            cvd_spatial_layer_ids = set()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks 创建 depth 个 Mamba block
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    cvd_spatial_mixer=i in cvd_spatial_layer_ids,
                    cvd_spatial_mixer_cfg=cvd_spatial_mixer_cfg,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head   #输出归一化：  这是整个 backbone 最后的 norm。
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        # self.pre_logits = nn.Identity()

        # original init 初始化 patch embedding、分类头、位置编码、cls token。
        self.patch_embed.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        # mamba init  对整个模型应用 _init_weights。
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    # 给每一层 Mamba 分配推理缓存。这个更多用于序列生成式推理，图像分类训练里一般不用你手动碰。
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    # 是整份文件最重要的函数。它负责从图像中提取特征，但还不做最终分类。
    def  forward_features(self, x, inference_params=None, if_random_cls_token_position=False,
                         if_random_token_rank=False):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token

        # 先把图像变成 patch token： 比如单视角 224 x 224，patch 16，则：[B, 3, 224, 224]-> [B, 196, embed_dim]
        x = self.patch_embed(x)
        # 取出 batch size 和 token 数：
        B, M, _ = x.shape
        token_position = None
        grid_size = self.patch_embed.grid_size

        if self.if_cls_token:
            # 就在头尾各放一个 cls token。
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
            else:
                # 就在序列中间插入 cls token：
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    # add cls token in the middle
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                
                # 如果 if_random_cls_token_position=True，cls token 随机放一个位置。这通常是实验选项。否则默认放在最前面。
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        # 加绝对位置编码:位置编码的作用是告诉模型每个 token 在图像中的位置。否则 token 序列只是一堆 patch，模型不知道谁在左上、谁在右下。
        if self.if_abs_pos_embed:
            # if new_grid_size[0] == self.patch_embed.grid_size[0] and new_grid_size[1] == self.patch_embed.grid_size[1]:
            #     x = x + self.pos_embed
            # else:
            #     pos_embed = interpolate_pos_embed_online(
            #                 self.pos_embed, self.patch_embed.grid_size, new_grid_size,0
            #             )
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:

            # 是随机打乱 token 顺序的实验逻辑。打乱之后，还要重新找到 cls token 的位置。默认一般不开。
            shuffle_indices = torch.randperm(M)
            if isinstance(token_position, list):
                print("original value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行 shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                # 找到 cls token 在 shuffle 之后的新位置
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in
                                      range(len(token_position))]
                token_position = new_token_position
            else:
                # 找到 cls token 在 shuffle 之后的新位置
                token_position = torch.where(shuffle_indices == token_position)[0].item()

            if isinstance(token_position, list):
                print("new value: ", x[0, token_position[0], 0], x[0, token_position[1], 0])
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        # 是随机翻转 token 序列的实验逻辑。Mamba 对序列方向敏感，所以作者可能用这个测试顺序鲁棒性。默认通常不开。
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        # mamba impl  初始化残差和当前 hidden states：
        residual = None
        hidden_states = x
        # 是普通单向 Mamba 路径。
        if not self.if_bidirectional:
            hs_res_ls = []
            # 它遍历每一个 Mamba block：
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about  如果启用了 RoPE，就在每层前加旋转位置编码。默认注册模型里 if_rope=False，所以一般不会走 RoPE 分支。
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    token_position=token_position, grid_size=grid_size
                )
                hs_res_ls.append((hidden_states, residual))
                # todo: return last 4 hidden states: hidden_states are after norm
        else:
            # get two layers in a single for-loop
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params,
                    token_position=token_position, grid_size=grid_size
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]), None if residual == None else residual.flip([1]),
                    inference_params=inference_params, token_position=token_position, grid_size=grid_size
                )
                # 然后把反向结果再 flip 回来，与正向结果相加：
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.lin4:
                norm_hs_ls = []
        
                for hs, res in hs_res_ls[-4:]:
                    hs = fused_add_norm_fn(
                        self.drop_path(hs),
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=res,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                    )
                    norm_hs_ls.append(hs)

            else:
                hidden_states = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2
            else:
                if self.use_middle_cls_token:
                    if self.lin4:
                        return torch.cat([h[:, token_position, :] for h in norm_hs_ls], dim=-1)
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        x = self.forward_features(x, inference_params, if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
        # 如果 return_features=True，直接返回特征，不走分类头。这个可以用于提特征、可视化、下游任务。
        if return_features:
            return x
        if self.cat_cls:
            # 这个模式下，前面训练代码会把 frontal 和 lateral 沿 batch 维拼起来。比如原本 batch 有 B 个病人，每个病人两张图，送进模型时会变成 2B 张图。
            nb_sample = x.shape[0]//2
            x_cat = []
            for i in range(nb_sample):
                # 这里再把第 i 个 frontal 特征和第 i+B 个 lateral 特征拼起来：
                tmp = torch.cat([x[i, :], x[i+nb_sample, :]]).unsqueeze(0)
                x_cat.append(tmp)
            x = torch.cat(x_cat)
        # 最终分类：如果 num_classes=1，输出形状通常是：[B, 1]
        x = self.head(x)
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
        return x
'''
最核心你记住三句话就够了：PatchEmbed 把图像变 token，Block 用 Mamba 处理 token 序列，
VisionMamba.forward 取 cls token 后过线性层输出疾病风险 logit。
这个文件最难的不是 Python 语法，而是张量形状一路怎么变；只要形状跟住了，整套代码就会清楚很多。
'''

class ConvAct(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_chans, kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False
        )
        self.norm = nn.BatchNorm2d(out_chans)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        branch_channels = max(channels // 2, 16)
        self.branch1 = nn.Sequential(
            ConvAct(channels, branch_channels, 1),
            CVDConvAct(branch_channels, branch_channels, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvAct(channels, branch_channels, 1),
            ConvAct(branch_channels, branch_channels, 3, padding=3, dilation=3),
        )
        self.branch5 = nn.Sequential(
            ConvAct(channels, branch_channels, 1),
            ConvAct(branch_channels, branch_channels, 3, padding=5, dilation=5),
        )
        self.project = ConvAct(branch_channels * 3, channels, 1)
        self.skip = ConvAct(channels, channels, 1)

    def forward(self, x):
        x_cat = torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)
        return self.project(x_cat) + self.skip(x)


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = channels * 2
        self.in_proj = nn.Conv2d(hidden, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.out_proj = nn.Conv2d(hidden, channels, 1)
        self.refine = nn.Conv2d(channels, channels, 1)

    def forward(self, x, edge):
        fused = self.in_proj(torch.cat([x, edge], dim=1))
        value, gate = self.dwconv(fused).chunk(2, dim=1)
        fused = self.out_proj(F.gelu(value) * gate)
        return self.refine(fused + x)


class LightAttention2d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 16)
        self.query = nn.Conv2d(channels, reduced, 1, bias=False) 
        self.key = nn.Conv2d(channels, reduced, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.local3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.local5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = reduced ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        pooled = F.adaptive_avg_pool2d(x, output_size=(min(h, 16), min(w, 16)))
        ph, pw = pooled.shape[-2:]
        q = self.query(pooled).flatten(2).transpose(1, 2)
        k = self.key(pooled).flatten(2)
        v = self.value(pooled).flatten(2).transpose(1, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        low = (attn @ v).transpose(1, 2).reshape(b, c, ph, pw)
        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        high = self.local3(x) + self.local5(x)
        return self.out(low + high) + x


class LocalBranch(nn.Module):
    def __init__(
            self,
            in_chans: int,
            embed_dim: int,
            width: int = 64,
            use_rfb: bool = True,
            use_edge: bool = True,
            use_multiscale: bool = True,
            use_gated_fusion: bool = True,
            use_light_attention: bool = True,
    ):
        super().__init__()
        self.use_edge = use_edge
        self.use_gated_fusion = use_gated_fusion
        self.stem = nn.Sequential(
            ConvAct(in_chans, width, 3, stride=2, padding=1),
            ConvAct(width, width, 3, padding=1),
        )
        self.rfb = ReceptiveFieldBlock(width) if use_rfb else nn.Identity()
        if use_edge:
            self.edge = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Conv2d(width, width, 1, bias=False),
            )
        self.ms_mixer = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        ) if use_multiscale else nn.Identity()
        if use_gated_fusion:
            self.fusion = GatedFusion(width)
        self.light_attn = LightAttention2d(width) if use_light_attention else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(width),
            nn.Linear(width, embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.stem(x)
        local = self.rfb(x)
        edge = self.edge(local) if self.use_edge else torch.zeros_like(local)
        mixed = self.ms_mixer(local)
        if self.use_gated_fusion:
            fused = self.fusion(mixed, edge)
        else:
            fused = mixed + edge
        fused = self.light_attn(fused)
        return self.out(self.pool(fused))


class RiskHead(nn.Module):
    VALID_FUSION_TYPES = {"global", "add", "gated", "concat"}

    def __init__(
            self,
            feature_dim: int,
            num_classes: int,
            dropout: float = 0.1,
            fusion_type: str = "gated",
    ):
        super().__init__()
        if fusion_type not in self.VALID_FUSION_TYPES:
            valid = ", ".join(sorted(self.VALID_FUSION_TYPES))
            raise ValueError(f"Unknown CVD fusion type '{fusion_type}'. Expected one of: {valid}.")
        self.fusion_type = fusion_type
        self.norm_global = nn.LayerNorm(feature_dim)
        if fusion_type != "global":
            self.norm_local = nn.LayerNorm(feature_dim)
        if fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid(),
            )
        classifier_dim = feature_dim * 2 if fusion_type == "concat" else feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def fuse_features(self, global_feat: Tensor, local_feat: Optional[Tensor] = None) -> Tensor:
        global_feat = self.norm_global(global_feat)
        if self.fusion_type == "global":
            return global_feat
        if local_feat is None:
            raise ValueError(f"fusion_type='{self.fusion_type}' requires local features.")
        local_feat = self.norm_local(local_feat)
        if self.fusion_type == "add":
            return global_feat + local_feat
        if self.fusion_type == "concat":
            return torch.cat([global_feat, local_feat], dim=-1)
        gate = self.gate(torch.cat([global_feat, local_feat], dim=-1))
        return global_feat + gate * local_feat

    def forward(self, global_feat: Tensor, local_feat: Optional[Tensor] = None) -> Tensor:
        return self.classifier(self.fuse_features(global_feat, local_feat))


class SimpleRiskHead(nn.Module):
    """Minimal additive classifier used to ablate the complete CVDRiskHead."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def fuse_features(self, global_feat: Tensor, local_feat: Optional[Tensor] = None) -> Tensor:
        if local_feat is None:
            raise ValueError("CVDSimpleRiskHead requires local features.")
        return global_feat + local_feat

    def forward(self, global_feat: Tensor, local_feat: Optional[Tensor] = None) -> Tensor:
        return self.classifier(self.fuse_features(global_feat, local_feat))


class GLFusion(VisionMamba):
    def __init__(
            self,
            *args,
            cvd_branch_width: int = 64,
            cvd_head_dropout: float = 0.1,
            use_local_branch: bool = True,
            use_cvd_risk_head: bool = True,
            cvd_fusion_type: str = "gated",
            cvd_local_branch_cfg: Optional[Dict[str, Any]] = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not use_local_branch and cvd_fusion_type != "global":
            raise ValueError("A disabled local branch requires cvd_fusion_type='global'.")
        if not use_cvd_risk_head and not use_local_branch:
            raise ValueError("CVDRiskHead ablation requires the local branch.")
        self.use_local_branch = use_local_branch
        self.cvd_base_feature_dim = self.embed_dim * (4 if self.lin4 else 1)
        if use_local_branch:
            local_branch_cfg = cvd_local_branch_cfg or {}
            self.cvd_local_branch = LocalEvidenceBranch(
                in_chans=self.patch_embed.proj.in_channels,
                embed_dim=self.embed_dim,
                width=cvd_branch_width,
                **local_branch_cfg,
            )
            self.cvd_local_proj = nn.Linear(self.embed_dim, self.cvd_base_feature_dim)
            self.cvd_local_branch.apply(segm_init_weights)
            self.cvd_local_proj.apply(segm_init_weights)
        if use_cvd_risk_head:
            self.head = RiskHead(
                self.num_features,
                self.num_classes,
                dropout=cvd_head_dropout,
                fusion_type=cvd_fusion_type,
            )
        else:
            self.head = SimpleRiskHead(self.num_features, self.num_classes)

    def _pair_cat(self, x):
        nb_sample = x.shape[0] // 2
        return torch.cat([torch.cat([x[i], x[i + nb_sample]], dim=0).unsqueeze(0) for i in range(nb_sample)], dim=0)

    def forward(self, x, return_features=False, inference_params=None, if_random_cls_token_position=False,
                if_random_token_rank=False):
        global_feat = super().forward_features(
            x,
            inference_params=inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
        )
        if self.final_pool_type == "max":
            global_feat = global_feat.max(dim=1)[0]
            
        local_feat = None
        if self.use_local_branch:
            local_feat = self.cvd_local_proj(self.cvd_local_branch(x))
        if self.cat_cls:
            global_feat = self._pair_cat(global_feat)
            if local_feat is not None:
                local_feat = self._pair_cat(local_feat)
        if return_features:
            return self.head.fuse_features(global_feat, local_feat)
        return self.head(global_feat, local_feat)



def _build_cvd_hybrid(
        pretrained: bool = False,
        stride: int = 16,
        cvd_spatial_blocks=False,
        use_local_branch: bool = True,
        use_cvd_risk_head: bool = True,
        cvd_fusion_type: str = "gated",
        cvd_spatial_mixer_cfg: Optional[Dict[str, Any]] = None,
        cvd_local_branch_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
):
    """Build a controlled CVD variant while keeping backbone hyperparameters fixed."""
    del pretrained
    model = GLFusion(
        patch_size=16,
        stride=stride,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=True,
        cvd_branch_width=64,
        cvd_head_dropout=0.1,
        cvd_spatial_blocks=cvd_spatial_blocks,
        cvd_spatial_mixer_cfg=cvd_spatial_mixer_cfg,
        use_local_branch=use_local_branch,
        use_cvd_risk_head=use_cvd_risk_head,
        cvd_fusion_type=cvd_fusion_type,
        cvd_local_branch_cfg=cvd_local_branch_cfg,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model




@register_model
def GLFusion(pretrained=False, **kwargs):
    return _build_cvd_hybrid(
        pretrained=pretrained,
        cvd_spatial_blocks=True,
        cvd_fusion_type="gated",
        **kwargs,
    )
