#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MiniMax H3 Reference-to-Video (R2V) & Text-to-Video (T2V) 独立命令行外挂执行器 (面向对象架构重构版)

零侵入架构设计：
- 纯外挂无头运行：不篡改 ComfyUI 官方代码，不修改任何第三方插件源码。
- 面向对象分层：配置管理、环境探测与引导、显存隔离清理、动态 QKV 分块补丁、多模态加载与三阶段流水线解耦。
- 标准化日志：采用 logging 模块，具备结构化时间戳、等级控制与模块标签。
- 物理级三阶段内存隔离：文本/多模态提取 -> DiT/UNet 采样去噪 -> 双 VAE 解码与 MP4 导出。
"""

import argparse
import asyncio
import ctypes
import gc
import importlib
import logging
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch

# ==============================================================================
# 0. 底层硬件环境变量调优 (Intel oneAPI Level-Zero & PyTorch XPU / CUDA)
# ==============================================================================
os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")
os.environ.setdefault("TORCH_XPU_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ZE_COMMAND_QUEUE_SYNCHRONOUS_MODE", "0")
os.environ.setdefault("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "1")
os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")


# ==============================================================================
# 1. 规范化日志系统 (Standardized Logging)
# ==============================================================================
class LogFormatter(logging.Formatter):
    """标准控制台日志格式化器，附带组件标签与等级标识"""
    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        prefix = f"[{timestamp}] [{record.levelname:<5}] [{record.name}]"
        return f"{prefix} {record.getMessage()}"


def configure_logging(level_name: str = "INFO") -> logging.Logger:
    """初始化全局日志处理器"""
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger("H3")
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(LogFormatter())
    root_logger.addHandler(handler)
    root_logger.propagate = False
    return root_logger


logger = logging.getLogger("H3")


# ==============================================================================
# 2. 流水线配置与数学模型实体类 (H3PipelineConfig)
# ==============================================================================
@dataclass
class H3PipelineConfig:
    """封装 H3 CLI 运行时参数与数学物理换算逻辑"""
    prompt_file: str = "shot1.txt"
    images: Optional[List[str]] = None
    video: Optional[str] = None
    audio: Optional[str] = None
    seconds: float = 5.0
    aspect: str = "9:16"
    megapixels: float = 0.3
    steps: int = 8
    cfg: float = 1.0
    seed: int = 42
    chunks: int = 2
    head_chunks: int = 4
    qkv_chunk_size: int = 8192
    no_ref: bool = False
    ref_image_size: str = "max"
    output_prefix: str = "H3_R2V_Output"
    resume_latent: Optional[str] = None
    comfy_dir: Optional[str] = None
    log_level: str = "INFO"

    # 模型权重与 LoRA 集中配置 (默认当前代码调参所用版本)
    unet_name: str = "10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors"
    clip_name: str = "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"
    video_vae_name: str = "minimax_h3_video_vae_int8_convrot.safetensors"
    audio_vae_name: str = "minimax_h3_audio_vae_fp32.safetensors"
    turbo_lora_name: Optional[str] = "minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors"
    turbo_lora_strength: float = 1.0
    lora1_name: Optional[str] = None
    lora1_strength: float = 1.0

    ASPECT_RATIOS: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "16:9": (16, 9),
            "9:16": (9, 16),
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
        }
    )

    def is_text_to_video_mode(self) -> bool:
        """判定是否以纯文生视频模式运行 (若显式传入 --no-ref 或完全未提供有效参考输入)"""
        if self.no_ref:
            return True
        has_media = bool(self.images or self.video or self.audio)
        return not has_media

    def calc_frames(self) -> int:
        """MiniMax H3 专属帧数对齐公式：max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"""
        raw = max(5, round(self.seconds * 24))
        return raw + (5 - (raw % 17)) % 17

    def calc_resolution(self, multiple: int = 32) -> Tuple[int, int]:
        """基于百万像素与画幅比例计算输出分辨率，并严格对齐 32 像素倍数"""
        w_ratio, h_ratio = self.ASPECT_RATIOS.get(self.aspect, (9, 16))
        target_pixels = self.megapixels * 1_000_000
        h_base = math.sqrt(target_pixels / (w_ratio / h_ratio))
        w_base = h_base * (w_ratio / h_ratio)
        width = int(round(w_base / multiple) * multiple)
        height = int(round(h_base / multiple) * multiple)
        return width, height

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> "H3PipelineConfig":
        """从 argparse 命名空间安全构建配置实例"""
        valid_fields = cls.__dataclass_fields__.keys()
        kwargs = {k: v for k, v in vars(ns).items() if k in valid_fields}
        return cls(**kwargs)


# ==============================================================================
# 3. ComfyUI 运行环境引导与无头伪装管理器 (ComfyUIBootstrapper)
# ==============================================================================
class ComfyUIBootstrapper:
    """负责 ComfyUI 根目录探测、无头伪装注入及算子库延迟初始化"""

    def __init__(self, custom_path: Optional[str] = None):
        self.log = logging.getLogger("H3.Bootstrap")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.custom_path = custom_path
        self.comfy_dir = self._resolve_comfy_dir()
        self.nodes = None

    def _resolve_comfy_dir(self) -> str:
        candidates = []
        if self.custom_path:
            candidates.append(self.custom_path)
        if os.environ.get("COMFYUI_PATH"):
            candidates.append(os.environ.get("COMFYUI_PATH"))
        candidates.append(os.path.join(self.base_dir, "ComfyUI"))
        candidates.append(os.path.join(os.getcwd(), "ComfyUI"))
        candidates.append(os.path.join(os.path.dirname(self.base_dir), "ComfyUI"))
        candidates.append(self.base_dir)

        for c in candidates:
            if c and os.path.exists(c) and os.path.isdir(c):
                if os.path.exists(os.path.join(c, "main.py")) or os.path.exists(os.path.join(c, "comfy")):
                    return os.path.abspath(c)
        return os.path.abspath(os.path.join(self.base_dir, "ComfyUI"))

    def _install_headless_dummy_server(self):
        """注入全功能无头伪装 PromptServer，消除第三方插件 (如 kjnodes) 的 Web 前端依赖"""
        if "server" in sys.modules:
            return

        class _FlexibleDummy:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
            def __getattr__(self, name):
                return self
            def __call__(self, *args, **kwargs):
                return self
            def __bool__(self):
                return False

        class _DummyRoutes:
            frozen = True
            def __getattr__(self, name):
                return _FlexibleDummy()
            def post(self, *args, **kwargs):
                return lambda f: f
            def get(self, *args, **kwargs):
                return lambda f: f

        class _DummyPromptServer:
            def __init__(self):
                self.supports = []
                self.sockets = []
                self.routes = _DummyRoutes()
                self.app = _FlexibleDummy(router=_DummyRoutes())
            def __getattr__(self, name):
                return _FlexibleDummy()
            def send_sync(self, *args, **kwargs):
                pass

        class _PromptServerMeta(type):
            @property
            def instance(cls):
                if not hasattr(cls, "_inst"):
                    cls._inst = _DummyPromptServer()
                return cls._inst

        class _PromptServer(metaclass=_PromptServerMeta):
            instance = _DummyPromptServer()

        server_mod = types.ModuleType("server")
        server_mod.PromptServer = _PromptServer
        sys.modules["server"] = server_mod

    def bootstrap(self) -> Any:
        """执行完整环境挂载与节点算子加载流程"""
        if self.nodes is not None:
            return self.nodes

        if self.comfy_dir not in sys.path:
            sys.path.insert(0, self.comfy_dir)

        self._install_headless_dummy_server()
        self.log.info(f"已挂载 ComfyUI 核心目录: {self.comfy_dir}")

        import nodes
        self.nodes = nodes

        # 加载 comfy_extras 模块
        if hasattr(nodes, "init_extra_nodes"):
            res = nodes.init_extra_nodes()
            if asyncio.iscoroutine(res):
                asyncio.run(res)

        for extra_pkg in [
            "comfy_extras.nodes_minimax_h3",
            "comfy_extras.nodes_custom_sampler",
            "comfy_extras.nodes_audio",
            "comfy_extras.nodes_video",
        ]:
            try:
                mod = importlib.import_module(extra_pkg)
                if hasattr(mod, "NODE_CLASS_MAPPINGS"):
                    nodes.NODE_CLASS_MAPPINGS.update(mod.NODE_CLASS_MAPPINGS)
            except Exception:
                pass

        # 加载 custom_nodes 插件扩展
        if hasattr(nodes, "init_external_custom_nodes"):
            res = nodes.init_external_custom_nodes()
            if asyncio.iscoroutine(res):
                asyncio.run(res)
        elif hasattr(nodes, "load_custom_nodes"):
            nodes.load_custom_nodes()

        # 校验核心算子
        if "MiniMaxH3ReferenceToVideo" not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError("未检测到 MiniMaxH3ReferenceToVideo 节点，请确认 ComfyUI/comfy_extras 目录完整！")

        return self.nodes


# ==============================================================================
# 4. UMA 显存与系统物理工作集回收管理器 (MemoryManager)
# ==============================================================================
class MemoryManager:
    """负责跨阶段的物理显存卸载、垃圾回收以及 Windows 物理内存工作集归还"""

    def __init__(self):
        self.log = logging.getLogger("H3.Memory")

    def purge(self, stage_desc: str = ""):
        """强制卸载 ComfyUI 模型缓存、清空 PyTorch 显存缓存并收缩进程物理内存"""
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
        except Exception as e:
            self.log.debug(f"ComfyUI 模型卸载反馈: {e}")

        gc.collect()
        gc.collect()

        # XPU 显存回收
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
            alloc_mb = torch.xpu.memory_allocated() / (1024 ** 2)
            res_mb = torch.xpu.memory_reserved() / (1024 ** 2)
            self.log.info(f"[{stage_desc}] XPU 显存已释放: 已分配 {alloc_mb:.1f} MB, 已保留 {res_mb:.1f} MB")
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            self.log.info(f"[{stage_desc}] CUDA 显存已释放: 已分配 {alloc_mb:.1f} MB, 已保留 {res_mb:.1f} MB")
        else:
            self.log.info(f"[{stage_desc}] 显存缓存与模型引用已清空")

        # Windows 统一内存 (UMA) 物理工作集收缩
        try:
            ctypes.windll.psapi.EmptyWorkingSet(-1)
        except Exception:
            pass


# ==============================================================================
# 5. 纯外挂 QKV 序列分块动态热补丁器 (ModelPatcher)
# ==============================================================================
class ModelPatcher:
    """
    纯外挂热补丁：直接作用于已载入的 DiT 模型，为所有 Attention.qkv_proj 注入分块保护。
    无需修改任何 ComfyUI 源码或 custom_nodes 插件代码，杜绝 Intel Level-Zero 驱动 Error 39/40 报错。
    """

    def __init__(self):
        self.log = logging.getLogger("H3.Patcher")

    @staticmethod
    def get_diffusion_model(model_wrapper: Any) -> Any:
        """安全解析 ModelPatcher 中的底层 DiT 扩散模型实例"""
        if hasattr(model_wrapper, "get_model_object"):
            try:
                return model_wrapper.get_model_object("diffusion_model")
            except Exception:
                pass
        if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "diffusion_model"):
            return model_wrapper.model.diffusion_model
        if hasattr(model_wrapper, "diffusion_model"):
            return model_wrapper.diffusion_model
        return model_wrapper

    def apply_qkv_chunking(self, model_wrapper: Any, chunk_size: int = 8192) -> int:
        """为扩散模型的 Attention 投影层动态挂接 QKV 分块 forward hook"""
        diffusion_model = self.get_diffusion_model(model_wrapper)
        patched_count = 0

        def make_chunked_forward(orig_layer):
            orig_forward = orig_layer.forward

            def chunked_forward(x, *args, **kwargs):
                is_3d = (x.dim() == 3)
                seq_len = x.shape[1] if is_3d else x.shape[0]

                if seq_len > chunk_size:
                    out_features = getattr(orig_layer, "out_features", None)
                    if out_features is None and hasattr(orig_layer, "weight"):
                        out_features = orig_layer.weight.shape[0]

                    if out_features is not None:
                        if is_3d:
                            qkv = torch.empty((x.shape[0], seq_len, out_features), dtype=x.dtype, device=x.device)
                            for i in range(0, seq_len, chunk_size):
                                end = min(i + chunk_size, seq_len)
                                qkv[:, i:end, :] = orig_forward(x[:, i:end, :], *args, **kwargs)
                        else:
                            qkv = torch.empty((seq_len, out_features), dtype=x.dtype, device=x.device)
                            for i in range(0, seq_len, chunk_size):
                                end = min(i + chunk_size, seq_len)
                                qkv[i:end, :] = orig_forward(x[i:end, :], *args, **kwargs)
                        return qkv
                return orig_forward(x, *args, **kwargs)

            return chunked_forward

        for _, m in diffusion_model.named_modules():
            if hasattr(m, "qkv_proj"):
                layer = m.qkv_proj
                if not getattr(layer, "_is_chunk_hooked", False):
                    layer.forward = make_chunked_forward(layer)
                    layer._is_chunk_hooked = True
                    patched_count += 1

        self.log.info(f"QKV 分块补丁已挂载 (chunk_size={chunk_size}, 成功保护 {patched_count} 个 Attention 模块)")
        return patched_count


# ==============================================================================
# 6. 多模态素材加载与张量预处理器 (MultimodalLoader)
# ==============================================================================
class MultimodalLoader:
    """负责提示词及图像、视频、音频多模态参考输入的检索与张量化"""

    def __init__(self, base_dir: str, comfy_dir: str):
        self.log = logging.getLogger("H3.Loader")
        self.base_dir = base_dir
        self.comfy_dir = comfy_dir

    def resolve_candidate_path(self, filename: str, sub_dir: str = "") -> Optional[str]:
        """多候选路径检索"""
        if not filename:
            return None
        candidates = [
            filename,
            os.path.join(self.base_dir, filename),
            os.path.join(self.comfy_dir, filename),
        ]
        if sub_dir:
            candidates.append(os.path.join(self.comfy_dir, sub_dir, filename))
            candidates.append(os.path.join(self.base_dir, sub_dir, filename))

        for c in candidates:
            if c and os.path.exists(c) and os.path.isfile(c):
                return os.path.abspath(c)
        return None

    def load_prompt_text(self, prompt_file: str) -> str:
        """检索并读取提示词文本"""
        stem = prompt_file.replace(".txt", "")
        candidates = [
            os.path.join(self.base_dir, prompt_file),
            os.path.join(self.base_dir, f"{stem}.txt"),
            os.path.join(self.comfy_dir, prompt_file),
            os.path.join(self.comfy_dir, f"{stem}.txt"),
            os.path.join(self.base_dir, "prompt.txt"),
        ]
        for p in candidates:
            if os.path.exists(p) and os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        self.log.info(f"已从 {os.path.basename(p)} 加载提示词")
                        return content

        default_prompt = "Natural realistic live-action lifestyle video, authentic lighting."
        self.log.warning(f"未找到指定提示词文件 ({prompt_file})，采用默认保底提示词: '{default_prompt}'")
        return default_prompt

    def load_image_tensor(self, file_path: str) -> torch.Tensor:
        """加载图片并转换为标准 ComfyUI 图像张量 [1, H, W, C] (float32 [0, 1])"""
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None,]

    def load_video_frames(self, file_path: str) -> Optional[torch.Tensor]:
        """读取视频帧并转换为张量 [T, H, W, C]"""
        import cv2
        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        if not frames:
            return None
        return torch.from_numpy(np.stack(frames))

    def load_audio_dict(self, file_path: str) -> Dict[str, Any]:
        """读取音频文件并转换为 ComfyUI 音频字典"""
        import torchaudio
        waveform, sr = torchaudio.load(file_path)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sr}

    def build_reference_kwargs(self, config: H3PipelineConfig) -> Dict[str, Any]:
        """统筹组装传递给 MiniMaxH3ReferenceToVideo 的多模态字典"""
        kwargs: Dict[str, Any] = {}
        if config.is_text_to_video_mode():
            self.log.info("运行模式: 纯文生视频模式 (Text-to-Video / --no-ref)，跳过多模态参考抽取")
            return kwargs

        # 1. 处理参考图片
        if config.images:
            unique_images = []
            seen = set()
            for img_name in config.images:
                if img_name.lower() in ["none", "null", "no"]:
                    continue
                path = self.resolve_candidate_path(img_name, sub_dir="input")
                if path and path not in seen:
                    seen.add(path)
                    unique_images.append(path)
                elif not path:
                    self.log.warning(f"参考图未找到: {img_name}")

            if unique_images:
                ref_images_dict = {}
                for idx, img_path in enumerate(unique_images):
                    self.log.info(f"载入参考图像 [{idx}]: {os.path.basename(img_path)}")
                    ref_images_dict[f"ref_image_{idx}"] = self.load_image_tensor(img_path)
                kwargs["ref_images"] = ref_images_dict

        # 2. 处理参考视频
        if config.video:
            video_path = self.resolve_candidate_path(config.video, sub_dir="input")
            if video_path:
                self.log.info(f"载入参考视频: {os.path.basename(video_path)}")
                frames_tensor = self.load_video_frames(video_path)
                if frames_tensor is not None:
                    kwargs["ref_videos"] = {"ref_video_0": frames_tensor}
            else:
                self.log.warning(f"参考视频未找到: {config.video}")

        # 3. 处理参考音频
        if config.audio:
            audio_path = self.resolve_candidate_path(config.audio, sub_dir="input")
            if audio_path:
                self.log.info(f"载入参考音频: {os.path.basename(audio_path)}")
                kwargs["ref_audios"] = {"ref_audio_0": self.load_audio_dict(audio_path)}
            else:
                self.log.warning(f"参考音频未找到: {config.audio}")

        return kwargs


# ==============================================================================
# 7. 核心执行流水线类 (MiniMaxH3Pipeline)
# ==============================================================================
class MiniMaxH3Pipeline:
    """MiniMax H3 视频生成三阶段物理隔离主流水线"""

    def __init__(self, config: H3PipelineConfig, bootstrapper: ComfyUIBootstrapper):
        self.config = config
        self.bootstrapper = bootstrapper
        self.nodes = bootstrapper.bootstrap()
        self.comfy_dir = bootstrapper.comfy_dir
        self.base_dir = bootstrapper.base_dir

        self.memory = MemoryManager()
        self.patcher = ModelPatcher()
        self.loader = MultimodalLoader(self.base_dir, self.comfy_dir)
        self.log = logging.getLogger("H3.Pipeline")

    @staticmethod
    def execute_node(node_instance: Any, **kwargs) -> Any:
        """利用 ComfyUI 原生 FUNCTION 机制动态反射调用节点主函数"""
        func_name = getattr(node_instance, "FUNCTION")
        with torch.inference_mode():
            res = getattr(node_instance, func_name)(**kwargs)
        if hasattr(res, "args") and isinstance(res.args, tuple):
            return res.args
        return res

    def stage1_extract_conditions(self, width: int, height: int, length: int) -> Tuple[Any, Any, Any]:
        """阶段 1：检索资源、编码文本/视觉/音频参考条件并构建潜变量，完成后物理抹除该阶段权重"""
        self.log.info("--- [阶段 1/3] 提示词与多模态参考特征抽取 ---")
        prompt_text = self.loader.load_prompt_text(self.config.prompt_file)
        r2v_kwargs = self.loader.build_reference_kwargs(self.config)

        has_visual = "ref_images" in r2v_kwargs or "ref_videos" in r2v_kwargs
        has_audio = "ref_audios" in r2v_kwargs or "ref_videos" in r2v_kwargs

        self.log.info(f"加载 Text Encoder ({self.config.clip_name})...")
        clip_model = self.execute_node(
            self.nodes.CLIPLoader(),
            clip_name=self.config.clip_name,
            type="minimax",
            device="default"
        )[0]

        ref_video_vae = None
        if has_visual:
            self.log.info(f"检测到视觉参考，加载视频特征编码 VAE ({self.config.video_vae_name})...")
            ref_video_vae = self.execute_node(
                self.nodes.VAELoader(),
                vae_name=self.config.video_vae_name
            )[0]

        ref_audio_vae = None
        if has_audio:
            self.log.info(f"检测到音频参考，加载音频特征编码 VAE ({self.config.audio_vae_name})...")
            ref_audio_vae = self.execute_node(
                self.nodes.VAELoader(),
                vae_name=self.config.audio_vae_name
            )[0]

        ref2vid_node = self.nodes.NODE_CLASS_MAPPINGS["MiniMaxH3ReferenceToVideo"]()
        if has_visual or has_audio:
            self.log.info(f"执行多模态特征对齐与参考注入 (尺寸策略: {self.config.ref_image_size})...")
        else:
            self.log.info("执行纯文本提示词编码与基础潜变量构建 (纯文生视频模式)...")

        positive_cond, base_latent = self.execute_node(
            ref2vid_node,
            clip=clip_model,
            vae=ref_video_vae,
            audio_vae=ref_audio_vae,
            width=width,
            height=height,
            length=length,
            prompt=prompt_text,
            ref_image_size=self.config.ref_image_size,
            **r2v_kwargs
        )

        negative_cond = self.execute_node(self.nodes.CLIPTextEncode(), clip=clip_model, text="")[0]

        # 阶段 1 彻底销毁与显存工作集清扫
        del clip_model
        if ref_video_vae is not None:
            del ref_video_vae
        if ref_audio_vae is not None:
            del ref_audio_vae
        del ref2vid_node
        del r2v_kwargs
        self.memory.purge("阶段1: 文本与参考特征编码器已卸载")

        return positive_cond, negative_cond, base_latent

    def stage2_sample(self, positive_cond: Any, negative_cond: Any, base_latent: Any) -> Any:
        """阶段 2：DiT 主模型加载、LoRA 挂载、QKV 分块保护与 KSampler 采样去噪"""
        self.log.info("--- [阶段 2/3] UNet 模型装载、LoRA 融合与 KSampler 采样 ---")

        self.log.info(f"载入 DiT 主模型 ({self.config.unet_name})...")
        model = self.execute_node(
            self.nodes.UNETLoader(),
            unet_name=self.config.unet_name,
            weight_dtype="default"
        )[0]

        # 纯外挂 QKV 分块补丁注入
        self.patcher.apply_qkv_chunking(model, chunk_size=self.config.qkv_chunk_size)

        # 注入 LoRA 1: 风格增强 LoRA (自适应检测)
        if self.config.lora1_name:
            lora1_path = self.loader.resolve_candidate_path(self.config.lora1_name, sub_dir=os.path.join("models", "loras"))
            if lora1_path:
                self.log.info(f"注入 LoRA 1: {self.config.lora1_name} (Strength: {self.config.lora1_strength})...")
                try:
                    model = self.execute_node(
                        self.nodes.LoraLoaderModelOnly(),
                        model=model,
                        lora_name=self.config.lora1_name,
                        strength_model=self.config.lora1_strength
                    )[0]
                except Exception as e:
                    self.log.warning(f"LoRA 1 挂载跳过: {e}")
            else:
                self.log.warning(f"未找到 LoRA 1 模型文件: {self.config.lora1_name}，跳过挂载")
        else:
            self.log.debug("未配置 LoRA 1，跳过挂载")

        # 注入 LoRA 2: Turbo 蒸馏加速 LoRA (插件优先，平滑降级)
        if self.config.turbo_lora_name:
            if "MiniMaxH3TurboLoRA" in self.nodes.NODE_CLASS_MAPPINGS:
                self.log.info(f"注入 LoRA 2: Turbo 蒸馏加速 LoRA ({self.config.turbo_lora_name}, 调用 comfyui-minimax-h3-turbo 插件)...")
                model = self.execute_node(
                    self.nodes.NODE_CLASS_MAPPINGS["MiniMaxH3TurboLoRA"](),
                    model=model,
                    lora_name=self.config.turbo_lora_name,
                    strength=self.config.turbo_lora_strength,
                    low_vram=True
                )[0]
            else:
                self.log.info(f"未检测到 MiniMaxH3TurboLoRA 插件，采用原生 LoraLoaderModelOnly 兼容加载 ({self.config.turbo_lora_name})...")
                try:
                    model = self.execute_node(
                        self.nodes.LoraLoaderModelOnly(),
                        model=model,
                        lora_name=self.config.turbo_lora_name,
                        strength_model=self.config.turbo_lora_strength
                    )[0]
                except Exception as e:
                    self.log.warning(f"原生加载 Turbo LoRA 反馈: {e}")
        else:
            self.log.info("未配置 Turbo LoRA，以主模型基底直接采样")

        # 显存分块算子挂载 (kjnodes 容错)
        if "MiniMaxLowVRAMAttention" in self.nodes.NODE_CLASS_MAPPINGS:
            self.log.info(f"应用 MiniMaxLowVRAMAttention (注意力头分块: {self.config.head_chunks})...")
            model = self.execute_node(
                self.nodes.NODE_CLASS_MAPPINGS["MiniMaxLowVRAMAttention"](),
                model=model,
                head_chunks=self.config.head_chunks
            )[0]
        else:
            self.log.debug("未安装 comfyui-kjnodes，跳过 MiniMaxLowVRAMAttention (已由内置 QKV 补丁兜底防崩)")

        if "MiniMaxChunkFeedForward" in self.nodes.NODE_CLASS_MAPPINGS:
            self.log.info(f"应用 MiniMaxChunkFeedForward (FFN 序列分块: {self.config.chunks})...")
            model = self.execute_node(
                self.nodes.NODE_CLASS_MAPPINGS["MiniMaxChunkFeedForward"](),
                model=model,
                chunks=self.config.chunks,
                seq_threshold=4096
            )[0]
        else:
            self.log.debug("未安装 comfyui-kjnodes，跳过 MiniMaxChunkFeedForward")

        # 采样执行
        self.log.info(f"进入采样去噪 (步数: {self.config.steps} | CFG: {self.config.cfg} | 种子: {self.config.seed})...")
        sample_out = self.execute_node(
            self.nodes.KSamplerAdvanced(),
            model=model,
            add_noise="enable",
            noise_seed=self.config.seed,
            steps=self.config.steps,
            cfg=self.config.cfg,
            sampler_name="euler",
            scheduler="beta",
            positive=positive_cond,
            negative=negative_cond,
            latent_image=base_latent,
            start_at_step=0,
            end_at_step=10000,
            return_with_leftover_noise="disable"
        )[0]

        # 潜变量落盘持久化
        cache_latent_path = os.path.join(self.base_dir, f"{self.config.output_prefix}_latent.pt")
        torch.save(sample_out, cache_latent_path)
        self.log.info(f"潜变量已持久化落盘: {os.path.basename(cache_latent_path)}")

        # 阶段 2 彻底销毁与显存清扫
        del model
        del positive_cond
        del negative_cond
        del base_latent
        self.memory.purge("阶段2: UNet 主模型与采样图已卸载")

        return sample_out

    def stage3_decode_and_export(self, sample_out: Any):
        """阶段 3：视音频双 VAE 解码与 MP4 容器封装合成"""
        self.log.info("--- [阶段 3/3] 视音频 VAE 解码与最终视频导出 ---")

        # 1. 视频 VAE 解码
        self.log.info(f"载入视频解码 VAE ({self.config.video_vae_name})...")
        video_vae = self.execute_node(
            self.nodes.VAELoader(),
            vae_name=self.config.video_vae_name
        )[0]
        self.log.info("执行视频潜变量解码 (时空自适应分块)...")
        decoded_images = self.execute_node(self.nodes.VAEDecode(), samples=sample_out, vae=video_vae)[0]

        del video_vae
        self.memory.purge("阶段3: 视频 VAE 解码完成并卸载")

        # 2. 音频 VAE 解码
        self.log.info(f"载入音频解码 VAE ({self.config.audio_vae_name})...")
        audio_vae = self.execute_node(
            self.nodes.VAELoader(),
            vae_name=self.config.audio_vae_name
        )[0]
        self.log.info("解码潜空间音频轨道...")
        decoded_audio = self.execute_node(
            self.nodes.NODE_CLASS_MAPPINGS["VAEDecodeAudio"](),
            samples=sample_out,
            vae=audio_vae
        )[0]

        del audio_vae
        del sample_out
        self.memory.purge("阶段3: 音频 VAE 解码完成并卸载")

        # 3. 封装合成 MP4
        self.log.info("合成 MP4 视音频容器 (24 FPS, sRGB)...")
        video_obj = self.execute_node(
            self.nodes.NODE_CLASS_MAPPINGS["CreateVideo"](),
            images=decoded_images,
            fps=24,
            bit_depth=8,
            color_space="sRGB",
            audio=decoded_audio
        )[0]

        save_cls = self.nodes.NODE_CLASS_MAPPINGS["SaveVideo"]
        save_cls.hidden = types.SimpleNamespace(extra_pnginfo=None, prompt=None)
        save_node = save_cls()
        saved = False

        try:
            self.execute_node(
                save_node,
                video=video_obj,
                filename_prefix=f"video/{self.config.output_prefix}",
                format="auto",
                codec={"codec": "auto"}
            )
            saved = True
        except Exception as e:
            self.log.warning(f"SaveVideo 算子导出提示: {e}，启用底层直写通道...")

        if not saved:
            out_dir = os.path.join(self.comfy_dir, "output", "video")
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{self.config.output_prefix}_{int(time.time())}.mp4")
            video_obj.save_to(out_file)
            self.log.info(f"底层直写导出成功: {out_file}")

    def run(self):
        """执行完整流水线调度"""
        start_time = time.time()
        final_length = self.config.calc_frames()
        final_width, final_height = self.config.calc_resolution()
        is_t2v = self.config.is_text_to_video_mode()

        mode_title = "纯文生视频模式 (Text-to-Video / --no-ref)" if is_t2v else "多模态参考转视频模式 (Reference-to-Video)"

        print("=" * 68)
        self.log.info(f">>> 启动 MiniMax H3 视频生成独立流水线 <<<")
        self.log.info(f"模式分类: {mode_title}")
        self.log.info(f"ComfyUI 目录: {self.comfy_dir}")
        self.log.info(f"DiT 主模型: {self.config.unet_name}")
        self.log.info(f"文本编码器: {self.config.clip_name}")
        self.log.info(f"Turbo LoRA: {self.config.turbo_lora_name or '无'}")
        self.log.info(f"目标时长: {self.config.seconds}s -> 对齐帧数: {final_length} 帧")
        self.log.info(f"画幅规格: {self.config.aspect} ({self.config.megapixels} MP) -> 32倍对齐尺寸: {final_width}x{final_height}")
        self.log.info(f"采样参数: 步数={self.config.steps}, CFG={self.config.cfg}, 随机种子={self.config.seed}")
        self.log.info(f"分块设置: QKV上限={self.config.qkv_chunk_size}, FFN分块={self.config.chunks}, 头分块={self.config.head_chunks}")
        print("=" * 68)

        # 断点恢复逻辑
        sample_out = None
        if self.config.resume_latent and os.path.exists(self.config.resume_latent):
            self.log.info(f"检测到断点潜变量缓存: {self.config.resume_latent}，直接进入阶段 3 解码导出")
            try:
                sample_out = torch.load(self.config.resume_latent, weights_only=False)
            except TypeError:
                sample_out = torch.load(self.config.resume_latent)
        else:
            positive_cond, negative_cond, base_latent = self.stage1_extract_conditions(
                final_width, final_height, final_length
            )
            sample_out = self.stage2_sample(positive_cond, negative_cond, base_latent)

        self.stage3_decode_and_export(sample_out)

        elapsed_mins = (time.time() - start_time) / 60.0
        print("=" * 68)
        self.log.info("渲染流水线执行顺利完成！")
        self.log.info(f"输出文件归档至: ComfyUI/output/video/{self.config.output_prefix}_xxxx.mp4")
        self.log.info(f"全流程总计耗时: {elapsed_mins:.2f} 分钟")
        print("=" * 68)


# ==============================================================================
# 8. 命令行入口总线 (H3VideoCLI)
# ==============================================================================
class H3VideoCLI:
    """CLI 应用程序入口与参数调度总控"""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="MiniMax H3 Reference-to-Video (R2V) & Text-to-Video (T2V) 独立命令行执行器 (面向对象版)"
        )
        parser.add_argument("--comfy-dir", type=str, default=None, help="ComfyUI 根目录路径 (可选，默认自动探测同级或上级目录)")
        parser.add_argument("--prompt-file", type=str, default="shot1.txt", help="提示词文本文件名")
        parser.add_argument("--images", nargs="*", default=None, help="参考图片列表 (未显式传入时默认以 --no-ref 纯文生视频模式运行)")
        parser.add_argument("--video", type=str, default=None, help="参考视频文件名 (可选)")
        parser.add_argument("--audio", type=str, default=None, help="参考音频文件名 (可选)")
        parser.add_argument("--seconds", type=float, default=5.0, help="生成时长(秒)")
        parser.add_argument("--aspect", type=str, default="9:16", choices=["9:16", "16:9", "1:1", "4:3", "3:4"], help="画面宽高比")
        parser.add_argument("--megapixels", type=float, default=0.3, help="分辨率档位 (推荐 0.3MP)")
        parser.add_argument("--steps", type=int, default=8, help="采样步数 (推荐 8)")
        parser.add_argument("--cfg", type=float, default=1.0, help="CFG 引导强度 (搭配 Turbo LoRA 推荐 1.0)")
        parser.add_argument("--seed", type=int, default=42, help="随机种子")
        parser.add_argument("--chunks", type=int, default=2, help="FFN/MLP 序列分块数 (推荐 2~8)")
        parser.add_argument("--head-chunks", type=int, default=4, help="Attention 注意力头分块数 (默认 4)")
        parser.add_argument("--qkv-chunk-size", type=int, default=8192, help="纯外挂 QKV 序列分块上限 (默认 8192，杜绝 Level-Zero Error 39/40 崩溃)")
        parser.add_argument("--no-ref", action="store_true", help="显式启用纯文生视频模式 (Text-to-Video)，跳过所有图像/视频/音频参考与视觉编码 VAE")
        parser.add_argument("--ref-image-size", type=str, default="max", choices=["match", "max"], help="参考图尺寸策略：'match' 降采样至输出画幅；'max' 保留原始分辨率")
        parser.add_argument("--output-prefix", type=str, default="H3_R2V_Output", help="生成文件名前缀")
        parser.add_argument("--resume-latent", type=str, default=None, help="跳过阶段1与阶段2，直接加载已有潜变量进行阶段3 VAE解码与视频导出")
        parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="控制台日志等级 (默认: INFO)")

        # 模型权重与 LoRA 统一管理调参配置
        model_group = parser.add_argument_group("模型权重与 LoRA 统一配置 (Model Weights & LoRAs)")
        model_group.add_argument("--unet-name", type=str, default="10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors", help="DiT 扩散主模型名称 (位于 ComfyUI/models/diffusion_models)")
        model_group.add_argument("--clip-name", type=str, default="qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors", help="Qwen3-VL 文本编码器名称 (位于 ComfyUI/models/text_encoders 或 clip)")
        model_group.add_argument("--video-vae-name", type=str, default="minimax_h3_video_vae_int8_convrot.safetensors", help="视频 VAE 解码模型名称 (位于 ComfyUI/models/vae)")
        model_group.add_argument("--audio-vae-name", type=str, default="minimax_h3_audio_vae_fp32.safetensors", help="音频 VAE 解码模型名称 (位于 ComfyUI/models/vae)")
        model_group.add_argument("--turbo-lora-name", type=str, default="minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors", help="Turbo 快速采样加速 LoRA 文件名 (位于 ComfyUI/models/loras)")
        model_group.add_argument("--turbo-lora-strength", type=float, default=1.0, help="Turbo LoRA 权重强度 (默认: 1.0)")
        model_group.add_argument("--lora1-name", type=str, default=None, help="画质或风格增强 LoRA 文件名 (可选，位于 ComfyUI/models/loras)")
        model_group.add_argument("--lora1-strength", type=float, default=1.0, help="LoRA 1 权重强度 (默认: 1.0)")
        return parser

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> int:
        parser = cls.build_parser()
        args = parser.parse_args(argv)

        configure_logging(args.log_level)
        config = H3PipelineConfig.from_namespace(args)

        try:
            bootstrapper = ComfyUIBootstrapper(custom_path=config.comfy_dir)
            pipeline = MiniMaxH3Pipeline(config=config, bootstrapper=bootstrapper)
            pipeline.run()
            return 0
        except KeyboardInterrupt:
            logger.warning("用户主动中止执行中断流水线。")
            return 130
        except Exception as e:
            logger.error(f"执行异常退出: {e}", exc_info=(args.log_level == "DEBUG"))
            return 1


if __name__ == "__main__":
    sys.exit(H3VideoCLI.main())
