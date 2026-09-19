# MiniMax H3 独立命令行执行器 (`h3_cli.py`)

> **轻量 · 纯外挂 · 零侵入 · 专为 32GB 统一内存 / Intel Arc 核显优化的百亿级视频生成方案。**  
> 告别浏览器后台内存泄漏与显存溢出，一键命令行即可自动流转：**多模态特征编码 -> Turbo 快速采样 -> 视音频双 VAE 解码 -> MP4 容器合成导出**。

---

## 一、本机配置详情与模型体量 (Environment & Model Sizes)

### 1. 本机实测硬件规格 (Hardware)

| 组件类别 | 规格参数 | 运行说明 |
| :--- | :--- | :--- |
| **处理器 (CPU)** | Intel Core Ultra 7 255H | 移动端高性能处理器 |
| **显卡 / 核显 (GPU)** | Intel Arc 140T GPU (16GB) | Xe2 / XPU 架构，共享动态上限约 25GB |
| **显卡驱动版本** | 32.0.101.8991 | 支持 Intel Level-Zero 底层接口 |
| **系统内存 (RAM)** | 32 GB (物理可用约 31.4 GB) | 统一内存架构 (UMA，CPU 与核显动态共享) |
| **操作系统 (OS)** | Windows 11 专业工作站版 64-bit | Build 10.0.28120 (Insider Preview) |
| **Python / 框架** | Python 3.13.14 / PyTorch 2.13.0+xpu | 便携版 `python_embeded`，Intel XPU 加速 |

### 2. 当前测试配置与本地模型全景对照表 (Active Test Config & Models Reference)

本项目已实现模型参数完全解耦，可在命令行随时动态覆盖。根据 `h3_cli.py` 当前代码实测默认配置与本地模型库，对照整理如下：

#### (1) 当前脚本默认激活测试套件 (Active Baseline Config in `h3_cli.py`)

不传入任何覆盖参数时，直接执行 `python h3_cli.py` 默认生效的黄金极速组合：

| 模块类别 | 对应 CLI 参数 | 当前默认采用模型文件名 | 文件大小 | 量化/精度 | 当前阶段峰值显存/角色 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **扩散主模型** | `--unet-name` | `10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors` | **20.97 GB** | INT8 ConvRot | 阶段 2 主体，约 20.9GB 显存，融合 Turbo 蒸馏加速与边缘跳连 |
| **文本编码器** | `--clip-name` | `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | **14.61 GB** | NVFP4 极致量化 | 阶段 1 主体，显存消耗最低 (~14.6GB)，多模态抽取速度最快 |
| **加速 LoRA** | `--turbo-lora-name` | `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | **0.17 GB** | BF16 (Rank 19) | 强度 1.0，赋能 **3 步极速采样**，大幅缩短去噪迭代轮次 |
| **视频 VAE** | `--video-vae-name` | `minimax_h3_video_vae_int8_convrot.safetensors` | **2.95 GB** | INT8 ConvRot | 阶段 1 参考图编码 + 阶段 3 潜变量视频解码，显存开销小且防爆 |
| **音频 VAE** | `--audio-vae-name` | `minimax_h3_audio_vae_fp32.safetensors` | **0.56 GB** | FP32 全精度 | 阶段 3 潜变量音频轨道解码，生成原生对齐音效 |
| **画质 LoRA 1** | `--lora1-name` | `None` (默认关闭) | — | — | 可选扩展槽位，用于风格化或人像微距增强 |

---

#### (2) 本地实测模型库全景对照表 (Local Test Models Matrix)

已在本地 `ComfyUI/models/` 目录就绪、可供当前多轮调试与横向对比的所有模型：

| 模型分类与目录 | 本地文件名 / 别名 | 实际大小 | 量化规格 | 开源来源 / 仓库链接 | 测试角色、适配场景与特性 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **扩散主模型**<br>`diffusion_models/` | `10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors` | **20.97 GB** | INT8 ConvRot | [HuggingFace: cicalooo](https://huggingface.co/cicalooo/10Eros-Max-h3-int8-convrot/blob/main/10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors) | ⭐ **当前默认主模型**。Turbo 蒸馏与跳连边缘双重加持，32G UMA 稳定性最佳。 |
| **扩散主模型**<br>`diffusion_models/` | `DasiwaMinimaxH3_dasiwaHybridV1_int8.safetensors` | **19.53 GB** | INT8 ConvRot | [Civitai: dasiwa-minimax-h3](https://civitai.com/models/2877206/dasiwa-minimax-h3) | 韩国社区精调，针对多模态 R2V 增强，行级混合量化，人像质感细腻。 |
| **扩散主模型**<br>`diffusion_models/` | `DasiwaMinimaxH3_dasiwaHybridV1_int4.safetensors` | **11.68 GB** | INT4 极限压缩 | [Civitai: dasiwa-minimax-h3](https://civitai.com/models/2877206/dasiwa-minimax-h3) | **极限低显存主模型**。体积减少超 44%，大幅降低内存带宽吞吐压力。 |
| **扩散主模型**<br>`diffusion_models/` | `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors` | **19.53 GB** | INT8 Pruned | [HuggingFace: WarmBloodAban](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity/blob/main/Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors) | 经典精简裁剪版 INT8，结构精练，适合标准 R2V 对比测试。 |
| **扩散主模型**<br>`diffusion_models/` | `minimax_h3_fl2va_pruned_nvfp4.safetensors` | **11.67 GB** | NVFP4 浮点量化 | 社区 NVFP4 优化版 | 4位浮点量化版本，显存占用极小，适合测试极限显存边界。 |
| **文本编码器**<br>`text_encoders/` | `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | **14.61 GB** | NVFP4 极致量化 | 本地 NVFP4 优化 | ⭐ **当前默认编码器**。阶段 1 内存占用仅 ~15GB，加载卸载极速。 |
| **文本编码器**<br>`text_encoders/` | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | **24.55 GB** | INT8 ConvRot | [HuggingFace: ethanfel](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot/blob/main/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors) | 未审查全保真版，语义理解与视觉多模态还原度极高，适合复杂长难 Prompt。 |
| **文本编码器**<br>`text_encoders/` | `qwen3vl_8b_abliterated_fp8_scaled.safetensors` | **9.86 GB** | 8B FP8 Scaled | 社区轻量版 | 8B 轻量级蒸馏版本，体量仅 9.86GB，适合文生视频极速冒烟测试。 |
| **加速 LoRA**<br>`loras/` | `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | **0.17 GB** | BF16 (Rank 19) | [HuggingFace: Kijai](https://huggingface.co/Kijai/MiniMax-H3_comfy/blob/main/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors) | ⭐ **当前默认 Turbo LoRA**。推荐 **3 步采样**，15 秒视频约 43 分钟出片。 |
| **加速 LoRA**<br>`loras/` | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` | **0.58 GB** | BF16 EMA Pruned | [HuggingFace: drbaph](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors) | 经典 Turbo 蒸馏 LoRA，推荐 **4~8 步采样**，动作连贯度与光影稳定性优秀。 |
| **加速 LoRA**<br>`loras/` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | **1.82 GB** | BF16 4-Step | 官方/社区 4步专训版 | 专为参考转视频设计的 4 步蒸馏 LoRA，动作幅度与参考保真度兼备。 |
| **加速 LoRA**<br>`loras/` | `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | **1.82 GB** | BF16 8-Step | 官方/社区 8步768P版 | 专为 768P 高分辨率微调的 8 步 Turbo LoRA，画面细致锐利。 |
| **风格 LoRA**<br>`loras/` | `MysticXXX_MMH3-V4.safetensors` | **0.14 GB** | BF16 微调 | 社区人像微调 | 可选画质细节与肤质微距增强 LoRA，搭配 `--lora1-name` 挂载使用。 |
| **视频 VAE**<br>`vae/` | `minimax_h3_video_vae_int8_convrot.safetensors` | **2.95 GB** | INT8 ConvRot | [HuggingFace: Kijai](https://huggingface.co/Kijai/MiniMax-H3_comfy) | ⭐ **当前默认视频 VAE**。显存占用极小，完美契合 32GB 统一内存防爆要求。 |
| **视频 VAE**<br>`vae/` | `minimax_h3_video_vae_fp16.safetensors` | **4.85 GB** | FP16 半精度 | [HuggingFace: Kijai](https://huggingface.co/Kijai/MiniMax-H3_comfy) | 原生半精度视频 VAE，色彩过渡更细腻，适合显存充足时的高清解码。 |
| **音频 VAE**<br>`vae/` | `minimax_h3_audio_vae_fp32.safetensors` | **0.56 GB** | FP32 全精度 | [HuggingFace: Kijai](https://huggingface.co/Kijai/MiniMax-H3_comfy) | ⭐ **当前默认音频 VAE**。用于解码潜变量音轨并与视频合成封装 MP4。 |

---

#### (3) 典型测试方案组合与一键调用预设 (Testing Presets)

根据不同调试目的，可通过命令行自由搭配以下 4 种测试方案：

| 方案定位 | 核心模型搭配 (UNet + CLIP + LoRA + VAE) | 推荐参数 | 预期单步耗时 / 显存压力 | 适用调试场景 |
| :--- | :--- | :--- | :--- | :--- |
| **方案 A：极速出片基准**<br>*(当前默认推荐)* | • 10Eros INT8<br>• Qwen3-VL NVFP4<br>• Taomate 3-Step LoRA<br>• INT8 Video VAE | `--steps 3`<br>`--chunks 4`<br>`--cfg 1.0` | 采样总耗时 ~32 分钟<br>显存占用适中 (~21GB) | 日常测试、长镜头预览、快速出片验证 |
| **方案 B：极限低显存/轻量测试** | • Dasiwa INT4 (或 NVFP4)<br>• Qwen3-VL 8B (或 NVFP4)<br>• Taomate 3-Step LoRA<br>• INT8 Video VAE | `--steps 3`<br>`--unet-name DasiwaMinimaxH3_dasiwaHybridV1_int4.safetensors`<br>`--clip-name qwen3vl_8b_abliterated_fp8_scaled.safetensors` | 显存峰值降低 30%~40%<br>整机极其流畅 | 后台挂机测试、多任务并行、防 OOM 极限验证 |
| **方案 C：高保真/画质微调测试** | • Dasiwa INT8 (或 10Eros)<br>• Qwen3-VL INT8 ConvRot (24.5G)<br>• Turbo 8Step / v4 LoRA<br>• FP16 Video VAE | `--steps 8`<br>`--clip-name qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors`<br>`--turbo-lora-name minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`<br>`--video-vae-name minimax_h3_video_vae_fp16.safetensors` | 阶段 1 耗时约 4 分钟<br>阶段 2 耗时约 85 分钟 | 复杂提示词遵循度对比、高动态光影效果验证 |
| **方案 D：风格/人像画质增强** | • 10Eros INT8<br>• Qwen3-VL NVFP4<br>• Taomate 3-Step LoRA<br>• **挂载 MysticXXX-V4 LoRA** | `--steps 3`<br>`--lora1-name MysticXXX_MMH3-V4.safetensors`<br>`--lora1-strength 0.8` | 仅增加微量 LoRA 融合开销<br>耗时基本不变 | 真实感人像、微距特写、皮肤质感增强对比 |


> **内存关键挑战**：模型总权重 (~48GB) 远超本机物理内存上限 (32GB UMA)。传统 WebUI 常驻方式会直接因 OOM 或页面交换卡死；本执行器通过三阶段物理内存隔离，确保任意时刻整机显存开销稳控在 20GB 以内。

### 3. 本机实测基准对照 (Benchmark: 3步极速 vs 8步标准)

基于 Intel Arc 140T 核显在 PowerShell 终端运行 **15.0 秒超长视频（362 帧，416x736 竖屏，CFG 1.0）** 的实测数据对照：

| 执行阶段与指标 | 方案 A：8 步 Turbo 采样 | 方案 B：3 步 Taomate 极速采样 *(终端实测)* | 效率提升说明 |
| :--- | :---: | :---: | :--- |
| **命令行调用** | `--steps 8` | `--steps 3 --chunks 4` | 3 步蒸馏 LoRA 极大削减去噪迭代轮次 |
| **阶段 1：多模态特征抽取** | 2.6 分钟 | **2.18 分钟** | 提取完成后显存立即物理归零 (剩 2 MB) |
| **阶段 2：DiT 降噪采样** | 85.1 分钟 (~10.6 min/步) | **32.73 分钟 (~10.9 min/步)** | **采样时间大幅缩减 61.5%** |
| **阶段 3：双 VAE 解码导出** | 6.6 分钟 | **6.45 分钟** | 视频 VAE 解码 6.1 分钟 + 音频与 MP4 封装 15 秒 |
| **全流程总计耗时** | **94.3 分钟** (约 1.57 小时) | **43.00 分钟** | **总耗时减少超 51 分钟，出片效率提升 219%！** |
| **内存/显存状态** | 全程无 OOM，显存 100% 回收 | 全程无 OOM，显存 100% 回收 | 潜变量均自动落盘为 `xxx_latent.pt` |

> [!TIP]
> - **日常推荐**：生成 15 秒超长视频首选 `--steps 3`（搭配 `minimax_h3_taomate_3step_lora`），43 分钟即可完整出片；
> - **5 秒短片推算**：若生成标准 5.0 秒（121 帧）视频，3 步采样全流程预计仅需 **14~16 分钟** 即可完成！

---

## 二、脚本定位：它干什么、为什么做、解决什么问题？

### 1. 它主要干什么？
`h3_cli.py` 是一个**脱离浏览器与 WebUI 的独立命令行执行管道**。它直接调用 ComfyUI 底层算子与模型加载器，以最小开销全自动串联执行：
`读取提示词/参考图 -> Qwen3-VL 提取多模态特征 -> DiT Turbo 快速采样 -> 视音频双 VAE 解码 -> 容器合成导出 MP4`。

### 2. 为什么这样做？
- **摆脱 Web 界面常驻开销**：传统 ComfyUI 启动后，浏览器前端长连接、画布图状态及节点缓存会常驻锁定大量张量，在 32GB 紧凑设备上极易诱发 Windows 虚拟内存爆满。
- **外挂解耦，零侵入**：不修改 ComfyUI 任何内核代码，不更改任何第三方插件源码，以纯独立封装器方式运行，随用随开，用完即退，物理归还系统所有内存。

### 3. 核心解决了什么问题？
1. **解决 48GB 级模型塞入 32GB 内存的 OOM 难题（三阶段物理隔离）**：
   - **阶段 1（多模态编码）**：仅载入 Qwen3-VL (24.55GB) 与编码 VAE，提取完成后**物理级销毁实例，强制触发垃圾回收与操作系统工作集清空 (`EmptyWorkingSet`)**，将内存完全归零；
   - **阶段 2（UNet 采样去噪）**：在纯净显存中载入 DiT 主模型 (19.53GB) 与 Turbo LoRA 采样，采样结束**立即将潜变量保存到磁盘 (`xxx_latent.pt`)** 并彻底卸载主模型；
   - **阶段 3（双 VAE 解码）**：轻量加载视音频 VAE 解码潜变量，封装导出 MP4。三个阶段内存峰值完全隔离，整机平稳运行。
2. **解决 Intel 显卡驱动崩溃问题（外挂 Attention 分块）**：
   - 针对 Intel Level-Zero 驱动在长序列自注意力运算下易报 Error 39 / 40 的问题，运行时内存级注入 8192 QKV 分块热补丁，杜绝驱动闪退。
3. **解决长流程中断重跑的痛点（断点恢复机制）**：
   - 采样完成即落盘保存潜变量文件。若后续解码或合成环节被中断，可通过 `--resume-latent` 参数直奔阶段 3 解码，节省 90% 以上重复计算时间。

---

## 三、精简用法 (Quickstart)

在 `ComfyUI_windows_portable` 根目录下直接打开终端（PowerShell）运行：

### 1. 常用命令示例

```powershell
# 1. 极速 3 步生成（当前默认最速工作流，15秒视频约 43 分钟出片）
.\python_embeded\python.exe h3_cli.py --images "./1.png" --seconds 15.0 --steps 3 --chunks 4

# 2. 纯文生视频（最简模式，默认读取 shot1.txt 或 prompt.txt，跳过视觉编码，最省显存）
.\python_embeded\python.exe h3_cli.py --no-ref --seconds 5.0 --steps 3

# 3. 自定义切换主模型（例如切换为 Dasiwa 或 Singularity）
.\python_embeded\python.exe h3_cli.py --unet-name DasiwaMinimaxH3_dasiwaHybridV1_int8.safetensors --steps 3

# 4. 自定义切换文本编码器与加速 LoRA（切换为 Qwen3-VL INT8 与 Turbo v4 LoRA）
.\python_embeded\python.exe h3_cli.py --clip-name qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors --turbo-lora-name minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors --steps 6

# 5. 注入画质增强 LoRA 1
.\python_embeded\python.exe h3_cli.py --images "./1.png" --lora1-name MysticXXX_MMH3-V4.safetensors --lora1-strength 0.8 --steps 3

# 6. 断点恢复（读取已有潜变量文件，跳过前两阶段，直接进入 VAE 解码与封装导出）
.\python_embeded\python.exe h3_cli.py --resume-latent H3_R2V_Output_latent.pt
```

### 2. 核心参数与模型调参速查

#### 基础生成参数
| 参数名 | 默认值 | 推荐可选值 | 用途与说明 |
| :--- | :---: | :---: | :--- |
| `--prompt-file` | `shot1.txt` | txt 文件名 | 提示词文件（不存在时自动顺延查找 `prompt.txt`） |
| `--images` | 无 | 文件路径列表 | 指定参考图；未传时默认以纯文生视频模式运行 |
| `--seconds` | `5.0` | `3.0` ~ `15.0` | 视频时长（秒），脚本按 H3 算式自动对齐合法帧数 |
| `--aspect` | `9:16` | `9:16`, `16:9`, `1:1` | 画面画幅宽高比 |
| `--megapixels` | `0.3` | `0.3` 或 `0.5` | 分辨率档位（32G 内存设备推荐 **0.3**，416x736） |
| `--steps` | `8` | `3` (Taomate) / `4~8` (Turbo v4) | 采样步数（配合 3 步 LoRA 推荐设置为 **3**） |
| `--cfg` | `1.0` | `1.0` | 引导强度（搭配 Turbo LoRA 推荐 **1.0**，免除负向无用开销） |
| `--seed` | `42` | 任意整数 | 随机数种子 |
| `--chunks` | `2` | `2` ~ `8` | FFN/MLP 序列分块数（高负载 15 秒推荐 `4`） |
| `--head-chunks` | `4` | `4` 或 `8` | Attention 头分块数（默认 4） |
| `--qkv-chunk-size`| `8192` | `4096` / `8192` | QKV 序列分块硬上限（防 Level-Zero 闪退） |
| `--no-ref` | 关闭 | 标志参数 | 显式强制纯文生视频模式，跳过参考图加载 |
| `--resume-latent` | 无 | `.pt` 路径 | 断点模式：跳过前两阶段，直接加载已有潜变量重跑解码 |
| `--log-level` | `INFO` | `DEBUG`, `INFO` | 控制台日志等级 |

#### 模型权重统一管理调参参数
| 参数名 | 默认值 | 所属模型目录 | 用途与说明 |
| :--- | :--- | :--- | :--- |
| `--unet-name` | `10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors` | `ComfyUI/models/diffusion_models` | DiT 扩散主模型文件名 |
| `--clip-name` | `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | `text_encoders` 或 `clip` | Qwen3-VL 文本/多模态特征编码器文件名 |
| `--video-vae-name` | `minimax_h3_video_vae_int8_convrot.safetensors` | `ComfyUI/models/vae` | 视频特征编码与潜变量解码 VAE 模型文件名 |
| `--audio-vae-name` | `minimax_h3_audio_vae_fp32.safetensors` | `ComfyUI/models/vae` | 音频特征编码与音轨解码 VAE 模型文件名 |
| `--turbo-lora-name` | `minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors` | `ComfyUI/models/loras` | Turbo 蒸馏加速 LoRA 文件名（传空则不挂载） |
| `--turbo-lora-strength` | `1.0` | — | Turbo 加速 LoRA 融合强度 |
| `--lora1-name` | `None` (空) | `ComfyUI/models/loras` | 画质或风格扩展 LoRA 1 文件名（可选） |
| `--lora1-strength` | `1.0` | — | LoRA 1 融合强度 |

---

## 四、文件目录与输出位置

```text
ComfyUI_windows_portable/
├── python_embeded/                # 便携版内置 Python 环境
├── ComfyUI/
│   ├── models/                    # 模型权重存放根目录
│   │   ├── text_encoders/         # Qwen3-VL 文本编码器 (NVFP4 / INT8)
│   │   ├── diffusion_models/      # DiT 扩散主模型 (10Eros / Dasiwa / Singularity)
│   │   ├── vae/                   # 视频 VAE (2.95 GB) 与 音频 VAE (0.56 GB)
│   │   └── loras/                 # Taomate 3步 LoRA / Turbo v4 LoRA / 风格 LoRA
│   └── output/video/              # 生成的 MP4 视频默认保存于此
├── h3_cli.py                      # 本独立执行器主脚本
├── shot1.txt / prompt.txt         # 提示词文本
└── H3_R2V_Output_latent.pt        # 阶段 2 采样后自动落盘的潜变量文件（支持断点恢复）
```

---

## 五、常见排错 (FAQ)

1. **未检测到 MiniMaxH3ReferenceToVideo 节点**：检查 `ComfyUI/comfy_extras/nodes_minimax_h3.py` 是否存在，确保 ComfyUI 官方内核已更新。
2. **提示显存不足或运行迟缓**：
   - 保持 `--megapixels 0.3` 分辨率；
   - 保持 `--cfg 1.0`，削减 50% 负向特征计算；
   - 使用 `--steps 3` 搭配 `minimax_h3_taomate_3step_lora`，仅需 3 步即可快速出片；
   - 运行前退出浏览器等多标签高内存占用程序。
3. **切换模型后报错找不到文件**：
   - 请确认模型文件存放在对应目录下（主模型放 `models/diffusion_models/`，LoRA 放 `models/loras/`，编码器放 `models/text_encoders/` 或 `models/clip/`），参数中只需传入文件名即可。

---

## 六、致谢开源贡献者 (Acknowledgements)

- **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**：提供模块化算子图执行架构。
- **MiniMax 团队与 H3 社区**：提供突破性的百亿级多模态视频生成模型与潜空间架构。
- **Qwen 团队 (Alibaba Cloud)**：提供卓越的 Qwen3-VL 多模态文本编码底座。
- **[cicalooo (10Eros-Max-h3)](https://huggingface.co/cicalooo/10Eros-Max-h3-int8-convrot)**：提供高质量 10Eros Turbo 混合量化主模型。
- **[Dasiwa (Civitai)](https://civitai.com/models/2877206/dasiwa-minimax-h3)**：提供出色的 Dasiwa H3 多模态混合量化版本。
- **[WarmBloodAban (Singularity)](https://huggingface.co/WarmBloodAban/Minimax-h3_Singularity)**：提供精简裁剪版 Singularity INT8 主模型。
- **[ethanfel (Qwen3-VL Ultra Heretic)](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot)**：提供 ConvRot 未审查文本编码底座。
- **[Kijai (comfyui-kjnodes / MiniMax-H3_comfy)](https://github.com/kijai/comfyui-kjnodes)**：开发精湛的注意力分块、Taomate 3步 LoRA 及 VAE 方案。
- **[drbaph (MiniMax-H3-Turbo-Lora)](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI)**：提供经典 4~8 步 Turbo 蒸馏加速方案。
- **PyTorch 与 Intel oneAPI / Level-Zero 社区**：为 Windows 与 Intel Arc 统一内存架构提供底层计算支持。
