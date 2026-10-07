<div align="center">

# webtorch

### 打开浏览器就能跑大模型：不用装软件，不用配驱动，主流显卡都能用

在一台无风扇的 **MacBook Air（M5，24 GB）** 上用 Chrome 实测：**Qwen3-0.6B 每秒 175 个 token 以上** · **30B MoE 每秒 42 个 token** · **22 层决策模型一道题 15 毫秒**

## ▶ [在线体验：打开就能用](https://xnetsc.github.io/webpytorch/chat/)

<sub>不用安装，也不上传任何数据：权重缓存在浏览器里，模型用你自己的显卡计算。先试
<b>Qwen3-0.6B（0.4 GB）</b>，一分钟内就能开始聊；几个 GB 的大模型也在这个页面里，只是下载时间长一些。</sub>

<img src="images/chat-desktop.png" alt="webtorch 聊天界面：30B MoE 在浏览器里回答，带公式排版和代码高亮" width="900">

<sub>Qwen3-30B-A3B 在浏览器里回答问题：13.8 GB 的权重，不用装任何东西。</sub>

[English](README.md) · **中文** · [在线体验](https://xnetsc.github.io/webpytorch/chat/) ·
[快速开始](#快速开始) · [速度](#速度) · [文档](docs/API.md) ·
技术分享：[中文](docs/articles/webtorch-share.zh.md) · [English](docs/articles/webtorch-share.en.md)

</div>

---

## 亮点

**零环境配置。** 不用装 CUDA、ROCm，不用折腾驱动版本，不用配 Python 环境，也不用编译，打开网页就能跑。同一份着色器代码，浏览器会转成 Metal、Vulkan 或 D3D12，不同厂商、不同驱动之间的差异由浏览器处理。

**主流显卡都能用。** 苹果、英伟达、AMD、Intel，集显、独显都支持，默认走 WebGPU，没有 WebGPU 时自动退到 WebGL2。用到的硬件能力都先检测：每个内核需要的特性、缓冲区数量、共享内存大小，先和显卡实际支持的对一遍，不支持的不用，换一台电脑照样能加载。

**自动选最优路线。** 加载时，每个算子的几种实现（不同的内核、分块方式、线程排布）直接在你的显卡上比一遍，用 GPU 时间戳计时，最快的按显卡记下来，下次加载直接用。没有针对某家厂商写死的调优表，也不需要额外调用任何接口。权重按原始量化格式计算（Q4_K 直接在着色器里按 Q4_K 解码），更快的低精度算法只用在通过精度校验的权重上。

**可重测，可回退。** 设备状态会变，可以随时调用 `remeasure()`，在限定时间内把路线重新比一遍，中途能停，已经比完的结果保留。重测前的那套路线也会留着，用 `switchRoutes()` 随时在两套之间切换；正在用的那套比另一套慢时，`onRoutesSlower` 会发通知。

**数据不出本机。** 权重缓存在浏览器里，对话内容不会离开这台电脑。算力用的是用户自己的显卡，没有按次计费的服务器成本。

**PyTorch 的写法，Python 直接在网页里跑。** CPython 编译成 WebAssembly，在 Web Worker 里运行，`import torch` 就能用：`Tensor`、自动求导、`nn`、`optim` 都有，能加载 GGUF、GPTQ 和 HF 权重，支持 28 种量化格式。

```python
# ── 这段代码直接在浏览器里运行 ──
import webtorch
webtorch.use_default_io()

lm = await webtorch.AutoModelForCausalLM.from_pretrained("/models/qwen3-30b-a3b.gguf")
print(lm.generate("为什么这件事让人意外？", max_new=64))
```

## 速度

全部在浏览器里实测，没有服务器参与：无风扇的 MacBook Air（M5，24 GB），Chrome，WebGPU，贪心解码。

| 模型 | 体积 | 浏览器里的速度 |
|---|---:|---:|
| Qwen3-0.6B · Q4_K_M | 0.4 GB | **每秒 175 个 token 以上** |
| Qwen3-30B-A3B · MoE · UD-Q3_K_XL | 13.8 GB | **每秒 42 个 token** |
| Qwen3.8-27B · 混合 SSM · UD-Q2_K_XL | 9.8 GB | **每秒 7 到 8 个 token** |
| 22 层决策模型 · F16 | 0.7 GB | **一道题 15.6 毫秒** |

同一台机器、同样的请求，和原生 MLX 对比：

<img src="images/vs-mlx-zh.svg" alt="决策模型延迟：Chrome 里的 webtorch 对比同一台 MacBook Air（M5，24 GB）上的原生 MLX，间隔 2 秒问一道题 32~38 毫秒对 35.8 毫秒" width="860">

按日常的用法，隔几秒问一道题，浏览器里的速度和原生 MLX 持平。MLX 更快的那几组，用的是苹果 GPU 的矩阵单元：部分显卡有这类硬件特性（英伟达的 Tensor Core 也是），WebGPU 目前还用不了，所以原生暂时有部分性能优势；随着 WebGPU 补齐这些特性，两者会越来越接近。另外，MLX 只能在苹果芯片上运行，webtorch 同一个页面在英伟达、AMD、Intel 的显卡上也能跑。

## 更多功能

**按配置加载，不限模型列表。** `AutoModelForCausalLM.from_pretrained` 读取模型自带的配置，支持 CausalLM 系（Qwen2、Qwen3、Llama 结构）、MoE 系和混合循环结构；权重可以是 AutoGPTQ 目录、**GGUF** 文件，也可以是普通的 **fp16/bf16 HF** 目录。支持 28 种量化格式，从 `Q4_K` 到 2 bit 的 i-quant。

**直接替换 PyTorch。** `install_torch()` 之后，`import torch` 用的就是 webtorch。`Tensor` 带自动求导，`nn.{Linear, Conv1d/2d/3d, LayerNorm, RMSNorm, MultiheadAttention, …}`，`optim.{SGD, Adam, AdamW}`，在 WebGPU **和** WebGL 上都能训练真实的 GPT、CNN、Transformer。

**结构化决策。** `decide(state, questions)` 回答结构化的问题（单选、是否、打分），每道题都给出完整的概率分布，支持哪些题型由模型配置决定。

**流式量化。** fp16 模型转成 int4/int8，不需要把整个模型读进内存：权重边读边量化，量化好的分片通过你提供的异步 IO 回调边写出。产出的是标准 AutoGPTQ 格式，auto_gptq、vLLM、transformers 都能直接加载。

**通用多模态。** `register_encoder` + `MultimodalLM` 能把**任意**解码器和**任意**媒体编码器组合起来，视觉、音频不和某个模型系列绑定。自带语音合成（CosyVoice2 零样本声音克隆、VITS）、目标检测（YOLO、DETR）和图文理解（Qwen2.5-VL）。

**约束输出格式。** 约束根据已经生成的内容，判断接下来哪些续写合法，所以 `generate(..., constraint="json")` 生成的一定是合法 JSON。约束是一个回调函数，不是固定选项：每一步都会询问回调允许哪些内容，回调也可以直接指定"选这个并结束"或"之后不用再问"。小模型能稳定做**工具调用**靠的就是这套机制：传入 `tools=[...]`，按模型模板里的分隔符解析出调用；开启 `require_known_tools=True` 后，模型根本生成不出不存在的工具名。

**通用 ONNX 运行时。** 加载注册好的 ONNX 计算图，纯 Python 解析，约 50 个算子，无第三方依赖；量化的整数矩阵乘和卷积保持 INT8/UINT8 输入、INT32 累加，不会暗中把存储的张量转成浮点。

**任务流水线，可自由扩展。** `pipeline("text-generation" | "text-to-speech" | "object-detection" | "image-to-text" | …)`。内置任务只是*预先注册*好的加载器，自定义任务不用改 SDK 就能注册。

**IO 由你接管。** 核心不直接读写文件，数据从哪里来由你注册的回调决定：`use_default_io()` 读本地文件，`hf_read()` / `modelscope_read()` 按仓库 id 从 Hugging Face 或魔搭读取，也可以用 `set_io_read` / `set_io_write` 接入任意存储。读过的内容会缓存，支持断点续传，刷新页面后依然有效。

## 聊天应用

[`chat/`](chat/) 目录里是一个完整的本地聊天客户端，界面完全由 SDK 驱动。线上部署的就是这份代码，**[点这里打开](https://xnetsc.github.io/webpytorch/chat/)**，下面介绍的功能都能直接上手用。

<table>
<tr>
<td width="55%" valign="top">
<img src="images/chat-models.png" alt="模型选择：30B MoE、27B 混合、32B 稠密、Gemma、Mistral、gpt-oss 和各种尺寸的 Qwen，也可以选本机文件">
<br><sub><b>模型随便选。</b> 预设覆盖 0.4 GB 到 13.8 GB 的模型（稠密、MoE、混合 SSM），最后三项可以填任意仓库 id，或者选本机的 GGUF 文件、模型目录。</sub>
</td>
<td width="45%" valign="top">
<img src="images/chat-mobile.png" alt="同一段对话在手机屏幕上">
<br><sub><b>手机上也能用。</b> 同一个客户端，排版效果一样；代码块单独横向滚动，不会把页面撑宽。</sub>
</td>
</tr>
</table>

- **显存放得下就能跑。** 可以加载本机的 GGUF 文件或 HF 目录，也可以填模型站上的任意仓库 id。预设列表只是示例，不是白名单，唯一的限制是显存。
- **带图片的结构化决策。** 可选的 Laya Vision 预设，只有在模型声明支持图片输入后才会出现选图入口，调用方式和文字决策一样是 `decide(state, questions)`，同一张图的特征会复用。浏览器版用的是官方指定版本 201M 检查点导出的 FP16 ONNX，不是已经撤回的降质网页版。它的基础模型只声明支持英文，中文输入没有公开发布，也没有验证过。
- **回复排版完整。** 支持 Markdown、代码高亮、KaTeX 公式和表格，内容插入页面前会先做安全过滤。
- **模型能自己运行代码。** Python 和 JavaScript 作为工具提供给模型：模型写出调用，页面执行后把结果返回给模型。提供哪些工具由应用决定，SDK 只负责解析和约束。
- **代码一键运行。** Python 代码块旁边有个 ▶，点一下就在独立的 Pyodide 里运行（和加载模型的那个隔开），输出、报错和 matplotlib 图表直接显示在下面。numpy、pandas、matplotlib 已经预先加载，也能从网址或本机安装其他 wheel。
- **分块编辑。** 段落直接改文字，代码块在块里改，表格按单元格改；公式单元格用 LaTeX 编辑，图片单元格可以直接换图。
- **支持离线。** Service Worker 会永久缓存所有 wheel、wasm 和运行时；应用本身的文件优先从网络获取，联网时更新立即生效。
- **手机上**也能用。

## 快速开始

页面需要设置 COOP/COEP 响应头（`SharedArrayBuffer` 依赖它），浏览器需要支持 GPU 计算：WebGPU（Chrome/Edge 113+、Safari 18+）能达到上面的速度，WebGL 作为兜底。两者都没有时，权重只能放进页面的 WASM 堆（总共约 4 GB），超过 2 GB 左右的模型会在加载阶段内存不足。

```bash
# 1. 构建 WgPy 后端的 wheel，下载 Pyodide（只需一次）→ docs/BUILD.md
# 2. 带上必需的响应头启动服务
node serve-coi.mjs . 8119
# 3. 打开 http://localhost:8119/chat/   （示例运行页是 /webapp/）
```

完整步骤（包括权重从哪里下载）见 **[docs/BUILD.md](docs/BUILD.md)**。

## 目录结构

```
webtorch/            SDK
  _core.py             兼容 torch 的 Tensor、自动求导、nn、optim，以及 GPU 内核
  _sdk.py              transformers 风格的上层接口和任务流水线注册表
  llm.py               加载、预填充和解码、KV 复用、对话模板、工具调用
  lm_engine.py         通用解码器（稠密 + MoE）、采样器、录制和回放
  decision.py          基于编码器的结构化决策
  constrain.py         输出约束（回调、json、正则、选项等）
  toolcall.py          按模型自己的格式读写工具调用
  quantize.py          流式量化（核心不做 IO）
  webio.py             唯一的 IO 层：全局回调、模型站读取、缓存管理
  onnxrt.py            通用 ONNX 运行时
  torchshim.py         `import torch` 兼容层
chat/                聊天应用（index.html、app.js、cache-sw.js、pyworker.js）
tetris/              xDecision 自动玩俄罗斯方块，直接调用游戏自己的函数（见 tetris/README.md）
webtorch-sw.js       SDK 的 Service Worker：为静态托管提供跨源隔离
webapp/              示例运行页
examples/            可运行的示例
docs/                API.md · SDK_README.md · ARCHITECTURE.md · BUILD.md · WGPY_BACKEND.md · articles/
src/ webgl/ webgpu/ wgpy/ cupy/ cupyx/     内置的 WgPy 后端（有改动，见 NOTICE）
```

权重（`models/`）、Pyodide 运行时（`lib/`）和构建产物不在 git 里。

## 文档

- [SDK 用法](docs/SDK_README.md)：torch、大模型、量化、流水线
- [API 参考](docs/API.md)
- [架构](docs/ARCHITECTURE.md)：webtorch 如何构建在 WgPy 之上
- [构建](docs/BUILD.md)：后端、模型、运行演示
- [WgPy 后端](docs/WGPY_BACKEND.md)
- 技术分享：设计思路、踩过的坑和原因，[中文](docs/articles/webtorch-share.zh.md) · [English](docs/articles/webtorch-share.en.md)

## 许可证

MIT，见 [LICENSE](LICENSE)。本项目基于 **WgPy**（© 东京大学、Edge Intelligence Systems, Inc.；MIT）开发，修改了它的 WebGPU/WebGL 后端：批量矩阵乘、计算图录制和回放、融合内核、量化矩阵乘内核。详见 [NOTICE](NOTICE)。

聊天应用的渲染用了 [marked](https://github.com/markedjs/marked)、[DOMPurify](https://github.com/cure53/DOMPurify)、[KaTeX](https://katex.org) 和 [highlight.js](https://highlightjs.org)，运行时从 CDN 加载。
