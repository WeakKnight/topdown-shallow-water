# Top-down shallow water

面向 MOBA 的俯视双足涉水原型。**Slang 负责 GPU 水面模拟、脚步扰动及折射/反射；SlangPy 负责资源、窗口与 UI；Python 生成角色步态和 NPC 路线。** 默认 1 个主角 + 3 个 NPC，无外部模型、贴图、数据集或子模块依赖。

这是带阻尼的高度场波动原型，不是完整浅水方程（SWE）求解器。实现、光学近似和后续接入方式见 [技术说明](docs/architecture.md)。

## 环境要求

- **CPython 3.12–3.14，推荐 3.13**。`requirements.txt` 固定全部运行时依赖版本；其中 NumPy 要求 Python 3.12+。
- Git，以及首次安装时访问 PyPI 的网络连接。
- 支持相应后端的 GPU 和驱动。**Headless 和测试也会执行真实 GPU shader，没有 CPU 渲染回退。**

| 系统 | 默认后端 | 安装注意事项 |
| --- | --- | --- |
| macOS | Metal | 当前 SlangPy 0.43.1 wheel 要求 **macOS 26+、Apple Silicon（ARM64）**；Metal 编译还需要 Xcode 及可用的 Metal 工具链 |
| Windows x86-64 | D3D12 | 使用 64 位 Python，安装支持 D3D12 的显卡驱动 |
| Linux x86-64 | Vulkan | SlangPy wheel 要求 glibc 2.34+；需要 Vulkan 驱动/runtime，交互窗口另需桌面会话 |

以上是本仓库安装流程面向的平台。macOS 26.3 / Apple Silicon / Python 3.13 / Metal 已实际验证；Windows/Linux 的运行入口已提供，尚未在本项目中实机验证。上游的 [SlangPy 0.43.1 安装包列表](https://pypi.org/project/slangpy/0.43.1/#files)及 [SlangPy 安装要求](https://slangpy.shader-slang.org/en/latest/)可用于核对系统兼容性。不要把旧 macOS、Intel Mac 或不匹配的 Python 当作相同配置。

不需要单独编译 Slang、安装 CUDA/PyTorch，或克隆 DDDDGI。视频编码器由 `imageio-ffmpeg` 的安装包提供，常见支持平台无需系统 FFmpeg。

## 从克隆到运行

### macOS / Linux

先安装 Python 3.13，再执行：

```bash
git clone https://github.com/WeakKnight/topdown-shallow-water.git
cd topdown-shallow-water
python3.13 scripts/setup_env.py --check
./run.sh
```

### Windows（PowerShell 或 cmd）

安装 64 位 Python 3.13 及 Python Launcher 后执行：

```powershell
git clone https://github.com/WeakKnight/topdown-shallow-water.git
cd topdown-shallow-water
py -3.13 scripts/setup_env.py --check
.\run.bat
```

安装脚本只在仓库内创建 `.venv`，从固定版本清单安装二进制包，检查依赖、模块导入和 FFmpeg。`--check` 还会编译 shader 并渲染 `output/setup-check.png`；安装阶段没有可用 GPU 时可省略该参数。脚本可重复执行，不会删除现有环境；更换 Python 版本时先将旧 `.venv` 重命名为已忽略的 `.venv-backup`，再重新安装。

无需激活虚拟环境。安装完成后只需运行 `./run.sh` 或 `.\run.bat`，所有命令行参数都会传递给 `app.py`。macOS/Linux 若 Python 命令名是 `python3`，先确认版本在上面的范围内，再替换 `python3.13`。

### 手动安装（可选）

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python app.py
```

Windows 对应命令中的 Python 路径为 `.venv\Scripts\python.exe`。已有独立环境可直接用它执行 `app.py`，也可把 `WATER_PYTHON` 设为该解释器的绝对路径。启动脚本可以从其他目录调用；自定义的相对 `--output` / `--video` 路径相对于调用目录，默认输出始终在仓库的 `output/`。

## 体验双足交互

主角橙色为左脚，蓝色为右脚；另外 3 个 NPC 分别使用绿色、紫色、金色。默认 Zoom 为 **1.4**，Damping 为 **1.3**。主角默认自动行走，镜头只跟随主角。NPC 在水域中独立选择随机目的地，以不同速度、步频和短暂停步错开节奏，并检查整段路线以避开岸和岛屿。随机种子固定，重置后可复现；`--npcs 0` 可以恢复单人场景。两只脚轮流抬起、前摆、入水、踩稳；支撑脚在世界空间保持固定。停止移动后完成当前步态并收脚，随后仅保留水波自行传播与衰减。

| 操作 | 效果 |
| --- | --- |
| WASD | 手动步行，退出自动演示，速度 1.4 m/s |
| Shift + WASD | 跑动，速度 3.8 m/s，自动提高步频 |
| 鼠标左键点击场景 | 行走到点击的世界坐标 |
| 滚轮 | 调整镜头缩放 |
| Tab | 自动演示开关 |
| B | 显示双脚投影轮廓、摆动脚目标落点和角色中心 |
| F | 镜头跟随 / 固定在场景中心 |
| T | 倾斜正交 / 垂直俯视 |
| V | 水面 / 高度 / 速度与活动度 / 遮罩 / 折射 / 反射 / Fresnel |
| I | 开关折射弯曲，保留水下吸收 |
| O | 开关环境、局部物体反射与太阳高光 |
| G | 开关水底 25 cm 标定网格 |
| Space | 在角色中心施加一次独立测试脉冲 |
| P | 暂停 / 继续 |
| R | 重置水面、所有角色、随机路线和模拟时间 |
| F2 | 保存不含 UI 的 PNG |
| Esc | 退出 |

控制面板显示主角左右脚的 `SWING / PLANTED`、`WET / DRY` 状态。可调节波速、阻尼、总交互强度、水深、入水强度、离水回填、水中拖动、步频、抬脚高度、缩放、法线强度和表面细节。小窗口可滚动面板。

建议先关闭 Surface detail，按 B 打开脚掌轮廓，观察每次入水与波圈的关系；然后按 V 看不受材质高光影响的原始高度场。轮廓在浸水时采用左右脚颜色，完全离水时为黄色；十字标记是下一落脚点。

右侧 Water optics 面板控制折射率、粗糙度、吸收密度和太阳方向。按 **P 暂停 → G 显示底部网格 → I 开关折射**，可以比较同一波纹下网格和水下脚掌轮廓的变化。按 O 对比反射，或通过 V 查看未经 Fresnel 权重缩放的反射画面。Fresnel 视图直接显示 0–1 权重，不做曝光/gamma 转换；顶视清水反射本来就较弱。

## 常用命令

以下示例在仓库根目录执行。Windows 把 `./run.sh` 换为 `.\run.bat`，把 `.venv/bin/python` 换为 `.venv\Scripts\python.exe`。

```bash
# 列出全部参数。
./run.sh --help

# 无窗口运行 4 秒，保存截图、波高/速度场及光学调试数据。
./run.sh --headless --frames 240 --save-state --output output/water.png

# 只保留静止主角，检查静水（--no-demo 本身不会停止 NPC）。
./run.sh --headless --no-demo --npcs 0 --frames 120 --output output/still.png

# 原始波高视图，移除装饰性细节。
./run.sh --view height --top-down --foot-debug --no-detail

# 固定镜头查看完整水域。
./run.sh --no-follow --zoom 1

# 性能较低的机器可降低模拟网格和窗口尺寸。
./run.sh --resolution 384 --width 960 --height 600
```

Headless 的 `--frames` 每帧推进 1/60 秒，内部继续使用更小的固定子步；它不打开窗口，但仍需 GPU。默认模拟网格 `768×768`，覆盖 `24×24 m`，水深 `0.12 m`，Zoom `1.4`，Damping `1.3`，NPC 数 `3`。窗口尺寸是逻辑像素，HiDPI 显示器会提高实际渲染分辨率；headless 和视频尺寸是实际像素。

## 录制 MP4

```bash
./run.sh --video output/water-4-characters-20s.mp4 --seconds 20 --fps 60 --width 1920 --height 1080
```

输出 **20 秒、1080p、60 fps 的 H.264 / yuv420p MP4**，不含 UI 和音轨，相机按当前设置跟随主角。这会逐帧运行真实模拟和 Slang 渲染，不需要打开窗口或录屏软件；导出耗时不影响视频速度。支持 30/60 fps，宽高必须为偶数，时长按帧数取整。中间帧 PNG 会保存在视频旁边；完成时校验视频帧数和时长。

截图默认写入 `output/water.png`；`--save-state` 另存同名 NPZ。`output/`、MP4 和 NPZ 都不纳入版本控制。示例使用固定输出名，重复执行会覆盖同名结果；要保留多份结果请修改路径。

## 验证安装与修改

```bash
# 全部数值、动画、NPC 和真实 GPU 光学测试。
.venv/bin/python -m unittest discover -v

# 窗口、UI 和呈现测试：创建最小化窗口，20 帧后退出。
./run.sh --window-frames 20 --debug
```

当前共 22 项测试。全部测试和上面的 GPU 冒烟验证已在 macOS / Metal 跑通。仅运行不需要 GPU 的配置、步态和随机路线测试：

```bash
.venv/bin/python -m unittest -v test_water.ConfigurationTests test_water.GaitChecks test_crowd.CrowdChecks.test_defaults_and_seeded_safe_walks
```

这组测试仍需要安装 Python 依赖，但不会创建设备。完整测试不适合没有 GPU 的普通 CI runner。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| `Python environment not found` / `No module named slangpy` | 先运行安装脚本；用仓库启动脚本或 `.venv` 里的 Python，避免用系统 Python 启动 |
| `No matching distribution found` | 核对 Python 版本、CPU 架构和 OS；尤其 SlangPy 0.43.1 的 Mac wheel 需要 macOS 26+ / ARM64。安装脚本有意只接受 wheel，不自动尝试耗时的源码构建 |
| Linux 缺少 `venv` / `ensurepip` | 安装发行版中与所用 Python 对应的 `python3-venv` 或版本化 venv 包，再重新创建环境 |
| macOS 报 Metal 编译器不可用 | 安装并选择 Xcode 开发工具链；用 `xcrun --find metal` 检查，按 Xcode 提示补装 Metal Toolchain |
| 创建设备失败或 shader 编译失败 | 更新显卡驱动，确认当前机器支持默认后端；也可显式传入 `--backend metal`、`d3d12` 或 `vulkan`，但后端必须由当前系统支持 |
| 远程环境无法创建窗口 | 使用 `--headless` 或 `--video`；它们只省去窗口，不能省去 GPU 和驱动 |
| 视频编码器找不到 | 重新运行安装脚本检查 FFmpeg。特殊平台可安装兼容 FFmpeg，并用 `IMAGEIO_FFMPEG_EXE` 指向可执行文件 |
| `run.sh: Permission denied` | 用 `bash run.sh`；正常 Git 克隆会保留该脚本的可执行位 |

## 仓库布局与维护

```text
app.py / water.py         窗口、输入、GPU 资源和调度
gait.py / crowd.py        主角步态和 NPC 随机行走
recording.py              MP4 导出
shaders/                  水面模拟、场景、脚步、折射与反射
test_*.py                 配置、动画、GPU 回归测试
scripts/setup_env.py      标准库实现的环境安装入口
requirements.txt         固定版本的运行时依赖（含视频导出）
run.sh / run.bat          macOS/Linux 与 Windows 启动入口
docs/architecture.md      模拟、光学原理和实现边界
output/                  自动生成的本地输出，不提交
.venv/                   本地虚拟环境，不提交
```

安装后不应产生待提交的依赖、缓存或媒体文件。`.gitignore` 覆盖虚拟环境、Python/测试缓存、编辑器和 OS 文件、环境变量文件及生成的视频/模拟数据；`.gitattributes` 固定脚本换行，避免跨平台拉取带来的 CRLF 问题。需要纳入版本控制的文档图片请放进 `docs/`，不要放进被忽略的 `output/`。

更新依赖时显式修改 `requirements.txt`，在新环境里重新安装并执行完整测试、窗口冒烟测试和短视频导出。无需 `pip install -e .`，无需访问作者电脑上的任何路径。已有的本地视频和截图可继续保留，它们不会被 Git 纳入提交。
