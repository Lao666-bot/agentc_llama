# agentc

> 为嵌入式边缘设备设计的开箱即用 Agent 框架  
> 专为 FunctionGemma 系列模型优化，无需强大生成能力即可完成本地工具调用

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-embedded%20%7C%20linux%20%7C%20macos-lightgrey)]()
[![Language](https://img.shields.io/badge/language-C-blue)]()

基于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 的轻量级 C 语言 Agent 框架，专注于**函数调用（Function Calling）** 和**多轮对话记忆**。  
专为边缘场景设计——不需要强大的生成能力，只需要准确执行工具。本项目**主要适配并维护 FunctionGemma 系列模型**。

---

## 🎯 设计理念

- **边缘优先**：2.7 亿参数的 FunctionGemma 仅需约 550MB 内存，可运行在手机、开发板、嵌入式设备上
- **动作优先**：不追求长篇对话，专注于将自然语言转化为结构化的工具调用
- **零外部依赖**：纯 C 实现，仅依赖 llama.cpp 和标准系统库
- **开箱即用**：提供 `libagent.a` 静态库，可直接嵌入任何 C/C++ 项目

---

## 📁 目录结构

```
agentc/
├── Inc/                         # 公共头文件
│   ├── agent.h                  # Agent API
│   ├── tools.h                  # 工具注册、解析、执行
│   └── memory.h                 # 会话记忆管理
├── Src/                         # 实现代码
│   ├── agent.c                  # Agent 核心：生成、推理、函数调用循环
│   ├── tools.c                  # 内置工具（write_file 等）
│   └── memory.c                 # 对话历史的管理、裁剪、保存/加载
├── example/
│   ├── main.c                   # 交互式命令行对话示例
│   └── test.c                   # 测试入口（可扩展）
├── agent.conf                   # 默认配置文件
├── Makefile                     # GNU Make 构建（推荐）
├── CMakeLists.txt               # CMake 构建（可选）
└── README.md
```

---

## ✨ 关键特性

- **Agent 抽象**  
  封装模型加载、上下文初始化、采样配置、记忆管理，对外仅需 `agent_create`, `agent_chat`, `agent_destroy` 三个核心 API。

- **会话记忆**  
  基于 `memory_t` 维护完整对话历史，支持自动裁剪和 JSONL 持久化。

- **函数调用**  
  - 使用 `<start_function_call>call:func_name{key:value,...}<end_function_call>` 格式
  - 自动解析、执行、回传工具结果，继续模型推理
  - 内置重复调用检测，防止死循环

- **内置工具**  
  `write_file(filename, content)` —— 将文本内容写入磁盘文件

- **配置驱动**  
  通过 `agent.conf` 管理所有参数，无需重新编译

---

## 📦 依赖

| 依赖 | 说明 |
|------|------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | 需提前编译为静态库 |
| GCC / Clang | 支持 C11 |
| OpenMP | 用于 CPU 并行推理 |
| Linux 系统库 | `pthread`, `m`, `dl`, `stdc++`, `stdc++fs` |

---

## 🚀 快速开始

### 1. 编译 llama.cpp

```bash
cd /path/to/llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CURL=OFF
make -j$(nproc)
```

### 2. 构建 agentc

```bash
cd /path/to/agentc
make
```

### 3. 下载模型

下载 FunctionGemma 的 GGUF 文件（如 `functiongemma-270m-it-IQ4_XS.gguf`）。

### 4. 配置并运行

编辑 `agent.conf`，指向你的模型文件：

```ini
model_path      = ../models/functiongemma-270m-it-IQ4_XS.gguf
n_ctx           = 4096
n_threads       = 4
temperature     = 0.0
top_p           = 0.9
verbose         = false
```

运行：
```bash
./agent_app agent.conf
```

**对话示例：**
```
> write "Hello, world" to file demo.txt
[call write_file(filename:"demo.txt", content:"Hello, world") = File 'demo.txt' written successfully]

> what file did I just create?
You created "demo.txt" with the content "Hello, world".
```

---

## 🧑‍💻 API 使用

```c
#include "agent.h"

int main() {
    // 从配置文件创建
    Agent *agent = agent_create_from_config("agent.conf");

    // 或直接指定参数
    // Agent *agent = agent_create("model.gguf", 4096, 4, 0.0f, 0.9f, false);

    const char *reply = agent_chat(agent, "Hello");
    printf("%s\n", reply);

    agent_destroy(agent);
    return 0;
}
```

链接：`-lagent` + llama.cpp 静态库（参考 Makefile）。

---

## 🔧 添加新工具

1. 在 `Src/tools.c` 中实现工具函数（匹配 `tool_executor_t` 签名）
2. 在 `tools_init()` 中注册
3. 重新 `make`

工具声明会自动出现在 system prompt 中。

---

## 🧪 微调建议

未微调的 FunctionGemma 准确率约 60%。推荐使用 [Unsloth](https://github.com/unslothai/unsloth) 进行 LoRA 微调，10~20 条对话示例即可提升至 85%+。微调后导出 GGUF 并更新 `agent.conf`。

---

## ⚠️ 注意事项

- 本框架**主要维护 FunctionGemma 系列模型**，不保证其他模型的兼容性
- 工具执行为同步无沙盒，生产环境请添加安全控制
- 会话历史存于内存，长期运行建议配合持久化策略


---

## 🤝 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Google FunctionGemma](https://ai.google.dev/gemma)
- [Unsloth](https://unsloth.ai)

---

*轻量 · 边缘 · 可控 — 让设备自己完成任务。*
```
