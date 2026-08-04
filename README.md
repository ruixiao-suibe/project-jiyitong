# 技艺通 - 非遗制作技艺深度问答系统

本 README 已按当前代码与实际运行状态更新。

当前具备能力：

* DeepSeek 及其他 OpenAI 兼容 API 接入
* L1-L5 工艺问答（概述、类目、工序、参数、原理）
* 多轮上下文追问（如“这一步为什么这样做”）
* 自动追问建议（每轮 3 条）
* 原理问题：检索证据 + 受约束 LLM 拓展
* 索引重建支持离线后端，确保可生成本地 `faiss\_index`

\---

## 1\. 目录结构（实际）

```text
jiyitong/
├── app.py
├── rag\_pipeline.py
├── rag\_pipeline .py        # 旧文件（带空格），不参与运行
├── craft\_techniques.json
├── faiss\_index/            # 已生成：index.faiss / index.pkl / embedding\_backend.json
├── requirements.txt
├── README.md
└── \_\_init\_\_.py
```

> 运行入口文件是 `app.py` 与 `rag\_pipeline.py`。

\---

## 2\. 快速启动

### 2.1 安装依赖

```powershell
Set-Location "c:/Users/Admin/Desktop/开源大语言模型应用/jiyitong"
c:/Users/Admin/Desktop/开源大语言模型应用/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2.2 启动（DeepSeek 推荐）

```powershell
Set-Location "c:/Users/Admin/Desktop/开源大语言模型应用/jiyitong"
$env:DEEPSEEK\_API\_KEY="your-deepseek-key"
c:/Users/Admin/Desktop/开源大语言模型应用/.venv/Scripts/python.exe app.py
```

访问：

```text
http://127.0.0.1:7860
```

### 2.3 OpenAI 兼容 API 配置规则

系统优先级：

1. `OPENAI\_API\_KEY`
2. `DEEPSEEK\_API\_KEY`

当设置了 `DEEPSEEK\_API\_KEY` 且未设置 `OPENAI\_API\_BASE` 时，系统自动使用：

* `OPENAI\_API\_BASE=https://api.deepseek.com/v1`
* `LLM\_MODEL=deepseek-chat`

兼容提供方示例：

|提供方|OPENAI\_API\_BASE|模型示例|
|-|-|-|
|DeepSeek|`https://api.deepseek.com/v1`|`deepseek-chat`|
|通义千问|`https://dashscope.aliyuncs.com/compatible-mode/v1`|`qwen-plus`|
|Moonshot|`https://api.moonshot.cn/v1`|`moonshot-v1-8k`|

\---

## 3\. 索引重建

### 3.1 当前行为

界面“重建向量索引”按钮已改为：

* 强制 `embedding\_backend="offline"`
* 不依赖 HuggingFace 在线下载
* 成功后写入 `faiss\_index/` 下 3 个文件：

  * `index.faiss`
  * `index.pkl`
  * `embedding\_backend.json`

### 3.2 手动重建命令

```powershell
Set-Location "c:/Users/Admin/Desktop/开源大语言模型应用/jiyitong"
$env:DEEPSEEK\_API\_KEY="your-deepseek-key"
c:/Users/Admin/Desktop/开源大语言模型应用/.venv/Scripts/python.exe -c "import app; print(app.rebuild\_index())"
```

### 3.3 验证是否生成

```powershell
Get-ChildItem "c:/Users/Admin/Desktop/开源大语言模型应用/jiyitong/faiss\_index"
```

应看到：`index.faiss`, `index.pkl`, `embedding\_backend.json`。

\---

## 4\. 功能说明

### 4.1 问答层级

|层级|示例|当前状态|
|-|-|-|
|L1|龙泉青瓷是什么？|支持|
|L2|中国有哪些陶瓷类非遗？|支持|
|L3|龙泉青瓷施釉分几步？|支持|
|L4|施釉时比重控制多少？|支持|
|L5|为什么强还原焰让铁发青色？|支持（证据+拓展）|

### 4.2 多轮追问

支持上下文指代：

* 这一步为什么要这样做？
* 如果这个参数偏了会怎样？
* 该工艺还有哪些关键原理？

每次回答自动生成 3 条可继续追问建议。

\---

## 5\. 当前核心流程

```text
用户提问
  -> app.py(answer\_question)
  -> rag\_pipeline.py(answer)
    -> 上下文改写(\_resolve\_question\_with\_context)
    -> 问题类型判定(L1-L5)
    -> 检索(FAISS 或 LocalKeywordRetriever)
    -> 回答(规则回答 + LLM受约束拓展)
    -> 生成 follow\_up\_questions
  -> 前端展示：答案 + 来源 + 追问建议
```

\---

## 6\. 关键代码位置

* DeepSeek 自动配置：`app.py`
* 重建按钮离线后端：`app.py` -> `rebuild\_index()`
* 索引构建与离线哈希向量：`rag\_pipeline.py` -> `build\_vectorstore()`
* 索引加载后端识别：`rag\_pipeline.py` -> `load\_vectorstore()`
* 多轮追问上下文：`rag\_pipeline.py` -> `\_resolve\_question\_with\_context()`
* 原理类受约束拓展：`rag\_pipeline.py` -> `\_llm\_grounded\_expansion()`

\---

## 7\. 故障排查

### 7.1 429 / 配额不足

* 现象：接口返回配额不足
* 处理：更换可用 key 或切换兼容提供方

### 7.2 索引重建后无文件

* 当前版本应写入 `index.faiss/index.pkl/embedding\_backend.json`
* 若未出现，请确认运行目录是 `jiyitong/`，并重试手动重建命令

### 7.3 模型下载相关错误

* 当前重建默认离线后端，不应依赖 HuggingFace 在线下载
* 若仍遇到问题，先关闭正在运行的旧进程再重建

\---

## 8\. 许可证

MIT

