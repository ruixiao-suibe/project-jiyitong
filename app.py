"""
技艺通 - 非遗制作技艺深度问答系统
Gradio Web界面
"""

import os
import gradio as gr
from rag_pipeline import init_jiyitong, ask

# ─── 环境变量 ──────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "") or DEEPSEEK_API_KEY
API_BASE = os.environ.get("OPENAI_API_BASE", "")
if not API_BASE and DEEPSEEK_API_KEY:
    API_BASE = "https://api.deepseek.com/v1"

LLM_MODEL = os.environ.get("LLM_MODEL", "")
if not LLM_MODEL:
    LLM_MODEL = "deepseek-chat" if API_BASE == "https://api.deepseek.com/v1" else "gpt-3.5-turbo"

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "shibing624/text2vec-base-chinese")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "craft_techniques.json")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

if API_BASE == "https://api.deepseek.com/v1":
    ACTIVE_PROVIDER = "DeepSeek"
elif API_BASE:
    ACTIVE_PROVIDER = "OpenAI-compatible API"
else:
    ACTIVE_PROVIDER = "Knowledge-only mode"

# ─── 初始化系统 ─────────────────────────────────────────────
qa_chain = None


def format_follow_ups(follow_up_questions: list) -> str:
    if not follow_up_questions:
        return ""
    lines = ["\n\n---\n🔎 **可继续追问：**"]
    for item in follow_up_questions[:3]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def get_qa_chain():
    global qa_chain
    if qa_chain is None:
        qa_chain = init_jiyitong(
            data_path=DATA_PATH,
            index_path=INDEX_PATH,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            api_key=API_KEY or None,
            api_base=API_BASE or None,
        )
    return qa_chain


# ─── 问答函数 ──────────────────────────────────────────────

def answer_question(question: str, history: list, session_state: dict):
    """处理用户提问"""
    history = history or []
    session_state = session_state or {}
    if not question.strip():
        return "", history, session_state
    try:
        chain = get_qa_chain()
        result = ask(chain, question, session_state=session_state)
    except Exception as e:
        error_msg = (
            "问答请求失败，请检查 API Key / 网络 / 模型配置。\n"
            f"错误信息：{type(e).__name__}: {e}"
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return "", history, session_state

    # 构建来源信息
    sources_text = ""
    if result["sources"]:
        sources_text = "\n\n---\n📚 **检索来源：**\n"
        seen = set()
        for s in result["sources"]:
            key = f"{s['name']}-{s['block_type']}-{s.get('step_name', '')}"
            if key not in seen:
                seen.add(key)
                step_label = f" · {s['step_name']}" if s.get("step_name") else ""
                sources_text += f"- {s['name']}{step_label}\n"

    follow_up_text = format_follow_ups(result.get("follow_up_questions", []))
    answer = result["answer"] + sources_text + follow_up_text
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return "", history, result.get("session_state", session_state)


def rebuild_index():
    """重建向量索引"""
    global qa_chain
    from rag_pipeline import init_jiyitong
    try:
        qa_chain = init_jiyitong(
            data_path=DATA_PATH,
            index_path=INDEX_PATH,
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            api_key=API_KEY or None,
            api_base=API_BASE or None,
            rebuild_index=True,
            embedding_backend="offline",
        )
        return "✅ 向量索引重建完成！"
    except Exception as e:
        return f"❌ 重建失败：{type(e).__name__}: {e}"


# ─── 示例问题 ──────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "龙泉青瓷的施釉工序分几步？每步的关键要点是什么？",
    "宣纸捞纸时为什么要加杨桃藤汁？",
    "南京云锦的挑花结本是什么意思？跟编程有什么关系？",
    "武夷岩茶做青工序为什么要摇8-12次？",
    "茅台酒的'12987'工艺具体指什么？",
    "苏绣的劈丝技术最细能到什么程度？",
    "福州脱胎漆器为什么叫'脱胎'？跟普通漆器有什么区别？",
    "建盏的兔毫和油滴是怎么形成的？",
    "徽墨制作为什么要杵捣十万杵？",
    "苗族蜡染的冰裂纹是怎么产生的？",
    "景德镇青花瓷的分水技法为什么能画出深浅？",
    "苗族银饰的银丝最细能拉到多细？",
    "平遥推光漆器为什么要用手掌推光200遍以上？",
    "金华火腿腌制时为什么要上盐5-7次？",
    "雕版印刷刻版时'发刀深、挑刀快、修刀细'是什么意思？",
]


# ─── 界面构建 ──────────────────────────────────────────────

with gr.Blocks(
    title="技艺通 - 非遗制作技艺深度问答",
) as demo:

    gr.Markdown(
        """
        # 🏺 技艺通 — 非遗制作技艺深度问答系统
        > 现有非遗系统只能回答"是什么"，**技艺通首次实现"怎么做"的深度工序问答。**

        基于 RAG（检索增强生成）技术，知识库覆盖20项国家级/人类非遗的制作工序细节。
        """
    )
    gr.Markdown(f"当前智能引擎：**{ACTIVE_PROVIDER}** / 模型：`{LLM_MODEL}`")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            session_state = gr.State({})

            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="请输入关于非遗制作技艺的问题…",
                    show_label=False,
                    scale=5,
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_input,
                label="💡 试试这些问题",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📋 使用说明")
            gr.Markdown(
                """
                **支持的问题类型：**
                - 🔧 工序详解：某非遗的具体制作步骤
                - ❓ 原理追问：为什么这一步要这样做
                - 🔄 技艺对比：两种非遗的技艺异同
                - 🧪 材料科学：工艺背后的物理化学原理
                - 🔁 深度追问：支持“这一步为什么这样做”“如果参数偏了会怎样”
                - 🧭 自动建议：每次回答后自动给出下一步追问思路

                **知识库覆盖：**
                - 陶瓷烧造：龙泉青瓷、景德镇、建盏
                - 纺织织绣：云锦、蜀锦、苏绣
                - 漆艺：福州脱胎、平遥推光
                - 造纸印刷：宣纸、雕版印刷
                - 制茶酿造：武夷岩茶、茅台酒
                - 金属工艺：苗族银饰
                - 文房用品：宣笔、徽墨
                - 食品加工：金华火腿
                - 印染：苗族蜡染
                - 传统医药：同仁堂制剂

                **可替代 OpenAI 的兼容 API：**
                - DeepSeek：`OPENAI_API_BASE=https://api.deepseek.com/v1`
                - 通义千问：`OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`
                - Moonshot：`OPENAI_API_BASE=https://api.moonshot.cn/v1`
                - 只需替换 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`LLM_MODEL`
                """
            )

            rebuild_btn = gr.Button("🔄 重建向量索引", variant="secondary")
            rebuild_output = gr.Textbox(label="状态", visible=True)

    # ─── 事件绑定 ──
    question_input.submit(
        answer_question,
        inputs=[question_input, chatbot, session_state],
        outputs=[question_input, chatbot, session_state],
    )
    submit_btn.click(
        answer_question,
        inputs=[question_input, chatbot, session_state],
        outputs=[question_input, chatbot, session_state],
    )
    rebuild_btn.click(rebuild_index, outputs=[rebuild_output])


# ─── 启动 ──────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css="""
        .title-text { text-align: center; }
        .example-btn { font-size: 13px; }
        """,
    )
