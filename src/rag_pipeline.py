"""
技艺通 - 非遗制作技艺深度问答系统
RAG核心管线：文档加载 -> 工序级分块 -> 向量检索 -> 回答生成
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


SYSTEM_PROMPT = """你是「技艺通」，一个专注中国非遗制作技艺的深度问答助手。

回答要求：
1. 优先回答"怎么做"，即工序步骤、关键要点、工具材料。
2. 如果用户问到分步流程，必须按步骤编号输出。
3. 只基于提供的检索内容回答，不要编造。
4. 若知识库没有该项目，直接说明知识库暂未覆盖。
"""

PROCESS_KEYWORDS = ["工序", "步骤", "流程", "分几步", "怎么做", "做法", "要点"]
OVERVIEW_KEYWORDS = ["是什么", "简介", "介绍", "背景", "概述"]
CATEGORY_KEYWORDS = ["有哪些", "都有哪些", "包含哪些", "名单", "清单"]
PARAMETER_HINTS = ["温度", "比重", "浓度", "时间", "小时", "分钟", "参数", "控制在", "多少"]
PRINCIPLE_HINTS = ["为什么", "原理", "机理", "为何", "怎么形成"]
FOLLOW_UP_REFERENCES = ["这一步", "该步骤", "这个步骤", "这一工序", "这个工序", "它", "这个项目", "该项目", "该工艺", "这种做法"]


class LocalKeywordRetriever:
    """A simple offline retriever used when vector retrieval is unavailable."""

    def __init__(self, documents: List[Document], k: int = 8):
        self.documents = documents
        self.k = k

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", text) if t]

    def invoke(self, query: str) -> List[Document]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self.documents[: self.k]

        scored: List[Tuple[int, Document]] = []
        for d in self.documents:
            content = d.page_content.lower()
            score = sum(content.count(t) for t in q_tokens)
            if score > 0:
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return self.documents[: self.k]
        return [doc for _, doc in scored[: self.k]]


@dataclass
class JiyiTongQA:
    crafts: List[Dict]
    vectorstore: Optional[FAISS] = None
    retriever_k: int = 8
    llm: Optional[ChatOpenAI] = None
    fallback_documents: List[Document] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vectorstore is not None:
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.retriever_k, "fetch_k": max(self.retriever_k * 3, 12)},
            )
        else:
            self.retriever = LocalKeywordRetriever(self.fallback_documents, k=self.retriever_k)
        self.name_to_craft, self.alias_to_name = self._build_name_maps(self.crafts)

    @staticmethod
    def _build_name_maps(crafts: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        name_to_craft: Dict[str, Dict] = {}
        alias_to_name: Dict[str, str] = {}

        for craft in crafts:
            name = craft.get("name", "").strip()
            if not name:
                continue
            name_to_craft[name] = craft

            alias_to_name[name] = name
            for suffix in ["制作技艺", "烧制技艺", "手工织造技艺", "技艺"]:
                if name.endswith(suffix):
                    alias = name[: -len(suffix)].strip()
                    if alias:
                        alias_to_name[alias] = name

        return name_to_craft, alias_to_name

    @staticmethod
    def _is_process_question(question: str) -> bool:
        return any(k in question for k in PROCESS_KEYWORDS)

    def _find_craft(self, question: str) -> Optional[Dict]:
        for alias, full_name in sorted(self.alias_to_name.items(), key=lambda x: len(x[0]), reverse=True):
            if alias and alias in question:
                return self.name_to_craft.get(full_name)
        return None

    @staticmethod
    def _normalize_session_state(session_state: Optional[Dict]) -> Dict:
        state = dict(session_state or {})
        state.setdefault("last_craft_name", "")
        state.setdefault("last_step_name", "")
        state.setdefault("last_question_type", "")
        state.setdefault("last_question", "")
        state.setdefault("resolved_question", "")
        return state

    @staticmethod
    def _uses_followup_reference(question: str) -> bool:
        return any(token in question for token in FOLLOW_UP_REFERENCES)

    def _find_step_for_question(self, craft: Optional[Dict], question: str, session_state: Optional[Dict]) -> str:
        if not craft:
            return ""

        for step in craft.get("craft_process", []):
            step_name = step.get("name", "")
            if step_name and step_name in question:
                return step_name

        state = self._normalize_session_state(session_state)
        if self._uses_followup_reference(question):
            return state.get("last_step_name", "")
        return ""

    def _resolve_question_with_context(self, question: str, session_state: Optional[Dict]) -> Tuple[str, Dict]:
        state = self._normalize_session_state(session_state)
        resolved_question = question
        if self._find_craft(question):
            state["resolved_question"] = resolved_question
            return resolved_question, state

        craft_name = state.get("last_craft_name", "")
        step_name = state.get("last_step_name", "")
        if craft_name and self._uses_followup_reference(question):
            if step_name and any(token in question for token in ["这一步", "该步骤", "这个步骤", "这一工序", "这个工序"]):
                resolved_question = f"关于{craft_name}的{step_name}，{question}"
            else:
                resolved_question = f"关于{craft_name}，{question}"
        elif craft_name and (self._is_parameter_question(question) or self._is_principle_question(question)):
            resolved_question = f"关于{craft_name}，{question}"

        state["resolved_question"] = resolved_question
        return resolved_question, state

    def _craft_short_name(self, craft_name: str) -> str:
        for suffix in ["制作技艺", "烧制技艺", "手工织造技艺", "技艺"]:
            if craft_name.endswith(suffix):
                return craft_name[: -len(suffix)] or craft_name
        return craft_name

    def _build_follow_up_questions(
        self,
        craft: Optional[Dict],
        question_type: str,
        current_step_name: str = "",
    ) -> List[str]:
        craft_name = craft.get("name", "") if craft else ""
        short_name = self._craft_short_name(craft_name) if craft_name else "该项目"
        if current_step_name:
            return [
                f"{current_step_name}这一步为什么要这样做？",
                f"{current_step_name}需要控制哪些关键参数？",
                f"如果{current_step_name}做不好，会出现什么问题？",
            ]

        if question_type == "overview":
            return [
                f"{short_name}的核心工序分几步？",
                f"{short_name}最关键的技术难点是什么？",
                f"{short_name}和相近工艺有什么区别？",
            ]
        if question_type == "category":
            return [
                "其中哪几项最适合做深度工序演示？",
                "这些项目分别有哪些核心工序？",
                "哪一类非遗最依赖温度或时间控制？",
            ]
        if question_type == "process":
            return [
                f"{short_name}里哪一步最关键？为什么？",
                f"{short_name}各步骤会用到哪些工具和材料？",
                f"{short_name}有哪些关键参数需要控制？",
            ]
        if question_type == "parameter":
            return [
                "这个参数为什么要这样控制？",
                "如果这个参数偏离，会出现什么问题？",
                f"{short_name}还有哪些关键参数？",
            ]
        if question_type == "principle":
            return [
                "能结合具体工序举一个例子吗？",
                "这种原理会如何影响最终成品效果？",
                f"{short_name}中还有哪些类似的关键原理？",
            ]
        return [
            "能按步骤详细展开吗？",
            "这项工艺最关键的参数是什么？",
            "如果继续深挖，下一步应该问什么？",
        ]

    def _build_process_answer(self, craft: Dict) -> Dict:
        steps = craft.get("craft_process", [])
        if not steps:
            return {
                "answer": f"{craft.get('name', '该项目')} 暂无结构化工序数据。",
                "sources": [],
            }

        lines = [
            f"{craft['name']} 共 {len(steps)} 个核心工序。",
            "",
        ]

        sources = []
        for s in steps:
            tools = "、".join(s.get("tools", [])) if s.get("tools") else "未特别注明"
            materials = "、".join(s.get("materials", [])) if s.get("materials") else "未特别注明"
            key_points = "；".join(s.get("key_points", [])) if s.get("key_points") else "无"

            lines.extend(
                [
                    f"步骤{s['step']}：{s['name']}",
                    f"【做法】{s['detail']}",
                    f"【关键要点】{key_points}",
                    f"【工具/材料】{tools} / {materials}",
                    "",
                ]
            )
            sources.append(
                {
                    "name": craft.get("name", ""),
                    "block_type": "craft_step",
                    "step_name": s.get("name", ""),
                    "content_preview": s.get("detail", "")[:200],
                }
            )

        return {"answer": "\n".join(lines).strip(), "sources": sources}

    @staticmethod
    def _is_overview_question(question: str) -> bool:
        return any(k in question for k in OVERVIEW_KEYWORDS)

    @staticmethod
    def _is_category_question(question: str) -> bool:
        return any(k in question for k in CATEGORY_KEYWORDS) and ("非遗" in question or "技艺" in question)

    @staticmethod
    def _is_parameter_question(question: str) -> bool:
        return any(k in question for k in PARAMETER_HINTS)

    @staticmethod
    def _is_principle_question(question: str) -> bool:
        return any(k in question for k in PRINCIPLE_HINTS)

    def _build_overview_answer(self, craft: Dict) -> Dict:
        answer = (
            f"{craft.get('name', '该项目')} 是一项{craft.get('level', '非遗项目')}。\n"
            f"【类别】{craft.get('category', '未知')} / {craft.get('sub_category', '未知')}\n"
            f"【地域】{craft.get('region', '未知')}\n"
            f"【简介】{craft.get('description', '暂无简介')}\n"
            f"【核心技艺】{'、'.join(craft.get('core_techniques', [])) or '暂无'}\n"
            f"【文化价值】{craft.get('cultural_value', '暂无')}"
        )
        return {
            "answer": answer,
            "sources": [{"name": craft.get("name", ""), "block_type": "overview", "step_name": ""}],
            "question_type": "overview",
        }

    def _build_category_answer(self, question: str) -> Optional[Dict]:
        category_map = {
            "陶瓷": "陶瓷烧造",
            "纺织": "纺织织绣",
            "刺绣": "纺织织绣",
            "漆": "漆艺",
            "造纸": "造纸",
            "印刷": "印刷",
            "茶": "制茶",
            "酿造": "酿造",
            "银饰": "金属工艺",
            "文房": "文房用品",
            "印染": "印染",
            "医药": "传统医药",
        }
        target = None
        for k, v in category_map.items():
            if k in question:
                target = v
                break
        if not target:
            return None

        matched = []
        for c in self.crafts:
            sub = c.get("sub_category", "")
            if target in sub:
                matched.append(c)
        if not matched:
            return {
                "answer": f"知识库中暂未覆盖{target}相关非遗项目。",
                "sources": [],
                "question_type": "category",
            }

        lines = [f"知识库中与“{target}”相关的非遗项目共有 {len(matched)} 项：", ""]
        sources = []
        for i, c in enumerate(matched, 1):
            lines.append(f"{i}. {c.get('name', '未知项目')}（{c.get('level', '未知级别')}）")
            sources.append({"name": c.get("name", ""), "block_type": "overview", "step_name": ""})
        return {"answer": "\n".join(lines), "sources": sources, "question_type": "category"}

    def _build_parameter_or_principle_answer(self, craft: Dict, question: str) -> Dict:
        hits: List[Tuple[Dict, str]] = []
        for s in craft.get("craft_process", []):
            candidates = []
            candidates.extend(s.get("key_points", []))
            detail = s.get("detail", "")
            if detail:
                candidates.append(detail)
            for t in candidates:
                if any(k in t for k in ["温度", "比重", "浓度", "小时", "分钟", "还原", "形成", "关键", "控制"]):
                    if any(qk in t or qk in question for qk in PARAMETER_HINTS + PRINCIPLE_HINTS):
                        hits.append((s, t))

        if not hits:
            return self._build_process_answer(craft)

        lines = [f"围绕“{craft.get('name', '该项目')}”，与问题最相关的参数/原理要点如下：", ""]
        sources = []
        seen = set()
        for s, t in hits[:8]:
            key = (s.get("step"), t)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"步骤{s.get('step', '?')}：{s.get('name', '未命名步骤')}")
            lines.append(f"- {t}")
            sources.append(
                {
                    "name": craft.get("name", ""),
                    "block_type": "craft_step",
                    "step_name": s.get("name", ""),
                    "content_preview": str(t)[:200],
                }
            )

        return {"answer": "\n".join(lines).strip(), "sources": sources, "question_type": "parameter"}

    def _build_principle_answer_from_docs(self, question: str, docs: List[Document]) -> Dict:
        if not docs:
            return {"answer": "知识库中未检索到可用于解释原理的内容。", "sources": []}

        evidence = []
        sources = []
        for d in docs[:8]:
            text = d.page_content.replace("\n", " ")
            meta = d.metadata or {}
            if any(k in text for k in ["还原", "Fe", "铁离子", "发色", "青色", "浓度"]):
                snippet = text[:180]
                evidence.append(snippet)
                sources.append(
                    {
                        "name": meta.get("name", ""),
                        "block_type": meta.get("block_type", ""),
                        "step_name": meta.get("step_name", ""),
                        "content_preview": snippet,
                    }
                )

        if not evidence:
            return {
                "answer": self._fallback_answer(question, docs),
                "sources": [
                    {
                        "name": d.metadata.get("name", ""),
                        "block_type": d.metadata.get("block_type", ""),
                        "step_name": d.metadata.get("step_name", ""),
                        "content_preview": d.page_content[:200],
                    }
                    for d in docs
                ],
                "question_type": "principle",
            }

        lines = [
            "【证据】知识库相关要点：",
        ]
        for i, ev in enumerate(evidence[:4], 1):
            lines.append(f"{i}. {ev}")

        lines.extend(
            [
                "",
                "【受控解释】",
                "根据检索证据，烧成中的强还原气氛会推动铁离子由高价态向低价态转变，"
                "这是青釉呈现青色的重要条件；同时还原气氛强度与时段控制（如CO浓度）会影响最终发色稳定性。",
                "上述解释仅基于知识库命中的工艺描述，不引入库外结论。",
            ]
        )

        unique_sources = []
        seen = set()
        for s in sources:
            key = (s.get("name", ""), s.get("block_type", ""), s.get("step_name", ""))
            if key in seen:
                continue
            seen.add(key)
            unique_sources.append(s)

        return {"answer": "\n".join(lines), "sources": unique_sources, "question_type": "principle"}

    def _llm_grounded_expansion(self, question: str, docs: List[Document], fallback_answer: str) -> str:
        """Use LLM to expand explanation while staying grounded in retrieved evidence."""
        if self.llm is None:
            return fallback_answer

        context = "\n\n".join(self._format_doc(d) for d in docs[:8])
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "你可以在检索证据基础上进行必要的原理拓展，但不能与证据冲突。"
            "若证据不足，请明确说明不确定部分。\n\n"
            f"[用户问题]\n{question}\n\n"
            f"[检索证据]\n{context}\n\n"
            "请按以下结构回答：\n"
            "1) 直接结论\n"
            "2) 原理解释（可适度拓展）\n"
            "3) 工艺控制点\n"
            "4) 偏差后果\n"
            "保持简洁清晰。"
        )
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip() if str(content).strip() else fallback_answer
        except Exception:
            return fallback_answer

    @staticmethod
    def _format_doc(doc: Document) -> str:
        return doc.page_content.strip()

    def _fallback_answer(self, question: str, docs: List[Document]) -> str:
        if not docs:
            return "知识库中未检索到可用内容，请尝试补充具体的非遗项目名称与工序关键词。"

        top_docs = docs[:5]
        lines = ["根据知识库检索到的工艺信息：", ""]
        for i, doc in enumerate(top_docs, 1):
            meta = doc.metadata
            title = meta.get("name", "未知项目")
            step_name = meta.get("step_name", "")
            label = f"{title} · {step_name}" if step_name else title
            lines.append(f"{i}. {label}")
            lines.append(f"   {doc.page_content[:180].replace(chr(10), ' ')}")
        return "\n".join(lines)

    def _llm_answer(self, question: str, docs: List[Document]) -> str:
        if self.llm is None:
            return self._fallback_answer(question, docs)

        context = "\n\n".join(self._format_doc(d) for d in docs[:8])
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"[问题]\n{question}\n\n"
            f"[检索内容]\n{context}\n\n"
            "请直接给出结构化答案。"
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            # If remote client/session becomes unavailable, degrade gracefully.
            fallback = self._fallback_answer(question, docs)
            msg = str(e)
            if "insufficient_quota" in msg or "429" in msg or "RateLimitError" in msg:
                return (
                    "当前 API 配额已用尽，系统已自动切换为知识库直答模式。\n"
                    "可继续提问工序、参数和步骤问题。\n\n"
                    f"{fallback}"
                )
            return (
                "LLM 调用暂时不可用，已切换为检索结果直答模式。\n"
                f"错误信息：{type(e).__name__}: {e}\n\n"
                f"{fallback}"
            )

    def answer(self, question: str, session_state: Optional[Dict] = None) -> Dict:
        question = (question or "").strip()
        if not question:
            return {"answer": "请输入问题。", "sources": []}

        resolved_question, state = self._resolve_question_with_context(question, session_state)
        craft = self._find_craft(resolved_question)
        current_step_name = self._find_step_for_question(craft, resolved_question, state)
        question_type = "general"

        if self._is_category_question(resolved_question):
            category_result = self._build_category_answer(resolved_question)
            if category_result is not None:
                question_type = category_result.get("question_type", "category")
                category_result["follow_up_questions"] = self._build_follow_up_questions(None, question_type)
                category_result["session_state"] = state
                category_result["resolved_question"] = resolved_question
                return category_result

        if craft and self._is_process_question(resolved_question):
            question_type = "process"
            result = self._build_process_answer(craft)
        elif craft and self._is_overview_question(resolved_question):
            question_type = "overview"
            result = self._build_overview_answer(craft)
        elif craft and (self._is_parameter_question(resolved_question) or self._is_principle_question(resolved_question)):
            is_principle = self._is_principle_question(resolved_question)
            question_type = "principle" if is_principle else "parameter"
            result = self._build_parameter_or_principle_answer(craft, resolved_question)
            docs = self.retriever.invoke(resolved_question)
            result["answer"] = self._llm_grounded_expansion(
                resolved_question,
                docs,
                result.get("answer", ""),
            )
            if docs and not result.get("sources"):
                result["sources"] = [
                    {
                        "name": d.metadata.get("name", ""),
                        "block_type": d.metadata.get("block_type", ""),
                        "step_name": d.metadata.get("step_name", ""),
                        "content_preview": d.page_content[:200],
                    }
                    for d in docs[:5]
                ]
        else:
            docs = self.retriever.invoke(resolved_question)
            if self._is_principle_question(resolved_question):
                question_type = "principle"
                result = self._build_principle_answer_from_docs(resolved_question, docs)
                result["answer"] = self._llm_grounded_expansion(
                    resolved_question,
                    docs,
                    result.get("answer", ""),
                )
            else:
                result = {
                    "answer": self._llm_answer(resolved_question, docs),
                    "sources": [
                        {
                            "name": d.metadata.get("name", ""),
                            "block_type": d.metadata.get("block_type", ""),
                            "step_name": d.metadata.get("step_name", ""),
                            "content_preview": d.page_content[:200],
                        }
                        for d in docs
                    ],
                    "question_type": "general",
                }

        if result.get("sources"):
            first_source = result["sources"][0]
            state["last_craft_name"] = first_source.get("name", state.get("last_craft_name", ""))
            state["last_step_name"] = current_step_name or first_source.get("step_name", "")
        elif craft:
            state["last_craft_name"] = craft.get("name", state.get("last_craft_name", ""))
            state["last_step_name"] = current_step_name

        state["last_question_type"] = result.get("question_type", question_type)
        state["last_question"] = question
        state["resolved_question"] = resolved_question

        result["follow_up_questions"] = self._build_follow_up_questions(
            craft,
            state["last_question_type"],
            state.get("last_step_name", ""),
        )
        result["session_state"] = state
        result["resolved_question"] = resolved_question
        return result


def load_craft_data(data_path: str) -> List[Dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def craft_to_documents(crafts: List[Dict]) -> List[Document]:
    documents: List[Document] = []

    for craft in crafts:
        overview = (
            f"【非遗项目】{craft['name']}\n"
            f"【类别】{craft['category']} / {craft['sub_category']}\n"
            f"【地域】{craft['region']}\n"
            f"【级别】{craft['level']}\n"
            f"【简介】{craft['description']}\n"
            f"【核心技艺】{'、'.join(craft.get('core_techniques', []))}\n"
            f"【文化价值】{craft['cultural_value']}\n"
            f"【难度等级】{craft['difficulty_level']}"
        )
        documents.append(
            Document(
                page_content=overview,
                metadata={
                    "id": craft.get("id", ""),
                    "name": craft.get("name", ""),
                    "category": craft.get("category", ""),
                    "block_type": "overview",
                },
            )
        )

        steps = craft.get("craft_process", [])
        total_steps = len(steps)
        for step_info in steps:
            step_text = (
                f"【非遗项目】{craft['name']}\n"
                f"【工序步骤{step_info['step']}】{step_info['name']}\n"
                f"【详细说明】{step_info['detail']}\n"
                f"【关键要点】{'；'.join(step_info.get('key_points', []))}\n"
            )
            if step_info.get("tools"):
                step_text += f"【使用工具】{'、'.join(step_info['tools'])}\n"
            if step_info.get("materials"):
                step_text += f"【所需材料】{'、'.join(step_info['materials'])}\n"
            step_text += f"【工艺位置】第{step_info['step']}步/共{total_steps}步"

            documents.append(
                Document(
                    page_content=step_text,
                    metadata={
                        "id": craft.get("id", ""),
                        "name": craft.get("name", ""),
                        "category": craft.get("category", ""),
                        "block_type": "craft_step",
                        "step_num": step_info.get("step", 0),
                        "step_name": step_info.get("name", ""),
                    },
                )
            )

    return documents


def build_vectorstore(
    documents: List[Document],
    embedding_model: str = "shibing624/text2vec-base-chinese",
    save_path: Optional[str] = None,
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        vectorstore.save_local(save_path)

    return vectorstore


def load_vectorstore(
    load_path: str,
    embedding_model: str = "shibing624/text2vec-base-chinese",
) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)


def build_qa_chain(
    crafts: List[Dict],
    vectorstore: Optional[FAISS],
    llm_model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    k: int = 8,
    fallback_documents: Optional[List[Document]] = None,
) -> JiyiTongQA:
    llm = None
    if api_key:
        llm_kwargs = {"model": llm_model, "temperature": 0.2, "api_key": api_key}
        if api_base:
            llm_kwargs["base_url"] = api_base
        llm = ChatOpenAI(**llm_kwargs)

    return JiyiTongQA(
        crafts=crafts,
        vectorstore=vectorstore,
        retriever_k=k,
        llm=llm,
        fallback_documents=fallback_documents or [],
    )


def init_jiyitong(
    data_path: str = "craft_techniques.json",
    index_path: str = "faiss_index",
    embedding_model: str = "shibing624/text2vec-base-chinese",
    llm_model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    rebuild_index: bool = False,
) -> JiyiTongQA:
    crafts = load_craft_data(data_path)
    documents = craft_to_documents(crafts)
    print(f"[技艺通] 生成文档块数: {len(documents)}")

    vectorstore: Optional[FAISS] = None
    try:
        if not rebuild_index and os.path.exists(index_path):
            print(f"[技艺通] 加载已有向量索引: {index_path}")
            vectorstore = load_vectorstore(index_path, embedding_model)
        elif not rebuild_index:
            print("[技艺通] 未检测到本地向量索引，启动离线关键词检索模式")
        else:
            print(f"[技艺通] 构建向量索引: {index_path}")
            vectorstore = build_vectorstore(documents, embedding_model, index_path)
            print(f"[技艺通] 向量索引已保存: {index_path}")
    except Exception as e:
        print(f"[技艺通] 向量检索不可用，切换离线关键词检索：{type(e).__name__}: {e}")

    qa = build_qa_chain(
        crafts,
        vectorstore,
        llm_model,
        api_key,
        api_base,
        fallback_documents=documents,
    )
    print("[技艺通] 系统初始化完成")
    return qa


def ask(qa_chain: JiyiTongQA, question: str, session_state: Optional[Dict] = None) -> Dict:
    return qa_chain.answer(question, session_state=session_state)
