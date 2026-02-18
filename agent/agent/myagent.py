# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import os
import re
from io import BytesIO
from typing import Any, Optional, Union

from datarobot_genai.core.agents import (
    make_system_prompt,
)
from datarobot_genai.langgraph.agent import LangGraphAgent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from PIL import Image

from agent.config import Config
from agent.prompts import ANALYZER_1_PROMPT, ANALYZER_2_PROMPT

# Module-level reference to the agent instance for tools to access
_current_agent: Optional["MyAgent"] = None


def _load_image_base64(user_text: str, config: Config) -> tuple[str, str]:
    """Load image from file path, return (base64_str, clean_text).

    Parses [IMAGE:path] token from user text. Falls back to sample image.
    Resizes large images to max 1920px width.
    """
    image_path = None
    clean_text = user_text

    # Parse [IMAGE:path] token
    match = re.search(r"\[IMAGE:(.*?)\]", user_text)
    if match:
        image_path = match.group(1).strip()
        clean_text = re.sub(r"\[IMAGE:.*?\]\s*", "", user_text).strip()

    # Determine image file path
    if image_path and os.path.exists(image_path):
        file_path = image_path
    else:
        file_path = config.sample_image_path
        if not os.path.isabs(file_path):
            agent_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            file_path = os.path.join(agent_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Load and resize image
    img = Image.open(file_path)
    max_width = 1920
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to base64
    buffer = BytesIO()
    img_format = "JPEG"
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffer, format=img_format, quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    if not clean_text:
        clean_text = "この店舗棚画像を分析してください。"

    return image_base64, clean_text


def _call_vision_llm(
    model_name: str, system_prompt: str, user_text: str,
    agent: "MyAgent",
) -> str:
    """Call a vision LLM with an image and return the text response."""
    image_base64, clean_text = _load_image_base64(user_text, agent.config)

    vision_llm = agent._create_vision_llm(model_name)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": clean_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ]
        ),
    ]

    response = vision_llm.invoke(messages)
    return str(response.content)


@tool
def analyze_with_gpt(user_message: str) -> str:
    """Analyzer A (GPT): 店舗棚画像をGPTで分析します。ゴールデンゾーン分析、フェイス数・シェルフシェア分析、価格戦略分析、競合商品特定を行います。user_messageにはユーザーの元のメッセージ（[IMAGE:path]トークン含む）をそのまま渡してください。"""
    global _current_agent
    if _current_agent is None:
        return "エラー: エージェントが初期化されていません"
    return _call_vision_llm(
        _current_agent.config.llm_analyzer_1_model,
        ANALYZER_1_PROMPT,
        user_message,
        _current_agent,
    )


@tool
def analyze_with_gemini(user_message: str) -> str:
    """Analyzer B (Gemini): 店舗棚画像をGeminiで分析します。デジタルサイネージ・販促物分析、新商品・季節商品分析、陳列品質分析、カテゴリー配置分析を行います。user_messageにはユーザーの元のメッセージ（[IMAGE:path]トークン含む）をそのまま渡してください。"""
    global _current_agent
    if _current_agent is None:
        return "エラー: エージェントが初期化されていません"
    return _call_vision_llm(
        _current_agent.config.llm_analyzer_2_model,
        ANALYZER_2_PROMPT,
        user_message,
        _current_agent,
    )


AGENT_SYSTEM_PROMPT = """あなたはカルビー株式会社の店舗棚分析AIエージェントです。

ユーザーから店舗棚画像の分析リクエストを受け取ったら、以下の手順で分析を実行してください。

## 分析手順:
1. まず、analyze_with_gpt ツールと analyze_with_gemini ツールの**両方**を呼び出してください
   - 両方のツールに、ユーザーのメッセージをそのまま渡してください（[IMAGE:...]トークンを含む）
2. 両方のツールの結果が返ってきたら、統合レポートを作成してください

## 統合レポートの形式:

### 総合評価
- 店舗棚全体のカルビー商品の展開状況を5段階（★〜★★★★★）で評価
- 評価理由を1〜2文で簡潔に記載

### 棚割り・配置分析
- Analyzer Aの結果を基に、ゴールデンゾーン分析とシェルフシェアをまとめる

### 価格・販促分析
- 価格戦略の妥当性と競合との比較
- 販促物の効果

### デジタル・販促ツール状況
- Analyzer Bの結果を基に、デジタルサイネージと販促物の状況をまとめる

### 新商品・季節商品展開
- 新商品の視認性と展開状況の評価

### 改善提案（アクションアイテム）
- 具体的な改善提案を優先度順（高・中・低）にリスト化

### 競合動向
- 競合他社の棚戦略に関する重要な観察

## 出力ルール:
- 日本語で回答してください
- 2つのAnalyzerの結果を必ず両方参照してください
- 具体的で実行可能な提案を心がけてください
- 必ず analyze_with_gpt と analyze_with_gemini の両方のツールを呼び出してから統合レポートを作成してください
"""


class MyAgent(LangGraphAgent):
    """MyAgent is a custom agent that analyzes store shelf images for Calbee.
    It uses two different LLMs via tools to extract insights from shelf images,
    then the agent summarizes the combined analysis into a comprehensive report.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        verbose: Optional[Union[bool, str]] = True,
        timeout: Optional[int] = 90,
        **kwargs: Any,
    ):
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            model=model,
            verbose=verbose,
            timeout=timeout,
            **kwargs,
        )
        self.config = Config()
        self.default_model = self.config.llm_default_model
        if model in ("unknown", "datarobot-deployed-llm"):
            self.model = self.default_model

        # Set module-level reference so tools can access this agent
        global _current_agent
        _current_agent = self

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        langgraph_workflow = StateGraph[
            MessagesState, None, MessagesState, MessagesState
        ](MessagesState)

        langgraph_workflow.add_node("agent", self.agent)
        langgraph_workflow.add_edge(START, "agent")
        langgraph_workflow.add_edge("agent", END)

        return langgraph_workflow  # type: ignore[return-value]

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "user",
                    "{user_prompt_content}",
                ),
            ]
        )

    def llm(
        self,
        auto_model_override: bool = True,
    ) -> ChatLiteLLM:
        """Returns the ChatLiteLLM to use for a given model.

        If a `self.model` is provided, it will be used. Otherwise, the default model will be used.
        If auto_model_override is True, it will try and use the model specified in the request
        but automatically back out to the default model if the LLM Gateway is not configured

        Args:
            auto_model_override: Optional[bool]: If True, it will try and use the model
                specified in the request but automatically back out if the LLM Gateway is
                not available.

        Returns:
            ChatLiteLLM: The model to use.
        """
        api_base = self.litellm_api_base(self.config.llm_deployment_id)
        model = self.model or self.default_model
        if auto_model_override and not self.config.use_datarobot_llm_gateway:
            model = self.default_model
        if self.verbose:
            print(f"Using model: {model}")

        config = {
            "model": model,
            "api_base": api_base,
            "api_key": self.api_key,
            "timeout": self.timeout,
            "streaming": True,
            "max_retries": 3,
        }

        if not self.config.use_datarobot_llm_gateway and self._identity_header:
            config["default_headers"] = self._identity_header  # type: ignore[assignment]

        return ChatLiteLLM(**config)

    def _create_vision_llm(self, model_name: str) -> ChatLiteLLM:
        """Create a ChatLiteLLM instance for a specific vision model.

        Uses the same API base and auth pattern as self.llm() but with
        a specified model name and extended timeout for image processing.
        """
        api_base = self.litellm_api_base(self.config.llm_deployment_id)
        if self.verbose:
            print(f"Using vision model: {model_name}")

        config = {
            "model": model_name,
            "api_base": api_base,
            "api_key": self.api_key,
            "timeout": 180,
            "streaming": True,
            "max_retries": 3,
        }

        if not self.config.use_datarobot_llm_gateway and self._identity_header:
            config["default_headers"] = self._identity_header  # type: ignore[assignment]

        return ChatLiteLLM(**config)

    @property
    def agent(self) -> Any:
        """Create the ReAct agent with image analysis tools."""
        return create_react_agent(
            self.llm(),
            tools=[analyze_with_gpt, analyze_with_gemini],
            prompt=make_system_prompt(AGENT_SYSTEM_PROMPT),
        )
