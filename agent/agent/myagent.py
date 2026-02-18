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
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_litellm.chat_models import ChatLiteLLM
from langgraph.graph import END, START, MessagesState, StateGraph
from PIL import Image

from agent.config import Config
from agent.prompts import ANALYZER_1_PROMPT, ANALYZER_2_PROMPT, SUMMARIZER_PROMPT


class MyAgent(LangGraphAgent):
    """MyAgent is a custom agent that analyzes store shelf images for Calbee.
    It uses two different LLMs in parallel to extract insights from shelf images,
    then summarizes the combined analysis into a comprehensive report.
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

    @property
    def workflow(self) -> StateGraph[MessagesState]:
        langgraph_workflow = StateGraph[
            MessagesState, None, MessagesState, MessagesState
        ](MessagesState)

        langgraph_workflow.add_node(
            "image_input_node", self.image_input_node
        )
        langgraph_workflow.add_node(
            "analyzer_1_node", self.agent_analyzer_1
        )
        langgraph_workflow.add_node(
            "analyzer_2_node", self.agent_analyzer_2
        )
        langgraph_workflow.add_node(
            "summarizer_node", self.agent_summarizer
        )

        langgraph_workflow.add_edge(START, "image_input_node")
        # Fan-out: parallel execution of both analyzers
        langgraph_workflow.add_edge("image_input_node", "analyzer_1_node")
        langgraph_workflow.add_edge("image_input_node", "analyzer_2_node")
        # Fan-in: both feed into summarizer
        langgraph_workflow.add_edge("analyzer_1_node", "summarizer_node")
        langgraph_workflow.add_edge("analyzer_2_node", "summarizer_node")
        langgraph_workflow.add_edge("summarizer_node", END)

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

    def _load_image(self, user_text: str) -> tuple[str, str]:
        """Load image from file path or sample, return (base64_str, user_text).

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
            file_path = self.config.sample_image_path
            if not os.path.isabs(file_path):
                # Try relative to agent directory
                agent_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                file_path = os.path.join(agent_dir, file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Image file not found: {file_path}"
            )

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

    def image_input_node(
        self, state: MessagesState
    ) -> dict[str, list[HumanMessage]]:
        """Load image and prepare multimodal message for analyzers."""
        last_message = state["messages"][-1]
        user_text = (
            last_message.content
            if isinstance(last_message.content, str)
            else str(last_message.content)
        )

        image_base64, clean_text = self._load_image(user_text)

        multimodal_content: list[dict[str, Any]] = [
            {"type": "text", "text": clean_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ]

        return {
            "messages": [HumanMessage(content=multimodal_content)]
        }

    @property
    def agent_analyzer_1(self) -> Any:
        return create_agent(
            self._create_vision_llm(self.config.llm_analyzer_1_model),
            tools=[],
            system_prompt=make_system_prompt(ANALYZER_1_PROMPT),
            name="Analyzer A (GPT)",
        )

    @property
    def agent_analyzer_2(self) -> Any:
        return create_agent(
            self._create_vision_llm(self.config.llm_analyzer_2_model),
            tools=[],
            system_prompt=make_system_prompt(ANALYZER_2_PROMPT),
            name="Analyzer B (Gemini)",
        )

    @property
    def agent_summarizer(self) -> Any:
        return create_agent(
            self.llm(),
            tools=[],
            system_prompt=make_system_prompt(SUMMARIZER_PROMPT),
            name="Summarizer",
        )
