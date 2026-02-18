# DataRobot AutoML Operations Agent - Project Design Document

このドキュメントは、DataRobotのAutoML機能やMLOps機能を自然言語で操作するエージェントアプリケーションの設計および実装指針を定義します。開発にあたっては、`copilot-instructions.md` に記載された制約とベストプラクティスを厳守してください。

## 1. プロジェクト概要

ユーザーがチャットインターフェースを通じて自然言語で指示を出し、DataRobotのライフサイクル（データアップロード、モデリング、デプロイ、予測、監視）を自動化・操作できるエージェントを構築します。

### 技術スタック
* **Orchestration**: LangGraph (`langgraph.graph.StateGraph`)
* **Backend**: Python 3.10+, FastAPI
* **AI Integration**: `datarobot_genai` (LLM Gateway利用)
* **Tools**: Model Context Protocol (MCP) Server
* **Frontend**: React, TypeScript (Plain text input)

## 2. エージェントアーキテクチャ設計

### 2.1 デザインパターン: ReAct + MCP Integration
`copilot-instructions.md` の推奨パターンに従い、**ReActパターン**を採用します。DataRobot LLM Gateway と MCPツールを統合し、LLMが自律的にツールを選択・実行する構成とします。

* **Graph Type**: `StateGraph[MessagesState, None, MessagesState, MessagesState](MessagesState)`
    * **重要**: DataRobot環境での互換性を保つため、必ず標準の `MessagesState` を使用します。カスタムStateは使用しません。
* **Agent Construction**: `create_react_agent` を使用して構築します。

### 2.2 データフローとインターフェース

#### 入力 (Frontend -> Agent)
* **制約**: フロントエンドからは**必ずプレーンテキスト**（自然言語）を送信します。
* **DRUM制約**: エージェントの起動（`invoke`）は単一の引数のみを受け取る実装とします。
    * ✅ Good: `invoke({"input": "医療費データの分析を開始して"})`
    * ❌ Bad: `invoke(target="cost", dataset="medical.csv")`

#### 出力 (Agent -> Frontend)
* **基本**: エージェントの思考プロセスとツール実行結果を含む自然言語。
* **構造化データ**: リストや表形式のデータが必要な場合は、プロンプトでJSON形式での出力を指示し、フロントエンド側でパースして表示します。

## 3. 実装詳細

### 3.1 エージェント定義 (`agent/agentic_workflow/agent.py`)

```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from datarobot_genai.langgraph.agent import LangGraphAgent
from datarobot_genai.core.agents import make_system_prompt

class AutoMLAgent(LangGraphAgent):
    @property
    def workflow(self) -> StateGraph[MessagesState]:
        # MessagesStateを使用し、シンプルなReActフローを構築
        workflow = StateGraph[MessagesState, None, MessagesState, MessagesState](MessagesState)
        workflow.add_node("agent", self.agent)
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        return workflow

    @property
    def agent(self):
        # LLM Gateway と MCPツールを自動統合
        # copilot-instructions.md に従い self.llm() を使用
        return create_react_agent(
            self.llm(preferred_model="datarobot/azure/gpt-4o"), 
            tools=self.mcp_tools, # 自動的にロードされたMCPツール
            prompt=make_system_prompt(self.system_prompt_content),
        )
## 7. UI/UX および出力処理のガイドライン (Frontend)

### 7.1 エージェント応答の処理
* **JSON出力の推奨**: リストや表を表示する場合、エージェントにはJSON形式での回答を指示し、フロントエンドでパースして表示します。
* **パースロジック**: ` ```json ` ブロックを正規表現で抽出するフォールバック処理を実装します。

### 7.2 デザインシステム (Strict)
* **カラー**: キーカラー `#81FBA5` を使用し、ダークテーマ (`bg-gray-900`) を基本とします。
* **アクセシビリティ**: WCAG AA基準（コントラスト比 4.5:1）を満たす配色を徹底します。