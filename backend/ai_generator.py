import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses.

    Falls back to MockAIGenerator automatically when the API key is invalid or absent.
    """

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self._mock = MockAIGenerator()
        self._use_mock = not api_key or api_key.startswith("sk-ant-dummy")
        if not self._use_mock:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.model = model
                self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}
            except Exception:
                self._use_mock = True

    def generate_response(self, query: str,
                          conversation_history: Optional[str] = None,
                          tools: Optional[List] = None,
                          tool_manager=None) -> str:
        if self._use_mock:
            return self._mock.generate_response(query, tool_manager)

        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        try:
            response = self.client.messages.create(**api_params)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.APIStatusError):
            # Key is present but invalid/unauthorized — fall back to mock for the rest of the session
            self._use_mock = True
            return self._mock.generate_response(query, tool_manager)

        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        return response.content[0].text

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for block in initial_response.content:
            if block.type == "tool_use":
                result = tool_manager.execute_tool(block.name, **block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_params = {**self.base_params, "messages": messages, "system": base_params["system"]}
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text


class MockAIGenerator:
    """Offline fallback: searches the vector store and formats the results directly.

    Used when no valid Anthropic API key is available. The full RAG pipeline
    (document loading, chunking, sentence-transformer embeddings, ChromaDB
    semantic search) still runs — only the final LLM generation step is replaced
    by a simple formatter so the app works without any API subscription.
    """

    def generate_response(self, query: str, tool_manager=None) -> str:
        if tool_manager is None:
            return "No search tools available. Please check the server configuration."

        raw = tool_manager.execute_tool("search_course_content", query=query)

        if not raw or raw.startswith("No relevant content"):
            return (
                f"No course content found matching **\"{query}\"**.\n\n"
                "Try rephrasing your question or ask about one of the available courses."
            )

        # Parse the block-separated results returned by CourseSearchTool
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        if not blocks:
            return raw

        lines = ["Here is what the course materials say:\n"]
        for block in blocks:
            block_lines = block.splitlines()
            header = block_lines[0] if block_lines else ""
            body = " ".join(block_lines[1:]).strip() if len(block_lines) > 1 else ""
            # Trim body to ~300 chars for readability
            if len(body) > 300:
                body = body[:297] + "..."
            lines.append(f"**{header}**\n{body}")

        lines.append(
            "\n---\n*Running in offline/demo mode — add a valid `ANTHROPIC_API_KEY` "
            "to `.env` for full AI-generated answers.*"
        )
        return "\n\n".join(lines)