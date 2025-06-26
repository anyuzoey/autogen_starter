import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import SseServerParams, SseMcpToolAdapter
from autogen_core.tools import FunctionTool
from tavily import AsyncTavilyClient

# Load environment variables from .env file
load_dotenv()

GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GITHUB_MCP_URL:
    raise ValueError("GITHUB_MCP_URL environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")

async def main():
    server_params = SseServerParams(
        url=GITHUB_MCP_URL,
        headers=None,
        timeout=10,
        sse_read_timeout=300,
    )
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-nano-2025-04-14",
        api_key=OPENAI_API_KEY
    )

    async def tavily_search_func(query: str, max_results: int = 5) -> dict:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        result = await client.search(query=query, max_results=max_results, include_answer=True)
        return result

    tavily_tool = FunctionTool(
        func=tavily_search_func,
        name="tavily_search",
        description="Perform a web search using Tavily and return summarized results."
    )

    async def get_tool_adapter(tool_name: str):
        return await SseMcpToolAdapter.from_server_params(server_params, tool_name)

    tool_adapter_add_issue_comment = await get_tool_adapter("add_issue_comment")
    tool_adapter_get_issue = await get_tool_adapter("get_issue")

    issue_reader = AssistantAgent(
        name="issue_reader", model_client=model_client, tools=[tool_adapter_get_issue], reflect_on_tool_use=True,
        description="Extracts structured information from a GitHub issue.",
        system_message="You are a GitHub Issue Reader. Extract key problem details, error messages, user environment, and summarize the issue."
    )

    researcher = AssistantAgent(
        name="researcher", model_client=model_client, tools=[tavily_tool], reflect_on_tool_use=True,
        description="Researches related info to assist with resolving the issue.",
        system_message="You are a web researcher. Based on the issue summary, find related GitHub issues, documentation, and known solutions."
    )

    reasoner = AssistantAgent(
        name="reasoner", model_client=model_client, description="Analyzes and generates an action plan.",
        system_message="You are a technical expert. Given a GitHub issue and related research, suggest potential root causes and actionable next steps."
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func=input,
        description="A human-in-the-loop agent that must approve the plan before posting a comment. Type APPROVE to continue."
    )

    commenter = AssistantAgent(
        name="commenter", model_client=model_client, tools=[tool_adapter_add_issue_comment], reflect_on_tool_use=True,
        description="Writes a GitHub comment.",
        system_message="Turn the response from the researcher agent and reasoner agent output into a detailed GitHub comment. Do not end the comment with a question or an incomplete thought. Only act after user_proxy says APPROVE."
    )

    # Termination: end when user says APPROVE (so commenter can act), or after 6 turns
    termination = TextMentionTermination("APPROVE")

    # Prompt for GitHub issue URL
    issue_url = input("Enter the GitHub issue URL: ")
    task = f"Summarize and add next steps for this issue: {issue_url}"

    team = RoundRobinGroupChat([
        issue_reader,
        researcher,
        reasoner,
        user_proxy,
        commenter
    ], termination_condition=termination, max_turns=6)

    stream = team.run_stream(task=task)
    await Console(stream)
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
