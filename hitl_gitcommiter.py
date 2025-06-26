import os
import asyncio
from datetime import timedelta
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StreamableHttpServerParams, StreamableHttpMcpToolAdapter
from autogen_core.tools import FunctionTool
from tavily import AsyncTavilyClient

# Load environment variables from .env file
load_dotenv()

GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL")
GITHUB_PAT = os.getenv("GITHUB_PAT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GITHUB_MCP_URL:
    raise ValueError("GITHUB_MCP_URL environment variable not set.")
if not GITHUB_PAT:
    raise ValueError("GITHUB_PAT environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")

async def main():
    auth_headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Content-Type": "application/json"
    }
    
    server_params = StreamableHttpServerParams(
        url=GITHUB_MCP_URL,
        headers=auth_headers,
        timeout=timedelta(seconds=30),
        sse_read_timeout=timedelta(seconds=60 * 5),
        terminate_on_close=True,
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
        try:
            adapter = await asyncio.wait_for(
                StreamableHttpMcpToolAdapter.from_server_params(server_params, tool_name),
                timeout=30.0  # 30 second timeout
            )
            print(f"✅ Tool adapter created for: {tool_name}")
            return adapter
        except asyncio.TimeoutError:
            print(f"⏰ Timeout creating tool adapter for {tool_name} after 30 seconds")
            raise
        except Exception as e:
            print(f"Failed to create tool adapter for {tool_name}: {e}")
            raise

    tool_adapter_add_issue_comment = await get_tool_adapter("add_issue_comment")
    tool_adapter_get_issue = await get_tool_adapter("get_issue")

    issue_reader = AssistantAgent(
        name="issue_reader", model_client=model_client, tools=[tool_adapter_get_issue], reflect_on_tool_use=True,
        description="Extracts structured information from a GitHub issue.",
        system_message="You are a GitHub Issue Reader. Extract key problem details, error messages, user environment, and summarize the issue."
    )
    print("✅ Issue reader agent created")

    researcher = AssistantAgent(
        name="researcher", model_client=model_client, tools=[tavily_tool], reflect_on_tool_use=True,
        description="Researches related info to assist with resolving the issue.",
        system_message="You are a web researcher. Based on the issue summary, find related GitHub issues, documentation, and known solutions."
    )
    print("✅ Researcher agent created")

    reasoner = AssistantAgent(
        name="reasoner", model_client=model_client, description="Analyzes and generates an action plan.",
        system_message="You are a technical expert. Given a GitHub issue and related research, suggest potential root causes and actionable next steps."
    )
    print("✅ Reasoner agent created")

    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func=input,
        description="A human-in-the-loop agent that must approve the plan before posting a comment. Type APPROVE to continue. Ask the user to approve the plan."
    )
    print("✅ User proxy agent created")

    commenter = AssistantAgent(
        name="commenter", model_client=model_client, tools=[tool_adapter_add_issue_comment], reflect_on_tool_use=True,
        description="Writes a GitHub comment.",
        system_message="Turn the response from the researcher agent and reasoner agent output into a detailed GitHub comment. Do not end the comment with a question or an incomplete thought. Only act after user_proxy says APPROVE."
    )
    print("✅ Commenter agent created")

    termination = TextMentionTermination("APPROVE")

    # Prompt for GitHub issue URL
    print("📝 Prompting for GitHub issue URL...")
    issue_url = input("Enter the GitHub issue URL: ")
    task = f"Summarize and add next steps for this issue: {issue_url}"

    team = RoundRobinGroupChat([
        issue_reader,
        researcher,
        reasoner,
        user_proxy,
        commenter
    ], termination_condition=termination, max_turns=6)

    try:
        stream = team.run_stream(task=task)
        await Console(stream)
    except Exception as e:
        print(f"Error during team execution: {e}")
        raise
    finally:
        await model_client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
