import json
import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

import aiofiles
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StreamableHttpServerParams, StreamableHttpMcpToolAdapter
from autogen_core.tools import FunctionTool
from tavily import AsyncTavilyClient
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

def convert_datetime_to_string(obj):
    """Recursively convert datetime objects to ISO format strings"""
    if isinstance(obj, dict):
        return {key: convert_datetime_to_string(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
state_path = "hitl_team_state.json"
history_path = "hitl_team_history.json"

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    """Serve the chat interface HTML file."""
    return FileResponse("hitl_gitcommiter_web.html")

async def get_github_team(
    user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
) -> RoundRobinGroupChat:
    """Create the GitHub issue commenter team using MCP tools"""
    
    # Load environment variables
    from dotenv import load_dotenv
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
    
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-nano-2025-04-14",
        api_key=OPENAI_API_KEY
    )
    
    # Setup MCP server parameters
    server_params = StreamableHttpServerParams(
        url=GITHUB_MCP_URL,
        headers={
            "Authorization": f"Bearer {GITHUB_PAT}",
            "Content-Type": "application/json"
        },
        timeout=timedelta(seconds=30),
        sse_read_timeout=timedelta(seconds=60 * 5),
        terminate_on_close=True,
    )

    # Create tools
    async def tavily_search_func(query: str, max_results: int = 5) -> dict:
        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        result = await client.search(query=query, max_results=max_results, include_answer=True)
        return result

    tavily_tool = FunctionTool(
        func=tavily_search_func,
        name="tavily_search",
        description="Perform a web search using Tavily and return summarized results."
    )

    # Create MCP tool adapters
    async def get_tool_adapter(tool_name: str):
        adapter = await StreamableHttpMcpToolAdapter.from_server_params(server_params, tool_name)
        return adapter

    tool_adapter_add_issue_comment = await get_tool_adapter("add_issue_comment")
    tool_adapter_get_issue = await get_tool_adapter("get_issue")

    # Create agents
    issue_reader = AssistantAgent(
        name="issue_reader", 
        model_client=model_client, 
        tools=[tool_adapter_get_issue], 
        reflect_on_tool_use=True,
        description="Extracts structured information from a GitHub issue using tool_adapter_get_issue tools.",
        system_message="You are a GitHub Issue Reader. Extract key problem details, error messages, user environment, and summarize the issue using the tool_adapter_get_issue tool."
    )

    researcher = AssistantAgent(
        name="researcher", 
        model_client=model_client, 
        tools=[tavily_tool], 
        reflect_on_tool_use=True,
        description="Researches related info to assist with resolving the issue using tavily_tool.",
        system_message="You are a web researcher. Based on the issue summary, find top 3 related GitHub issues, documentation, and known solutions using the tavily_tool."
    )

    reasoner = AssistantAgent(
        name="reasoner", 
        model_client=model_client, 
        description="Analyzes and generates an action plan.",
        system_message="You are a technical expert. Given a GitHub issue and related research, suggest potential root causes and actionable next steps and format it as a github comment draft. Keep it concise."
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func=user_input_func,
        description="A human-in-the-loop agent that can edit the draft comment and must approve it before posting. Type APPROVE to post as-is, or provide an edited version of the comment."
    )

    commenter = AssistantAgent(
        name="commenter", 
        model_client=model_client, 
        tools=[tool_adapter_add_issue_comment], 
        reflect_on_tool_use=True,
        description="Writes a GitHub comment using tool_adapter_add_issue_comment tool.",
        system_message="You are a github commenter. When user_proxy provides input, if it's 'APPROVE', use the original draft from reasoner (remove the 'DRAFT:' prefix, Add 'bot' prefix). If user_proxy provides edited text, use that as the comment. Use the tool_adapter_add_issue_comment tool to post the final comment."
    )

    # Create team
    team = RoundRobinGroupChat([
        issue_reader,
        researcher,
        reasoner,
        user_proxy,
        commenter
    ], termination_condition=TextMentionTermination("APPROVE"), max_turns=6)
    
    # Load state from file if it exists
    if os.path.exists(state_path):
        async with aiofiles.open(state_path, "r") as file:
            state = json.loads(await file.read())
        await team.load_state(state)
    
    return team

async def get_history() -> list[dict[str, Any]]:
    """Get chat history from file."""
    if not os.path.exists(history_path):
        return []
    async with aiofiles.open(history_path, "r") as file:
        return json.loads(await file.read())

@app.get("/history")
async def history() -> list[dict[str, Any]]:
    try:
        return await get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    # User input function used by the team
    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
        data = await websocket.receive_json()
        message = TextMessage.model_validate(data)
        return message.content

    try:
        while True:
            # Get user message
            data = await websocket.receive_json()
            request = TextMessage.model_validate(data)

            # Get the team and respond to the message
            team = await get_github_team(_user_input)
            history = await get_history()
            
            # Create task from the message
            task = f"Summarize and add next steps for this issue: {request.content}"
            
            stream = team.run_stream(task=task)
            async for message in stream:
                if isinstance(message, TaskResult):
                    continue
                # Convert datetime objects to strings before sending
                message_data = convert_datetime_to_string(message.model_dump())
                await websocket.send_json(message_data)
                if not isinstance(message, UserInputRequestedEvent):
                    # Don't save user input events to history
                    history.append(message_data)

            # Save team state and chat history to file
            async with aiofiles.open(state_path, "w") as file:
                state = await team.save_state()
                await file.write(json.dumps(state, default=str))

            async with aiofiles.open(history_path, "w") as file:
                await file.write(json.dumps(history, default=str))

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Error: {str(e)}",
                "source": "system"
            })
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003) 