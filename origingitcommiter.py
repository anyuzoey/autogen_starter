import asyncio
import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# Autogen imports
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import SseServerParams, SseMcpToolAdapter
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from tavily import AsyncTavilyClient
from dotenv import load_dotenv
from autogen_agentchat.ui import Console

# Load environment variables from .env file
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GitHub Issue Agent Team API",
    description="An API to trigger an agent team to analyze a GitHub issue and suggest next steps.",
)

# --- CORS Middleware Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for API ---
class IssueRequest(BaseModel):
    """Request model for the issue link."""
    issue_link: str

class IssueResponse(BaseModel):
    """Response model for the generated comment."""
    response: str | Dict[str, Any]

# --- Agent and Tool Configuration (Global Setup) ---
# Load configurations from environment variables
GITHUB_MCP_URL = os.getenv("GITHUB_MCP_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GITHUB_MCP_URL:
    raise ValueError("GITHUB_MCP_URL environment variable not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")

# --- Helper Functions ---
async def setup_agents_and_tools():
    """Setup common agents and tools for both endpoints."""
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

    # --- Tool Definitions ---
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
        """Helper function to get the tool adapter by name."""
        return await SseMcpToolAdapter.from_server_params(server_params, tool_name)

    tool_adapter_add_issue_comment = await get_tool_adapter("add_issue_comment")
    tool_adapter_get_issue = await get_tool_adapter("get_issue")

    # --- Agent Definitions ---
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

    commenter = AssistantAgent(
        name="commenter", model_client=model_client, tools=[tool_adapter_add_issue_comment], reflect_on_tool_use=True,
        description="Writes a GitHub comment.",
        system_message="Turn the response from the researcher agent and reasoner agent output into a detailed GitHub comment. Do not end the comment with a question or an incomplete thought."
    )

    return {
        "model_client": model_client,
        "issue_reader": issue_reader,
        "researcher": researcher,
        "reasoner": reasoner,
        "commenter": commenter
    }

@app.post("/github_webhook")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    if payload.get("action") == "opened" and "issue" in payload:
        issue_url = payload["issue"]["html_url"]
        # Use background task so webhook returns quickly
        background_tasks.add_task(analyze_issue_with_comment, IssueRequest(issue_link=issue_url))
        return {"status": "accepted", "detail": f"Processing issue {issue_url}"}
    return {"status": "ignored", "reason": "Not an 'opened' issue event"}

# --- API Endpoints ---
@app.post("/issue_next_steps_with_comment", response_model=IssueResponse)
async def analyze_issue_with_comment(request: IssueRequest):
    """
    Accepts a GitHub issue URL, processes it with an agent team including commenter,
    and posts the suggested next steps as a comment to the GitHub issue.
    """
    print(f"Received request for issue with comment: {request.issue_link}")

    try:
        agents_and_tools = await setup_agents_and_tools()
        
        # Create team with commenter agent
        team = RoundRobinGroupChat([
            agents_and_tools["issue_reader"], 
            agents_and_tools["researcher"], 
            agents_and_tools["reasoner"], 
            agents_and_tools["commenter"]
        ], max_turns=4)
        
        # --- Running the Agent Team ---
        task = f"Summarize and add next steps for this issue: {request.issue_link}"
        stream = team.run_stream(task=task)
        
        # Capture the final output from the stream
        final_output = None
        async for chunk in stream:
            # Each chunk contains information about the conversation
            if hasattr(chunk, 'content') and chunk.content:
                final_output = chunk.content
            elif hasattr(chunk, 'message') and chunk.message:
                final_output = chunk.message.get('content', '')
            elif isinstance(chunk, dict) and 'content' in chunk:
                final_output = chunk['content']
            elif isinstance(chunk, str):
                final_output = chunk
        
        await agents_and_tools["model_client"].close()
        
        # Return the actual final output if captured, otherwise a success message
        if final_output:
            return {"response": final_output}
        else:
            return {"response": "Comment generated and posted successfully. Check the issue for details."}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/issue_next_steps_analysis", response_model=IssueResponse)
async def analyze_issue_without_comment(request: IssueRequest):
    """
    Accepts a GitHub issue URL, processes it with an agent team (excluding commenter),
    and returns the analysis and suggested next steps without posting to GitHub.
    """
    print(f"Received request for issue analysis only: {request.issue_link}")

    try:
        agents_and_tools = await setup_agents_and_tools()
        
        # Create team without commenter agent
        team = RoundRobinGroupChat([
            agents_and_tools["issue_reader"], 
            agents_and_tools["researcher"], 
            agents_and_tools["reasoner"]
        ], max_turns=3)
        
        # --- Running the Agent Team ---
        task = f"Analyze this issue and provide detailed next steps: {request.issue_link}"
        stream = team.run_stream(task=task)
        
        # Capture the final output from the stream
        final_output = None
        async for chunk in stream:
            # Each chunk contains information about the conversation
            if hasattr(chunk, 'content') and chunk.content:
                final_output = chunk.content
            elif hasattr(chunk, 'message') and chunk.message:
                final_output = chunk.message.get('content', '')
            elif isinstance(chunk, dict) and 'content' in chunk:
                final_output = chunk['content']
            elif isinstance(chunk, str):
                final_output = chunk
        
        await agents_and_tools["model_client"].close()
        
        # Return the actual final output if captured, otherwise a success message
        if final_output:
            return {"response": final_output}
        else:
            return {"response": "Analysis completed successfully. Check the console output for detailed results."}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Legacy endpoint for backward compatibility ---
@app.post("/issue_next_steps", response_model=IssueResponse)
async def analyze_issue(request: IssueRequest):
    """
    Legacy endpoint that redirects to the with-comment version for backward compatibility.
    """
    return await analyze_issue_with_comment(request)