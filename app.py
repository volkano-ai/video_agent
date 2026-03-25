"""FastAPI service exposing Researcher, ScriptWriter, and Storyboard agents as separate APIs."""

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from video_agent.agents.researcher import researcher_agent
from video_agent.agents.script_writer import script_writer_agent
from video_agent.agents.storyboard_artist import storyboard_agent


# --- Session service (shared across all agents) ---
session_service = InMemorySessionService()

# --- Runners for each agent ---
APP_NAME = "video_agent"

researcher_runner = Runner(
    agent=researcher_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

script_writer_runner = Runner(
    agent=script_writer_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

storyboard_runner = Runner(
    agent=storyboard_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# --- Request / Response models ---

class ResearchRequest(BaseModel):
    concept: str = Field(..., description="The video concept or topic to research")
    source_links: list[str] = Field(default_factory=list, description="Optional source URLs to analyze")
    target_platform: str = Field(default="YouTube Shorts", description="Target social media platform")


class ResearchResponse(BaseModel):
    research_brief: str


class ScriptWriteRequest(BaseModel):
    concept: str = Field(..., description="What the ad is about")
    research_brief: str = Field(default="", description="Research brief from the researcher agent (optional)")
    target_duration: int = Field(default=30, description="Target video duration in seconds")


class ScriptWriteResponse(BaseModel):
    draft_script: str


class StoryboardRequest(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID for GCS folder organization")
    draft_script: str = Field(..., description="The video script to generate storyboard images for")


class StoryboardResponse(BaseModel):
    storyboard: dict


class HealthResponse(BaseModel):
    status: str = "ok"


# --- Helper to run an agent and collect output ---

async def run_agent(runner: Runner, user_message: str, initial_state: dict | None = None):
    """Run an agent with a message and optional initial state, return the final text output and session state."""
    user_id = f"user_{uuid.uuid4().hex[:8]}"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    # Create session with initial state
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state or {},
    )

    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_text = ""
    final_state = {}

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text += part.text

    # Get final session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if session:
        final_state = dict(session.state)

    return final_text, final_state


# --- FastAPI app ---

app = FastAPI(
    title="Video Agent API",
    description="Multi-agent video production pipeline with separate endpoints for research, scriptwriting, and storyboarding.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()


@app.post("/api/v1/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    """Research a video topic using web search and source analysis.

    Returns a structured research brief with facts, data points, perspectives,
    and visual asset suggestions.
    """
    message_parts = [f"Research this video concept: {request.concept}"]
    if request.source_links:
        message_parts.append(f"Source links to analyze: {', '.join(request.source_links)}")
    message_parts.append(f"Target platform: {request.target_platform}")

    user_message = "\n".join(message_parts)

    try:
        text, state = await run_agent(
            runner=researcher_runner,
            user_message=user_message,
        )
        research_brief = state.get("research_brief", text)
        return ResearchResponse(research_brief=research_brief)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/script", response_model=ScriptWriteResponse)
async def write_script(request: ScriptWriteRequest):
    """Write a Meta ad video script from a concept and optional research brief.

    Returns a scene-by-scene script with narration, on-screen text, visuals,
    and timing optimized for Facebook/Instagram Reels and Stories.
    """
    message_parts = [f"Write a {request.target_duration} second Meta ad script for: {request.concept}"]

    initial_state = {}
    if request.research_brief:
        initial_state["research_brief"] = request.research_brief

    user_message = "\n".join(message_parts)

    try:
        text, state = await run_agent(
            runner=script_writer_runner,
            user_message=user_message,
            initial_state=initial_state,
        )
        draft_script = state.get("draft_script", text)
        return ScriptWriteResponse(draft_script=draft_script)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/storyboard", response_model=StoryboardResponse)
async def generate_storyboard(request: StoryboardRequest):
    """Generate a visual storyboard from a script.

    Generates one image per scene using Gemini 3.1 Flash, uploads to GCS
    bucket 'video_generation_agent' under {workspace_id}/{timestamp}/,
    and returns structured JSON mapping each script line to its image URL.
    """
    initial_state = {
        "workspace_id": request.workspace_id,
        "draft_script": request.draft_script,
    }

    try:
        text, state = await run_agent(
            runner=storyboard_runner,
            user_message=f"Generate storyboard for workspace: {request.workspace_id}",
            initial_state=initial_state,
        )
        storyboard = state.get("storyboard", {})
        if not storyboard:
            # Try parsing the text output as JSON
            import json
            try:
                storyboard = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                storyboard = {"raw_output": text}

        return StoryboardResponse(storyboard=storyboard)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
