# Video Agent - System Architecture

## Overview

A multi-agent system built on **Google ADK (Agent Development Kit)** that takes a user's concept/idea and autonomously produces a social-media-ready video. Inspired by Agent Opus's pipeline: the user provides a concept, optionally with source links, and the system handles research, scripting, storyboarding, asset collection, motion design, voiceover, and final assembly.

---

## High-Level Pipeline

```
User Input (concept + optional links)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│              ORCHESTRATOR (SequentialAgent)           │
│                  "VideoProductionPipeline"            │
│                                                      │
│  ┌─────────────┐                                     │
│  │ 1. RESEARCH  │  LlmAgent - "ResearcherAgent"      │
│  │    AGENT     │  Searches web, extracts facts,      │
│  │              │  collects source material            │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │ 2. SCRIPT   │  LlmAgent - "ScriptWriterAgent"     │
│  │    WRITER    │  Writes video script from research   │
│  │              │  + user intent                      │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │ 3. HUMAN    │  HumanReviewAgent (custom)           │
│  │    REVIEW    │  User edits/approves the script      │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌──────────────────────────────────────┐            │
│  │ 4. PARALLEL PRODUCTION (ParallelAgent)│            │
│  │                                      │            │
│  │  ┌──────────────┐ ┌──────────────┐   │            │
│  │  │ Storyboard   │ │ Asset        │   │            │
│  │  │ Artist       │ │ Manager      │   │            │
│  │  └──────────────┘ └──────────────┘   │            │
│  │  ┌──────────────┐ ┌──────────────┐   │            │
│  │  │ Hook         │ │ Voice        │   │            │
│  │  │ Designer     │ │ Actor        │   │            │
│  │  └──────────────┘ └──────────────┘   │            │
│  └──────────────┬───────────────────┘    │            │
│                 ▼                                    │
│  ┌─────────────┐                                     │
│  │ 5. MOTION   │  LlmAgent - "MotionDesignerAgent"   │
│  │    DESIGNER  │  Creates animations from assets      │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │ 6. VIDEO    │  LlmAgent - "VideoEditorAgent"       │
│  │    EDITOR    │  Assembles final video               │
│  └──────┬──────┘                                     │
│         ▼                                            │
│  ┌─────────────┐                                     │
│  │ 7. QUALITY  │  LoopAgent - iterative refinement    │
│  │    REVIEW    │  Reviews and requests fixes          │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
        │
        ▼
  Final Video (social-media ready)
```

---

## Agent Definitions (Google ADK)

### 1. Root Orchestrator - `SequentialAgent`

The top-level agent that drives the entire pipeline sequentially.

```python
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent, LlmAgent

video_pipeline = SequentialAgent(
    name="VideoProductionPipeline",
    description="End-to-end video production from concept to final video.",
    sub_agents=[
        researcher_agent,
        script_writer_agent,
        human_review_agent,
        parallel_production_agent,
        motion_designer_agent,
        video_editor_agent,
        quality_review_agent,
    ],
)
```

### 2. Researcher Agent - `LlmAgent`

**Purpose:** Takes the user's concept + optional links and researches across the web to gather facts, statistics, context, and relevant source material.

```python
researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model="gemini-2.5-pro",
    instruction="""You are a video research specialist. Given a user's concept
    and optional source links, research the topic thoroughly:

    1. Extract key facts, statistics, and talking points from provided links
    2. Search the web for additional relevant information
    3. Identify potential visual assets (images, charts, data)
    4. Note any controversies or multiple perspectives
    5. Organize findings into a structured research brief

    User concept: {user_concept}
    Source links: {source_links}
    """,
    tools=[google_search_tool, web_scraper_tool],
    output_key="research_brief",
)
```

**Tools:**
- `google_search` - ADK built-in search tool
- `web_scraper` - Custom tool to extract content from user-provided URLs

**Output → `state["research_brief"]`**: Structured research document with facts, sources, and asset suggestions.

---

### 3. Script Writer Agent - `LlmAgent`

**Purpose:** Transforms the research brief into a video script with scene breakdowns, narration text, and timing cues.

```python
script_writer_agent = LlmAgent(
    name="ScriptWriterAgent",
    model="gemini-2.5-pro",
    instruction="""You are a professional video scriptwriter for social media content.

    Using the research brief below, write a compelling video script:

    Research: {research_brief}
    User concept: {user_concept}
    Target platform: {target_platform}
    Target duration: {target_duration}

    Output format:
    - HOOK: Opening 3-5 seconds to grab attention
    - SCENE 1..N: Each with narration text, suggested visuals, duration
    - CTA: Closing call-to-action

    Keep it engaging, informative, and match the tone the user wants.
    """,
    output_key="draft_script",
)
```

**Output → `state["draft_script"]`**: Complete scene-by-scene script.

---

### 4. Human Review Agent - Custom `BaseAgent`

**Purpose:** Pauses the pipeline and presents the script to the user for editing/approval. This is the human-in-the-loop step.

```python
from google.adk.agents import BaseAgent

class HumanReviewAgent(BaseAgent):
    """Pauses pipeline for human script review and approval."""

    async def _run_async_impl(self, ctx):
        draft = ctx.session.state["draft_script"]

        # Present script to user via UI callback
        user_edits = await self._request_human_review(draft)

        if user_edits.approved:
            ctx.session.state["approved_script"] = user_edits.final_script
        else:
            ctx.session.state["approved_script"] = None
            # Could escalate or loop back to script writer
```

---

### 5. Parallel Production Agent - `ParallelAgent`

**Purpose:** Once the script is approved, multiple agents work simultaneously on independent tasks.

```python
parallel_production_agent = ParallelAgent(
    name="ParallelProductionAgent",
    description="Runs storyboarding, asset collection, hook design, and voice generation concurrently.",
    sub_agents=[
        storyboard_agent,
        asset_manager_agent,
        hook_designer_agent,
        voice_actor_agent,
    ],
)
```

#### 5a. Storyboard Artist - `LlmAgent`

```python
storyboard_agent = LlmAgent(
    name="StoryboardArtistAgent",
    model="gemini-2.5-pro",
    instruction="""You are a storyboard artist. Break down the approved script
    into a visual storyboard:

    Script: {approved_script}

    For each scene produce:
    - Scene number and duration
    - Visual description (composition, framing, style)
    - Text overlays / lower thirds
    - Transition type (cut, fade, zoom, etc.)
    - Asset requirements (stock footage, AI-generated, screen recording)
    """,
    tools=[image_generation_tool],
    output_key="storyboard",
)
```

#### 5b. Asset Manager - `LlmAgent`

```python
asset_manager_agent = LlmAgent(
    name="AssetManagerAgent",
    model="gemini-2.5-flash",
    instruction="""You are a video asset manager. Based on the approved script,
    find and collect all visual assets needed:

    Script: {approved_script}
    Research: {research_brief}

    Tasks:
    1. Search for relevant stock footage / images (Creative Commons, fair use)
    2. Find charts, graphs, data visualizations
    3. Locate brand logos, product images if referenced
    4. Download and catalog all assets with metadata
    5. Flag any assets that need AI generation
    """,
    tools=[google_search_tool, image_search_tool, stock_footage_tool, file_download_tool],
    output_key="collected_assets",
)
```

#### 5c. Hook Designer - `LlmAgent`

```python
hook_designer_agent = LlmAgent(
    name="HookDesignerAgent",
    model="gemini-2.5-pro",
    instruction="""You are a social media hook specialist. Design 3 different
    hook options for the video opening (first 3-5 seconds):

    Script: {approved_script}
    Target platform: {target_platform}

    For each hook provide:
    - Hook style (question, bold claim, statistic, controversy, story)
    - Visual treatment (text animation, dramatic zoom, split screen)
    - Text overlay content
    - Recommended background music mood

    Hooks must stop the scroll and match the content tone.
    """,
    output_key="hook_designs",
)
```

#### 5d. Voice Actor - `LlmAgent`

```python
voice_actor_agent = LlmAgent(
    name="VoiceActorAgent",
    model="gemini-2.5-flash",
    instruction="""Generate the voiceover for the video script.

    Script: {approved_script}
    Voice style: {voice_preference}

    Tasks:
    1. Clean the script text for TTS (remove stage directions)
    2. Add SSML markers for emphasis, pauses, pace changes
    3. Generate audio using the selected voice
    4. Produce per-scene audio segments with timestamps
    """,
    tools=[tts_tool, audio_processing_tool],
    output_key="voiceover_audio",
)
```

---

### 6. Motion Designer Agent - `LlmAgent`

**Purpose:** Takes the storyboard + collected assets and creates animations/motion graphics. This is where static images become dynamic visuals. Runs after parallel production because it depends on both storyboard and assets.

```python
motion_designer_agent = LlmAgent(
    name="MotionDesignerAgent",
    model="gemini-2.5-pro",
    instruction="""You are a motion graphics designer. Transform static assets
    into dynamic, visually appealing animations:

    Storyboard: {storyboard}
    Assets: {collected_assets}
    Hook designs: {hook_designs}

    For each scene:
    1. Select the best AI video model for the scene type
    2. Generate motion from static images (ken burns, parallax, 3D zoom)
    3. Create text animations and lower thirds
    4. Design transitions between scenes
    5. Generate any AI video clips needed for scenes flagged in storyboard

    Output motion specifications and generated video clips per scene.
    """,
    tools=[
        ai_video_generation_tool,   # Wraps video gen APIs
        image_animation_tool,        # Ken burns, parallax effects
        text_animation_tool,         # Animated text overlays
    ],
    output_key="motion_assets",
)
```

---

### 7. Video Editor Agent - `LlmAgent`

**Purpose:** Final assembly - combines all produced assets into the finished video.

```python
video_editor_agent = LlmAgent(
    name="VideoEditorAgent",
    model="gemini-2.5-pro",
    instruction="""You are a professional video editor. Assemble the final video
    from all produced components:

    Storyboard: {storyboard}
    Motion assets: {motion_assets}
    Voiceover: {voiceover_audio}
    Hook designs: {hook_designs}

    Tasks:
    1. Arrange scenes according to storyboard timeline
    2. Sync voiceover audio with visual scenes
    3. Apply the selected hook design to the opening
    4. Add background music and sound effects
    5. Add captions/subtitles synced to voiceover
    6. Render in platform-appropriate format:
       - 9:16 for TikTok/Reels/Shorts
       - 16:9 for YouTube
       - 1:1 for Instagram feed
    7. Export final video file
    """,
    tools=[
        video_compositor_tool,  # FFmpeg-based composition
        caption_generator_tool, # Auto-caption from audio
        music_selector_tool,    # Background music library
        render_tool,            # Final render in target formats
    ],
    output_key="final_video",
)
```

---

### 8. Quality Review Agent - `LoopAgent`

**Purpose:** Iteratively reviews the output and requests fixes if needed.

```python
quality_reviewer = LlmAgent(
    name="QualityReviewerAgent",
    model="gemini-2.5-pro",
    instruction="""Review the final video for quality issues:

    Video: {final_video}
    Original script: {approved_script}

    Check for:
    - Audio/visual sync issues
    - Caption accuracy
    - Scene duration matches script
    - Hook effectiveness
    - Overall flow and pacing

    If issues found, output fix instructions.
    If quality is acceptable, set approved=True to end the loop.
    """,
    output_key="review_result",
)

quality_review_agent = LoopAgent(
    name="QualityReviewLoop",
    max_iterations=3,
    sub_agents=[quality_reviewer, video_editor_agent],
)
```

---

## Tools Architecture

Each agent uses specific tools. Here's the complete tool map:

```
┌─────────────────────────────────────────────────────────┐
│                    TOOL LAYER                           │
├─────────────────┬───────────────────────────────────────┤
│ SEARCH & DATA   │ google_search (ADK built-in)          │
│                 │ web_scraper (custom - BeautifulSoup)   │
│                 │ image_search (Google/Bing Image API)   │
│                 │ stock_footage (Pexels/Pixabay API)     │
├─────────────────┼───────────────────────────────────────┤
│ AI GENERATION   │ ai_video_gen (Runway/Kling/Veo API)   │
│                 │ image_gen (Imagen/DALL-E/Flux)         │
│                 │ tts (Google Cloud TTS / ElevenLabs)    │
├─────────────────┼───────────────────────────────────────┤
│ MEDIA PROCESS   │ video_compositor (FFmpeg wrapper)      │
│                 │ image_animation (Pillow + MoviePy)     │
│                 │ text_animation (custom renderer)       │
│                 │ caption_generator (Whisper)            │
│                 │ audio_processor (pydub)                │
├─────────────────┼───────────────────────────────────────┤
│ STORAGE & I/O   │ file_download (httpx)                 │
│                 │ cloud_storage (GCS bucket)             │
│                 │ render_tool (FFmpeg final encode)      │
│                 │ music_selector (royalty-free library)   │
└─────────────────┴───────────────────────────────────────┘
```

---

## State Flow (Session State Keys)

```
user_concept          ──→  ResearcherAgent
source_links          ──→  ResearcherAgent
target_platform       ──→  ScriptWriter, HookDesigner, VideoEditor
target_duration       ──→  ScriptWriter
voice_preference      ──→  VoiceActorAgent

research_brief        ←──  ResearcherAgent    ──→  ScriptWriter, AssetManager
draft_script          ←──  ScriptWriter       ──→  HumanReview
approved_script       ←──  HumanReview        ──→  All Production Agents
storyboard            ←──  StoryboardArtist   ──→  MotionDesigner, VideoEditor
collected_assets      ←──  AssetManager       ──→  MotionDesigner
hook_designs          ←──  HookDesigner       ──→  MotionDesigner, VideoEditor
voiceover_audio       ←──  VoiceActor         ──→  VideoEditor
motion_assets         ←──  MotionDesigner     ──→  VideoEditor
final_video           ←──  VideoEditor        ──→  QualityReview
review_result         ←──  QualityReview
```

---

## Project Structure

```
video_agent/
├── agent.py                    # Root agent definition (entry point for ADK)
├── agents/
│   ├── __init__.py
│   ├── researcher.py           # ResearcherAgent
│   ├── script_writer.py        # ScriptWriterAgent
│   ├── human_review.py         # HumanReviewAgent (custom BaseAgent)
│   ├── storyboard_artist.py    # StoryboardArtistAgent
│   ├── asset_manager.py        # AssetManagerAgent
│   ├── hook_designer.py        # HookDesignerAgent
│   ├── voice_actor.py          # VoiceActorAgent
│   ├── motion_designer.py      # MotionDesignerAgent
│   ├── video_editor.py         # VideoEditorAgent
│   └── quality_reviewer.py     # QualityReviewerAgent
├── tools/
│   ├── __init__.py
│   ├── search_tools.py         # Web search, image search, scraping
│   ├── ai_generation_tools.py  # Video gen, image gen, TTS
│   ├── media_tools.py          # FFmpeg, MoviePy, Pillow wrappers
│   └── storage_tools.py        # File download, cloud storage
├── config/
│   ├── models.py               # Model selection per agent
│   ├── platforms.py             # Platform specs (aspect ratios, limits)
│   └── prompts.py              # Prompt templates
├── ui/
│   ├── app.py                  # Streamlit/Gradio frontend
│   └── review_interface.py     # Script review UI
├── requirements.txt
├── .env                        # API keys (not committed)
└── ARCHITECTURE.md             # This file
```

---

## External API Dependencies

| Service | Purpose | Alternative |
|---------|---------|-------------|
| **Gemini 2.5 Pro** | LLM backbone for all agents | Gemini Flash for lighter agents |
| **Google Search API** | Web research | SerpAPI, Tavily |
| **Google Cloud TTS** | Voiceover generation | ElevenLabs, OpenAI TTS |
| **Runway ML** | AI video generation | Kling, Veo, Pika, Luma |
| **Pexels / Pixabay** | Stock footage & images | Unsplash, Shutterstock |
| **FFmpeg** | Video composition & rendering | MoviePy (Python wrapper) |
| **Whisper** | Caption generation from audio | Google Speech-to-Text |
| **GCS** | Asset & output storage | S3, local filesystem |

---

## Multi-Model Video Selection (Agent Opus Pattern)

Following Agent Opus's approach, the Motion Designer Agent should select the optimal AI video model **per scene** based on content type:

```python
MODEL_SELECTION_RULES = {
    "talking_head":     "kling",       # Best for human faces/movement
    "landscape_scenic": "runway",      # Best for cinematic nature shots
    "data_viz":         "custom",      # Use matplotlib + animation
    "product_showcase": "veo",         # Best for object-focused shots
    "abstract_concept": "pika",        # Best for creative/abstract
    "text_animation":   "custom",      # Use Pillow + MoviePy
    "screen_recording": "none",        # Use actual screenshot/recording
}
```

---

## Getting Started

```bash
# 1. Install Google ADK
pip install google-adk

# 2. Set up API keys
export GOOGLE_API_KEY="your-gemini-key"
export RUNWAY_API_KEY="your-runway-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
export PEXELS_API_KEY="your-pexels-key"

# 3. Run with ADK dev server
adk web .

# 4. Or run programmatically
python agent.py --concept "Cost of migrating to Mars" --links "https://..."
```

---

## Key Design Decisions

1. **SequentialAgent as root** - Video production is inherently sequential (research → script → review → produce → edit)
2. **ParallelAgent for production** - Storyboarding, asset collection, hook design, and voiceover are independent and run concurrently for speed
3. **Human-in-the-loop** - Script review is non-negotiable; the user must control the narrative
4. **LoopAgent for quality** - Iterative refinement catches sync issues, caption errors, etc.
5. **Per-scene model selection** - Different AI video models excel at different content types
6. **State-based data flow** - All agents communicate via ADK session state (`output_key` / `{variable}` references)
