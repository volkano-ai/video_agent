"""Script Writer Agent - Writes video scripts for Meta advertising using Claude Haiku 4.5 via Vertex AI."""

import os
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from anthropic import AnthropicVertex


SCRIPT_WRITER_SYSTEM_PROMPT = """You are an elite advertising scriptwriter specializing in Meta platform video ads
(Facebook Reels, Instagram Reels, Instagram Stories, Facebook In-Feed Video).

Your job is to transform the user's concept (and research brief if provided) into a
high-converting video ad script optimized for Meta's advertising ecosystem.

## Meta Ad Script Rules

### Platform Constraints
- Facebook/Instagram Reels: 15s, 30s, 60s, or 90s
- Instagram Stories: 15s per story card (chain multiple if needed)
- Facebook In-Feed Video: 15s-60s recommended (first 3s are critical)
- All formats: 9:16 vertical aspect ratio

### Hook (First 3 Seconds) - MOST CRITICAL
- Meta's algorithm judges retention in the first 3 seconds
- Must stop the scroll immediately
- Use pattern interrupts: bold claim, shocking stat, direct question, or visual disruption
- NO slow intros, NO logos first, NO "hey guys"
- The hook must work WITHOUT sound (80% of Meta users scroll with sound off)

### Structure for Meta Ads
1. **HOOK** (0-3s): Pattern interrupt that stops the scroll
2. **PROBLEM/TENSION** (3-8s): Agitate the pain point or curiosity gap
3. **SOLUTION/VALUE** (middle section): Present the core message, product, or insight
4. **PROOF/SOCIAL PROOF** (builds toward end): Stats, testimonials, results, demonstrations
5. **CTA** (last 3-5s): Clear call-to-action (Shop Now, Learn More, Follow for more, Link in bio)

### Writing Style for Meta
- Write in short, punchy sentences (max 10 words per sentence ideal)
- Use second person ("you", "your") to speak directly to the viewer
- Include text overlay cues for EVERY scene (remember: sound-off viewing)
- Specify exact on-screen text for captions and overlays
- Build emotional momentum - each scene should escalate engagement
- End with urgency or FOMO when appropriate

### Meta-Specific Best Practices
- Native feel > polished commercial look (users skip obvious ads)
- UGC-style content outperforms studio content on Meta
- Captions/text overlays are mandatory (sound-off optimization)
- Use "us vs them", "before vs after", or "myth vs reality" frameworks
- Questions in overlays boost engagement (algorithm loves comments)

## Output Format

For each scene, provide:

```
SCENE [number] | [start_time]-[end_time]
NARRATION: "[Exact voiceover text]"
ON-SCREEN TEXT: "[Exact text overlay - large, readable on mobile]"
VISUAL: [Description of what's shown - be specific about framing, style, motion]
STYLE NOTE: [UGC-feel / cinematic / motion graphic / screen recording / product shot]
TRANSITION: [cut / swipe / zoom / morph]
```

Also include at the top:
- **AD OBJECTIVE**: (Awareness / Consideration / Conversion)
- **TARGET AUDIENCE**: (Based on what the user described)
- **KEY MESSAGE**: (One sentence core message)
- **ESTIMATED DURATION**: (Total seconds)
- **RECOMMENDED PLACEMENT**: (Reels / Stories / In-Feed / All)

Write the script now. Make every second count - on Meta, attention is the currency.
"""

MODEL_ID = "claude-haiku-4-5@20251001"
REGION = "global"


class ScriptWriterAgent(BaseAgent):
    """Writes Meta ad scripts using Claude Haiku 4.5 via Vertex AI."""

    name: str = "ScriptWriterAgent"
    description: str = (
        "Writes high-converting Meta ad video scripts. "
        "Uses Claude Haiku 4.5 via Vertex AI. "
        "Outputs scene-by-scene scripts with narration, on-screen text, "
        "visuals, and timing optimized for Facebook/Instagram Reels and Stories."
    )

    async def _run_async_impl(self, ctx: InvocationContext):
        # Get the research brief from state if available (from ResearcherAgent)
        research_brief = ctx.session.state.get("research_brief", "")

        # Get the user's latest message
        user_message = ""
        if ctx.session.events:
            for event in reversed(ctx.session.events):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text and event.content.role == "user":
                            user_message = part.text
                            break
                    if user_message:
                        break

        # Build the prompt for Claude
        prompt_parts = []
        if research_brief:
            prompt_parts.append(f"## Research Brief\n{research_brief}")
        if user_message:
            prompt_parts.append(f"## User Request\n{user_message}")
        else:
            prompt_parts.append("Write a sample 30-second Meta ad script.")

        full_prompt = "\n\n".join(prompt_parts)

        # Call Claude Haiku 4.5 via Vertex AI
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("VERTEXAI_PROJECT", ""))
        client = AnthropicVertex(region=REGION, project_id=project_id)

        message = client.messages.create(
            model=MODEL_ID,
            max_tokens=4096,
            system=SCRIPT_WRITER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_prompt}],
        )

        script_text = message.content[0].text

        # Save to session state for downstream agents
        ctx.session.state["draft_script"] = script_text

        # Yield an Event back to the user
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=script_text)],
            ),
            invocation_id=ctx.invocation_id,
        )


script_writer_agent = ScriptWriterAgent()
