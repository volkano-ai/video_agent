"""Storyboard Artist Agent - Generates scene-by-scene storyboard images using Gemini 3.1 Flash.

Images are uploaded to GCS bucket 'video_generation_agent' under {workspace_id}/{timestamp}/.
Output is a structured JSON mapping each script line to its generated image URL.
"""

import io
import json
import os
from datetime import datetime
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from google import genai
from google.cloud import storage as gcs


IMAGE_MODEL = "gemini-3.1-flash-image-preview"
GCS_BUCKET = "video_generation_agent"

STORYBOARD_PLANNER_PROMPT = """You are a storyboard artist. Analyze the following video ad script and break it
down into individual storyboard frames.

## Script
{script}

For EACH scene in the script, produce a detailed image generation prompt that will create
a storyboard frame. Each prompt should describe:
- The exact visual composition (what's in frame, camera angle, framing)
- The style (UGC-feel, cinematic, motion graphic, product shot, etc.)
- Any text overlays that should appear in the image
- The mood and lighting
- The aspect ratio is always 9:16 (vertical, mobile-first)

Also extract the EXACT narration/script line for that scene from the original script.

Output your response as a numbered list in this EXACT format:

FRAME 1 | [scene time range]
SCRIPT_LINE: [exact narration/voiceover text for this scene]
PROMPT: [detailed image generation prompt for this frame]
OVERLAY_TEXT: [exact on-screen text that appears, or NONE]

FRAME 2 | [scene time range]
SCRIPT_LINE: [exact narration/voiceover text for this scene]
PROMPT: [detailed image generation prompt]
OVERLAY_TEXT: [exact text, or NONE]

... and so on for every scene.

Make the prompts vivid and specific. Think like a cinematographer - every frame must
stop the scroll on Meta. Use bold colors, high contrast, and eye-catching compositions.
"""


class StoryboardArtistAgent(BaseAgent):
    """Generates storyboard images for each scene using Gemini 3.1 Flash Image.

    Images are stored in GCS and output is a JSON mapping script lines to image URLs.
    """

    name: str = "StoryboardArtistAgent"
    description: str = (
        "Takes the approved script and generates a visual storyboard with one image per scene "
        "using Gemini 3.1 Flash Image generation. Uploads images to GCS and outputs structured JSON "
        "mapping each script line to its generated image."
    )

    def _parse_frames(self, planner_text: str) -> list[dict]:
        """Parse the planner output into structured frames."""
        frames = []
        current_frame = {}

        for line in planner_text.split("\n"):
            line = line.strip()
            if line.startswith("FRAME "):
                if current_frame:
                    frames.append(current_frame)
                parts = line.split("|", 1)
                frame_num = parts[0].strip()
                time_range = parts[1].strip() if len(parts) > 1 else ""
                current_frame = {
                    "frame": frame_num,
                    "time_range": time_range,
                    "script_line": "",
                    "prompt": "",
                    "overlay_text": "",
                }
            elif line.startswith("SCRIPT_LINE:"):
                current_frame["script_line"] = line[len("SCRIPT_LINE:"):].strip()
            elif line.startswith("PROMPT:"):
                current_frame["prompt"] = line[len("PROMPT:"):].strip()
            elif line.startswith("OVERLAY_TEXT:"):
                current_frame["overlay_text"] = line[len("OVERLAY_TEXT:"):].strip()

        if current_frame:
            frames.append(current_frame)

        return frames

    def _upload_to_gcs(self, bucket: gcs.bucket.Bucket, blob_path: str, image_bytes: bytes) -> str:
        """Upload image bytes to GCS and return the public URL."""
        blob = bucket.blob(blob_path)
        blob.upload_from_file(io.BytesIO(image_bytes), content_type="image/png")
        return f"gs://{GCS_BUCKET}/{blob_path}"

    async def _run_async_impl(self, ctx: InvocationContext):
        # Get workspace_id from state or user message
        workspace_id = ctx.session.state.get("workspace_id", "")

        # Get the draft script from state
        draft_script = ctx.session.state.get("draft_script", "")

        # Fall back to user message for both workspace_id and script
        if not draft_script or not workspace_id:
            for event in reversed(ctx.session.events):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text and event.content.role == "user":
                            user_text = part.text
                            if not workspace_id:
                                # Try to extract workspace_id from message
                                for line in user_text.split("\n"):
                                    line_lower = line.strip().lower()
                                    if "workspace_id" in line_lower or "workspace id" in line_lower:
                                        # Extract value after : or =
                                        for sep in [":", "="]:
                                            if sep in line:
                                                workspace_id = line.split(sep, 1)[1].strip().strip("\"'")
                                                break
                                        break
                            if not draft_script:
                                draft_script = user_text
                            break
                    if draft_script:
                        break

        if not workspace_id:
            workspace_id = "default_workspace"

        if not draft_script:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="No script found. Please provide a script or run the ScriptWriter first."
                    )],
                ),
                invocation_id=ctx.invocation_id,
            )
            return

        # Build GCS folder path: {workspace_id}/{YYYYMMDD_HHMMSS}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_folder = f"{workspace_id}/{timestamp}"

        # Initialize clients
        gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        gcs_client = gcs.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", os.getenv("VERTEXAI_PROJECT", "")))
        bucket = gcs_client.bucket(GCS_BUCKET)

        # Step 1: Plan storyboard frames from script
        planner_prompt = STORYBOARD_PLANNER_PROMPT.format(script=draft_script)
        plan_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=planner_prompt,
        )
        planner_text = plan_response.text
        frames = self._parse_frames(planner_text)

        if not frames:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text=f"Failed to parse storyboard frames. Raw plan:\n\n{planner_text}"
                    )],
                ),
                invocation_id=ctx.invocation_id,
            )
            return

        # Step 2: Generate images and upload to GCS
        storyboard_json = {
            "workspace_id": workspace_id,
            "timestamp": timestamp,
            "gcs_folder": f"gs://{GCS_BUCKET}/{gcs_folder}",
            "total_frames": len(frames),
            "frames": [],
        }

        for i, frame in enumerate(frames):
            frame_number = i + 1
            image_prompt = (
                f"Create a storyboard frame for a vertical 9:16 social media video ad. "
                f"{frame['prompt']} "
                f"Style: Bold, eye-catching, high contrast, mobile-optimized."
            )

            if frame.get("overlay_text") and frame["overlay_text"].upper() != "NONE":
                image_prompt += f" Include this text overlay prominently: \"{frame['overlay_text']}\""

            frame_entry = {
                "frame_number": frame_number,
                "time_range": frame["time_range"],
                "script_line": frame["script_line"],
                "overlay_text": frame["overlay_text"],
                "image_url": None,
                "status": "pending",
            }

            try:
                img_response = gemini_client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=image_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio="9:16",
                        ),
                    ),
                )

                # Find the image part and upload to GCS
                for part in img_response.parts:
                    if part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        blob_path = f"{gcs_folder}/frame_{frame_number:02d}.png"
                        gcs_url = self._upload_to_gcs(bucket, blob_path, image_bytes)

                        frame_entry["image_url"] = gcs_url
                        frame_entry["status"] = "success"
                        break

                if frame_entry["status"] != "success":
                    fallback = img_response.text if img_response.text else "No image generated"
                    frame_entry["status"] = "failed"
                    frame_entry["error"] = fallback

            except Exception as e:
                frame_entry["status"] = "error"
                frame_entry["error"] = str(e)

            storyboard_json["frames"].append(frame_entry)

        # Summary counts
        success_count = sum(1 for f in storyboard_json["frames"] if f["status"] == "success")
        storyboard_json["frames_generated"] = success_count

        # Save to state for downstream agents
        ctx.session.state["storyboard"] = storyboard_json

        # Output only the structured JSON
        output_text = json.dumps(storyboard_json, indent=2)

        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=output_text)],
            ),
            invocation_id=ctx.invocation_id,
        )


storyboard_agent = StoryboardArtistAgent()
