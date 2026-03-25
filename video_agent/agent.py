"""Video Agent - Root agent definition for Google ADK."""

from google.adk.agents import SequentialAgent
from video_agent.agents.researcher import researcher_agent
from video_agent.agents.script_writer import script_writer_agent
from video_agent.agents.storyboard_artist import storyboard_agent

# --- Toggle which agent to run ---

# Run only the Storyboard Agent:
root_agent = storyboard_agent

# Run the full pipeline (Research → Script → Storyboard):
# root_agent = SequentialAgent(
#     name="VideoProductionPipeline",
#     description="End-to-end video production pipeline.",
#     sub_agents=[researcher_agent, script_writer_agent, storyboard_agent],
# )
