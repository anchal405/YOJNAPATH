from dotenv import load_dotenv
import os
import sys
from uuid import uuid4, uuid5, UUID
from typing import cast
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    azure,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from langgraph_adapter import LangGraphAdapter

# Add the parent directory to the Python path for your imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your LangGraph implementation
from langgraph_app.graph_builder import (
    build_yojnapath_graph,
    init_conversation,
    add_user_input,
    stages,
    State
)

load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")

# For generating thread IDs for LangGraph state management
NAMESPACE = UUID("41010b5d-5447-4df5-baf2-97d69f2e9d06")

def get_thread_id(sid: str | None) -> str:
    """Generate a unique thread ID for each participant"""
    if sid is not None:
        return str(uuid5(NAMESPACE, sid))
    return str(uuid4())

class YojnaPathAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""आप YojnaPath एआई असिस्टेंट हैं। आपका मुख्य कार्य है:
            1. सरकारी योजनाओं की जानकारी प्रदान करना
            2. योजनाओं की पात्रता की जांच करना
            3. आवेदन प्रक्रिया में सहायता करना
            4. हमेशा हिंदी में उत्तर देना
            5. विनम्र और सहायक रहना"""
        )

def prewarm_resources():
    """Pre-warm necessary resources"""
    try:
        # Check if Groq API key is set
        if not os.getenv("GROQ_API_KEY"):
            print("⚠️  Warning: GROQ_API_KEY environment variable not set!")
            print("Please set your Groq API key in the .env file to use the LLM functionality.")
            raise ValueError("Missing GROQ_API_KEY")
        
        # Build and test the graph
        print("📊 Pre-warming YojnaPath LangGraph...")
        graph = build_yojnapath_graph()
        print("✅ Graph pre-warmed successfully!")
        return graph
    except Exception as e:
        print(f"❌ Error pre-warming resources: {e}")
        raise

async def entrypoint(ctx: agents.JobContext):
    stt_region = service_region if service_region is not None else ""
    tts_region = service_region if service_region is not None else ""
    
    if not speech_key or not stt_region or not tts_region:
        raise ValueError("Missing Azure Speech credentials")

    # Connect and wait for participant
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    thread_id = get_thread_id(participant.sid)
    
    print(f"🚀 Starting YojnaPath voice assistant for participant {participant.identity} (thread ID: {thread_id})")

    try:
        # Build your YojnaPath graph
        print("📊 Building YojnaPath LangGraph...")
        graph = build_yojnapath_graph()
        print("✅ Graph built successfully!")
        
        # Create LangGraph adapter with thread configuration
        langgraph_llm = LangGraphAdapter(
            graph=graph,
            config={
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 10
            }
        )

        session = AgentSession(
            stt=azure.STT(
                speech_key=speech_key, 
                speech_region=stt_region, 
                language="hi-IN"
            ),
            llm=langgraph_llm,  # Use YojnaPath LangGraph adapter
            tts=azure.TTS(
                speech_key=speech_key, 
                speech_region=tts_region, 
                voice="hi-IN-SwaraNeural"
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        await session.start(
            room=ctx.room,
            agent=YojnaPathAssistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        # Initial greeting in Hindi - using YojnaPath context
        await session.generate_reply(
            instructions="""यूज़र का अभिवादन करें और उन्हें YojnaPath के बारे में बताएं। 
            कहें कि आप सरकारी योजनाओं की जानकारी देने में सहायता कर सकते हैं।
            उदाहरण के लिए: PM Kisan, Pradhan Mantri Awas Yojana, आदि।"""
        )

        print("✅ YojnaPath voice assistant started successfully!")
        
    except Exception as e:
        print(f"❌ Error starting YojnaPath assistant: {e}")
        print("This might be due to missing Groq API key or network issues.")
        print("You can set GROQ_API_KEY in a .env file or as an environment variable.")
        raise

# Custom worker options with pre-warming
class YojnaPathWorkerOptions(agents.WorkerOptions):
    def __init__(self):
        super().__init__(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=lambda proc: prewarm_resources()
        )

def main():
    """Main function to run the YojnaPath LiveKit agent"""
    print("🎯 Starting YojnaPath LiveKit Voice Agent")
    print("=" * 50)
    
    # Validate environment
    required_env_vars = [
        "AZURE_SPEECH_KEY",
        "AZURE_SPEECH_REGION", 
        "GROQ_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return
    
    print("✅ Environment validation passed")
    
    # Run the agent
    try:
        agents.cli.run_app(YojnaPathWorkerOptions())
    except KeyboardInterrupt:
        print("\n👋 YojnaPath agent stopped by user")
    except Exception as e:
        print(f"❌ Error running YojnaPath agent: {e}")

if __name__ == "__main__":
    main()