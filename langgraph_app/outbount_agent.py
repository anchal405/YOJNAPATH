from dotenv import load_dotenv
import os
import sys
import base64
import logging
from uuid import uuid4, uuid5, UUID
from typing import cast
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.telemetry import set_tracer_provider
from livekit.plugins import (
    google,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from langgraph_adapter import LangGraphAdapter
import asyncio
from livekit import api

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

speech_key = os.getenv("GOOGLE_SPEECH_KEY")
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_TRUNK_ID")

NAMESPACE = UUID("41010b5d-5447-4df5-baf2-97d69f2e9d06")

logger = logging.getLogger("yojnapath-outbound-agent")


def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    """Setup Langfuse OpenTelemetry tracing for the agent."""
    print("🔍 Initializing Langfuse OpenTelemetry tracing...")
    
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        print("✅ OpenTelemetry imports successful")
    except ImportError as e:
        print(f"❌ Failed to import OpenTelemetry modules: {e}")
        print("💡 Install missing dependencies: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")
        return False

    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")

    print(f"🔑 Credentials check:")
    print(f"   - Public key: {'✅ Provided' if public_key else '❌ Missing'}")
    print(f"   - Secret key: {'✅ Provided' if secret_key else '❌ Missing'}")
    print(f"   - Host: {'✅ Provided' if host else '❌ Missing'} ({host if host else 'None'})")

    if not public_key or not secret_key or not host:
        print("❌ LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST not set. Tracing disabled.")
        logger.warning("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST not set. Tracing disabled.")
        return False

    try:
        print("🔧 Setting up OpenTelemetry configuration...")
        langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        endpoint = f"{host.rstrip('/')}/api/public/otel"
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
        
        print(f"📡 OTLP Endpoint: {endpoint}")
        print(f"🔐 Auth header configured")

        print("🏗️  Creating TracerProvider...")
        trace_provider = TracerProvider()
        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        set_tracer_provider(trace_provider)
        
        print(f"🎉 Langfuse tracing successfully enabled!")
        print(f"📊 All agent interactions will be traced to: {host}")
        logger.info(f"✅ Langfuse tracing enabled with host: {host}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup Langfuse tracing: {e}")
        print(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        print("⚠️  Continuing without tracing...")
        logger.error(f"❌ Failed to setup Langfuse tracing: {e}")
        logger.warning("Continuing without tracing...")
        return False

def get_thread_id(sid: str | None) -> str:
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
            5. विनम्र और सहायक रहना
            6. आवश्यकता पड़ने पर फ़ोन कॉल करने की सुविधा प्रदान करना"""
        )
        self.participant = None
    
    def set_participant(self, participant):
        self.participant = participant

async def make_outbound_call(phone_number: str, room_name: str = None):
    try:
        if not SIP_TRUNK_ID:
            raise ValueError("SIP_TRUNK_ID not configured. Please set LIVEKIT_SIP_TRUNK_ID in your .env file")
        
        if not room_name:
            room_name = f"outbound-call-{uuid4().hex[:8]}"
        
        livekit_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
        
        sip_participant = await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=SIP_TRUNK_ID,
                sip_call_to=phone_number,
                room_name=room_name,
                participant_identity=f"sip-caller-{uuid4().hex[:8]}",
                participant_name="YojnaPath Assistant"
            )
        )
        
        print(f"✅ Outbound call initiated to {phone_number}")
        print(f"📞 Room: {room_name}")
        print(f"🆔 Participant ID: {sip_participant.participant_identity}")
        
        return {
            "success": True,
            "room_name": room_name,
            "participant_id": sip_participant.participant_identity,
            "phone_number": phone_number
        }
        
    except Exception as e:
        print(f"❌ Error making outbound call: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def prewarm_resources():
    try:
        required_vars = ["GROQ_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"⚠️  Warning: Missing environment variables: {missing_vars}")
            print("Please set these in your .env file for full functionality.")
        
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Missing GROQ_API_KEY")
        
        print("📊 Pre-warming YojnaPath LangGraph...")
        graph = build_yojnapath_graph()
        print("✅ Graph pre-warmed successfully!")
        return graph
    except Exception as e:
        print(f"❌ Error pre-warming resources: {e}")
        raise

# ✅ Top-level function replacing lambda
def prewarm_resources_wrapper(proc):
    return prewarm_resources()

async def entrypoint(ctx: agents.JobContext):
    print("\n" + "="*50)
    print("🚀 Starting YojnaPath Agent Session")
    print("="*50)
    
    # Setup Langfuse tracing before starting the agent session
    tracing_enabled = setup_langfuse(
        secret_key="sk-lf-6e0b5df1-efcc-4739-af58-f85711a2f714",
        public_key="pk-lf-96b284ab-5ca8-492a-8838-27b599473c20",
        host="https://us.cloud.langfuse.com"
    )
    
    # Log tracing status
    if not tracing_enabled:
        print("\n⚠️  WARNING: Tracing is disabled. No telemetry data will be collected.")
        print("   To enable tracing, ensure all required environment variables are set correctly.\n")
    
    print(f"\n🔗 Connecting to room: {ctx.room.name}")
    await ctx.connect()
    
    is_outbound_call = False
    dial_info = None
    
    if ctx.job.metadata:
        try:
            import json
            dial_info = json.loads(ctx.job.metadata)
            is_outbound_call = "phone_number" in dial_info
            print(f"📞 Outbound call detected: {is_outbound_call}")
            if is_outbound_call:
                print(f"📱 Calling: {dial_info['phone_number']}")
        except (json.JSONDecodeError, KeyError):
            pass
    
    graph = build_yojnapath_graph()
    thread_id = get_thread_id(ctx.room.name)
    print(f"🧵 Thread ID: {thread_id}")
    
    initial_state = init_conversation(thread_id)
    
    langgraph_adapter = LangGraphAdapter(
        graph=graph,
        config={
            "configurable": {
                "thread_id": thread_id
            },
            "recursion_limit": 10
        }
    )
    
    assistant = YojnaPathAssistant()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    credentials_path = os.path.join(script_dir, "zynga-backend-5558169672f7.json")
    
    stt = google.STT(
        languages=["hi-IN", "en-IN"],
        credentials_file=credentials_path
    )
    
    tts = google.TTS(
        voice_name="hi-IN-Standard-B",
        language="hi-IN",
        credentials_file=credentials_path,
        use_streaming=False
    )
    
    session = AgentSession(
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        stt=stt,
        tts=tts,
        llm=langgraph_adapter,
    )
    
    if is_outbound_call and dial_info:
        print("🚀 Starting agent session before dialing...")
        session_started = asyncio.create_task(
            session.start(
                agent=assistant,
                room=ctx.room,
                room_input_options=RoomInputOptions(
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )
        )
        
        phone_number = dial_info["phone_number"]
        participant_identity = f"sip-caller-{phone_number.replace('+', '').replace('-', '')}"
        
        try:
            print(f"📞 Dialing {phone_number}...")
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=SIP_TRUNK_ID,
                    sip_call_to=phone_number,
                    participant_identity=participant_identity,
                    wait_until_answered=True,
                )
            )
            
            await session_started
            participant = await ctx.wait_for_participant(identity=participant_identity)
            print(f"✅ Participant joined: {participant.identity}")
            
            if hasattr(assistant, 'set_participant'):
                assistant.set_participant(participant)
            
            await session.generate_reply(
                instructions="""यह एक आउटबाउंड कॉल है। यूज़र का अभिवादन करें और बताएं कि आप YojnaPath असिस्टेंट हैं। 
                सरकारी योजनाओं की जानकारी देने में आपकी सहायता कर सकते हैं।"""
            )
            
        except api.TwirpError as e:
            print(f"❌ Error creating SIP participant: {e.message}")
            print(f"SIP status: {e.metadata.get('sip_status_code')} {e.metadata.get('sip_status')}")
            ctx.shutdown()
            return
    
    else:
        participant = await ctx.wait_for_participant()
        thread_id = get_thread_id(participant.sid)
        print(f"🚀 Starting YojnaPath voice assistant for participant {participant.identity} (thread ID: {thread_id})")
        
        is_sip_call = participant.kind == "sip"
        if is_sip_call:
            print(f"📞 SIP call detected from participant: {participant.identity}")
        
        await session.start(
            agent=assistant,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
        
        if is_sip_call:
            await session.generate_reply(
                instructions="""यह एक इनबाउंड फ़ोन कॉल है। यूज़र का अभिवादन करें और बताएं कि आप YojnaPath असिस्टेंट हैं। 
                सरकारी योजनाओं की जानकारी देने में आपकी सहायता कर सकते हैं।"""
            )
        else:
            await session.generate_reply(
                instructions="""यूज़र का अभिवादन करें और उन्हें YojnaPath के बारे में बताएं। 
                कहें कि आप सरकारी योजनाओं की जानकारी देने में सहायता कर सकते हैं।"""
            )

    print("✅ YojnaPath voice assistant started successfully!")

class YojnaPathWorkerOptions(agents.WorkerOptions):
    def __init__(self):
        super().__init__(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_resources_wrapper,  # ✅ FIXED here
            agent_name="yojna-path-agent"
        )

def main():
    print("🎯 Starting YojnaPath LiveKit Voice Agent with SIP Support")
    print("=" * 60)
    
    required_env_vars = [
        "GOOGLE_SPEECH_KEY",
        "GROQ_API_KEY",
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET"
    ]
    
    optional_env_vars = ["LIVEKIT_SIP_TRUNK_ID"]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file:")
        for var in missing_vars:
            print(f"  {var}=your_value_here")
        return
    
    if missing_optional:
        print(f"⚠️  Missing optional environment variables: {missing_optional}")
        print("Set LIVEKIT_SIP_TRUNK_ID to enable outbound calling functionality")
    
    print("✅ Environment validation passed")
    
    try:
        agents.cli.run_app(YojnaPathWorkerOptions())
    except KeyboardInterrupt:
        print("\n👋 YojnaPath agent stopped by user")
    except Exception as e:
        print(f"❌ Error running YojnaPath agent: {e}")

async def test_outbound_call():
    phone_number = input("Enter phone number to call (e.g., +1234567890): ")
    result = await make_outbound_call(phone_number)
    
    if result["success"]:
        print(f"✅ Call initiated successfully!")
        print(f"Room: {result['room_name']}")
    else:
        print(f"❌ Call failed: {result['error']}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test-call":
        asyncio.run(test_outbound_call())
    else:
        main()
