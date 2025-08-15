from dotenv import load_dotenv
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    azure,
    groq,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="आप एक सहायक वॉयस एआई असिस्टेंट हैं। हमेशा हिंदी में उत्तर दें।"
        )

async def entrypoint(ctx: agents.JobContext):
    stt_region = service_region if service_region is not None else ""
    tts_region = service_region if service_region is not None else ""
    if not speech_key or not stt_region or not tts_region:
        raise ValueError("Missing Azure Speech credentials")

    session = AgentSession(
        stt=azure.STT(speech_key=speech_key, speech_region=stt_region, language="hi-IN"),
        llm=groq.LLM(model="llama3-8b-8192"),
        tts=azure.TTS(speech_key=speech_key, speech_region=tts_region, voice="hi-IN-SwaraNeural"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    # Instruct the assistant to respond in Hindi
    await session.generate_reply(
        instructions="यूज़र का अभिवादन करें और सहायता की पेशकश करें।"
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))