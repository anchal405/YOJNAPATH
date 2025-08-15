import azure.cognitiveservices.speech as speechsdk
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your Azure Speech service key and region
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Event handler for recognized speech
def recognized_handler(evt):
    print(f"Recognized: {evt.result.text}")

# Event handler for canceled recognition
def canceled_handler(evt):
    print(f"Canceled: {evt}")
    if evt.reason == speechsdk.CancellationReason.Error:
        print(f"Error details: {evt.error_details}")

# Attach the event handlers
speech_recognizer.recognized.connect(recognized_handler)
speech_recognizer.canceled.connect(canceled_handler)

# Start continuous recognition
print("Listening... (Press Ctrl+C to stop)")
speech_recognizer.start_continuous_recognition()

try:
    # Keep the program running
    while True:
        pass
except KeyboardInterrupt:
    print("Stopping recognition...")
    speech_recognizer.stop_continuous_recognition()