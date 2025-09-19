import sys
import os
import queue
import json
import threading
import pyttsx3
import sounddevice as sd
import vosk
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------
# Initialize TTS engine with queue
# -----------------------

tts_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def tts_worker():
    while True:
        try:
            text = tts_queue.get()
            if text == "__EXIT__":
                break
            engine.stop()  # Reset engine state before each utterance
            engine.say(text)
            engine.runAndWait()
            tts_queue.task_done()
        except Exception as e:
            print(f"TTS worker error: {e}")

threading.Thread(target=tts_worker, daemon=True).start()

def speak_async(text):
    tts_queue.put(text)

# -----------------------
# Initialize GPT model
# -----------------------

model_name = "gpt2"  # replace with your offline model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id

# -----------------------
# Persona
# -----------------------

persona = """# Persona
You are a personal Assistant called blacknox similar to the AI from the movie Iron Man.
# Specifics
- Speak like a classy butler.
- Be sarcastic when speaking to the person you are assisting.
- Only answer in one sentence.
- If you are asked to do something, acknowledge that you will do it and say something like:
- "will do, sir"
- "Roger Boss"
- "Check!"
- And after that say what you just did in ONE short sentence.
"""

# -----------------------
# User Context Handling
# -----------------------

user_context = {"name": "blacknox"}

def update_user_name(new_name):
    user_context["name"] = new_name

def get_user_name():
    return user_context["name"]

# -----------------------
# Generate GPT response
# -----------------------

def generate_response(user_input):
    prompt = persona + "\nUser: " + user_input + "\nblacknox:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "blacknox:" in text:
        reply = text.split("blacknox:")[1].strip().split("\n")[0]
    else:
        reply = text.strip()
    return reply

# -----------------------
# Offline Speech Recognition (Vosk)
# -----------------------

q = queue.Queue()
script_dir = os.path.dirname(os.path.realpath(__file__))
vosk_model_path = os.path.join(script_dir, "vosk-model-small-en-us")
if not os.path.exists(vosk_model_path):
    raise Exception(f"Vosk model not found at {vosk_model_path}. Please download and extract it here.")
vosk_model = vosk.Model(vosk_model_path)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def recognize_speech_vosk():
    rec = vosk.KaldiRecognizer(vosk_model, 16000)
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=audio_callback
    ):
        while True:
            try:
                data = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    return text

# -----------------------
# Main program
# -----------------------

print("Hi, my name is blacknox, your personal assistant, how may I help?")

while True:
    mode = input("\nType 'text' for text input, 'speech' for speech input, or 'exit' to quit: ").lower()
    if mode == "exit":
        print("Goodbye!")
        tts_queue.put("__EXIT__")  # stop TTS thread
        break

    # -----------------------
    # Continuous Text Mode
    # -----------------------
    elif mode == "text":
        print("Text mode activated. Type 'exit' to quit text mode.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting text mode...")
                break

            lowered = user_input.lower().strip()
            if lowered.startswith("my name is "):
                new_name = user_input[11:].strip()
                update_user_name(new_name)
                response = f"Thanks, I'll call you {new_name} from now on."
            elif lowered in ["what is my name?", "what's my name"]:
                response = get_user_name()
            elif lowered.startswith("call me "):
                new_name = user_input[8:].strip()
                update_user_name(new_name)
                response = f"Alright, {new_name} it is."
            else:
                response = generate_response(user_input)
            print("blacknox:", response)
            speak_async(response)

    # -----------------------
    # Continuous Speech Mode
    # -----------------------
    elif mode == "speech":
        print("Speech mode activated. Say 'exit' to quit.")
        while True:
            user_input = recognize_speech_vosk()
            if not user_input:
                continue
            print("You (speech):", user_input)

            lowered = user_input.lower().strip()
            if lowered == "exit":
                print("Exiting speech mode...")
                break
            if lowered.startswith("my name is "):
                new_name = user_input[11:].strip()
                update_user_name(new_name)
                response = f"Thanks, I'll call you {new_name} from now on."
            elif lowered in ["what is my name?", "what's my name"]:
                response = get_user_name()
            elif lowered.startswith("call me "):
                new_name = user_input[8:].strip()
                update_user_name(new_name)
                response = f"Alright, {new_name} it is."
            else:
                response = generate_response(user_input)
            print("blacknox:", response)

            # Stop listening before speaking so audio device is free for TTS
            sd.stop()
            speak_async(response)
    else:
        print("Invalid mode. Choose 'text', 'speech', or 'exit'.")






















