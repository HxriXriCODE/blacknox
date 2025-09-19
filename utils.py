from transformers import AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr

# Load tokenizer and model from local folder
tokenizer = AutoTokenizer.from_pretrained('./your-model')
model = AutoModelForCausalLM.from_pretrained('./your-model')

# Fix: Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Function to get response from model
def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Speech-to-text conversion function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = sr.recognize_google(audio)
        print(f"You (speech): {text}")
        return text
    except Exception:
        print("Sorry, I didn't catch that.")
        return ""
