import os
import whisper
from groq import Groq
from gtts import gTTS
import streamlit as st
import tempfile
import io

# Retrieve the API key from Streamlit secrets
api_key = st.secrets["general"]["GROQ_API_KEY"]  # Make sure you have set it in Streamlit secrets

if api_key:
    st.write("API key is successfully retrieved")
else:
    st.write("API key not found")

# Initialize Whisper Model
whisper_model = whisper.load_model("base")

# Initialize Groq Client with the API key
client = Groq(api_key=api_key)

# Function for Text-to-Speech using gTTS
def text_to_voice_gtts(text):
    tts = gTTS(text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Main Function for the Chatbot Workflow
def voice_to_voice(audio_file):
    # Step 1: Speech-to-Text with Whisper
    result = whisper_model.transcribe(audio_file)
    user_text = result["text"]
    
    # Step 2: Interaction with Groq LLM
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_text,
            }
        ],
        model="llama3-8b-8192",
    )
    bot_response = chat_completion.choices[0].message.content
    
    # Step 3: Text-to-Speech with gTTS
    audio_output = text_to_voice_gtts(bot_response)
    
    return bot_response, audio_output

# Streamlit Interface
st.title("Real-Time Voice-to-Voice Chatbot")
st.markdown("### Record your voice, and the chatbot will respond in both text and audio")

# Audio input component for recording user's voice
audio_input = st.audio(label="Record Your Voice", type="audio/wav")

# If audio is uploaded
if audio_input:
    # Save the audio input to a temporary file
    audio_file = io.BytesIO(audio_input)

    # Get the response and audio output from the voice_to_voice function
    bot_response, audio_output = voice_to_voice(audio_file)

    # Display the chatbot response
    st.subheader("Chatbot Response")
    st.text(bot_response)

    # Display the audio output (Chatbot voice)
    st.subheader("Chatbot Voice Output")
    st.audio(audio_output, format="audio/mp3")
