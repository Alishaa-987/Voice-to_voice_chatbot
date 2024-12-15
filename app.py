import os
import whisper
from groq import Groq
from gtts import gTTS
import gradio as gr
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Retrieve the API key from environment variables
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set GROQ_API_KEY in your environment.")
logging.info("API key successfully retrieved.")

# Initialize Whisper Model
whisper_model = whisper.load_model("base")

# Initialize Groq Client with the API key
client = Groq(api_key=api_key)

# Function for Text-to-Speech using gTTS
def text_to_voice_gtts(text):
    tts = gTTS(text, lang="en")
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    audio_path = temp_file.name
    temp_file.close()
    return audio_path

# Main Function for the Chatbot Workflow
def voice_to_voice(audio_file):
    try:
        # Check if audio_file is provided and valid
        if not audio_file:
            return "Error: No audio file provided. Please record your voice again.", None

        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found at {audio_file}")
            return "Error: Invalid or missing audio file. Please record your voice again.", None

        logging.debug(f"Processing audio file: {audio_file}")

        # Step 1: Speech-to-Text with Whisper
        result = whisper_model.transcribe(audio_file)
        user_text = result.get("text", "").strip()

        if not user_text:
            logging.error("Transcription resulted in empty text.")
            return "Error: Unable to transcribe audio. Please try again.", None

        logging.info(f"Transcribed text: {user_text}")

        # Step 2: Interaction with Groq LLM
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_text}],
            model="llama3-8b-8192",
        )
        bot_response = chat_completion.choices[0].message.content

        logging.info(f"Chatbot response: {bot_response}")

        # Step 3: Text-to-Speech with gTTS
        audio_output = text_to_voice_gtts(bot_response)

        return bot_response, audio_output
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return f"Error: {str(e)}", None

# Gradio Interface with Reset Capability
with gr.Blocks() as demo:
    gr.Markdown("### Real-Time Voice-to-Voice Chatbot")
    audio_input = gr.Audio(type="filepath", label="Record Your Voice")
    chatbot_output = gr.Textbox(label="Chatbot Response")
    audio_output = gr.Audio(label="Chatbot Voice Output")

    submit_button = gr.Button("Submit")
    reset_button = gr.Button("Reset")

    submit_button.click(
        voice_to_voice,
        inputs=[audio_input],
        outputs=[chatbot_output, audio_output],
    )
    reset_button.click(
        lambda: ("", None),
        inputs=[],
        outputs=[chatbot_output, audio_output],
    )

# Launch the Gradio App
if __name__ == "__main__":
    demo.launch()
