import os
import whisper
from groq import Groq
from gtts import gTTS
import gradio as gr
import tempfile

# Retrieve the API key from environment variables
api_key = os.getenv('GROQ_API_KEY')  # Ensure your secret is set in GitHub Actions or your environment
if api_key:
    print("API key is successfully retrieved")
else:
    print("API key not found")

# Initialize Whisper Model
whisper_model = whisper.load_model("base")

# Initialize Groq Client with the API key
client = Groq(api_key=api_key)  # Or use os.getenv('GROQ_API_KEY')

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

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Real-Time Voice-to-Voice Chatbot")
    audio_input = gr.Audio(type="filepath", label="Record Your Voice")
    chatbot_output = gr.Textbox(label="Chatbot Response")
    audio_output = gr.Audio(label="Chatbot Voice Output")
    
    submit_button = gr.Button("Submit")
    submit_button.click(
        voice_to_voice,
        inputs=[audio_input],
        outputs=[chatbot_output, audio_output],
    )

# Launch the Gradio App
if __name__ == "__main__":
    demo.launch()
