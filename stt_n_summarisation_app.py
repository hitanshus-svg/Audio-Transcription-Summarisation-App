import gradio as gr
from transformers import pipeline
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# ---------------- IBM LLM ----------------
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 0.3,
}

model = Model(
    model_id="mistralai/mistral-medium-2505",
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(model)

# ---------------- Whisper (LOAD ONCE) ----------------
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny.en",
    chunk_length_s=30
)

# ---------------- FUNCTION ----------------
def transcript_audio(audio_file):
    transcript_text = pipe(audio_file)["text"]

    result = llm.invoke(f"summarize this text:\n{transcript_text}")

    return str(result)

# ---------------- GRADIO UI ----------------
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription + Summarization"
)

iface.launch(server_name="0.0.0.0", server_port=8000)