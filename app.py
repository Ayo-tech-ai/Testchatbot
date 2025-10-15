import streamlit as st
from transformers import pipeline
from gtts import gTTS
from audiorecorder import audiorecorder
import tempfile
import whisper
import os
import time

# -------------------------------
# 1. Load Models
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Automatically downloads first run

qa_pipeline = load_qa_model()
whisper_model = load_whisper_model()

# -------------------------------
# 2. Context (Knowledge Base)
# -------------------------------

sections = {
    "bacterial leaf blight": """Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae...""",
    "bacterial leaf streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola...""",
    "bakanae": """Bakanae Disease, also known as “Foolish Seedling,” is caused by the fungus Fusarium fujikuroi...""",
    "brown spot": """Brown Spot, or Helminthosporiosis, is caused by Bipolaris oryzae...""",
    "grassy stunt": """Rice Grassy Stunt Virus (RGSV) is a viral disease spread by the brown planthopper...""",
    "narrow brown spot": """Narrow Brown Spot, caused by the fungus Cercospora janseana...""",
    "ragged stunt": """Rice Ragged Stunt Virus (RRSV), also spread by brown planthoppers...""",
    "rice blast": """Rice Blast, caused by the fungus Magnaporthe oryzae...""",
    "false smut": """Rice False Smut, caused by Ustilaginoidea virens...""",
    "sheath blight": """Sheath Blight is caused by Rhizoctonia solani...""",
    "sheath rot": """Sheath Rot, caused by Sarocladium oryzae...""",
    "stem rot": """Stem Rot, caused by Sclerotium oryzae...""",
    "tungro": """Rice Tungro Virus is a dual infection caused by Rice Tungro Bacilliform and Spherical Viruses..."""
}

# -------------------------------
# 3. Page Setup & Styling
# -------------------------------

st.set_page_config(page_title="🌾 Voice-Enabled Rice Q&A Assistant", layout="centered")

st.markdown("""
<style>
.chat-container {
    max-width: 700px;
    margin: auto;
    padding: 10px;
}
.msg {
    border-radius: 12px;
    padding: 10px 15px;
    margin-bottom: 10px;
    max-width: 80%;
}
.user {
    background-color: #DCF8C6;
    align-self: flex-end;
    text-align: right;
    margin-left: auto;
}
.bot {
    background-color: #F1F0F0;
    align-self: flex-start;
}
.role {
    font-size: 0.75em;
    font-weight: bold;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 4. Initialize Chat History
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "bot", "content": "Hello 👋🏽! You can type or record your question about rice diseases — e.g., 'What causes rice blast?'."}
    ]

# -------------------------------
# 5. Display Chat History
# -------------------------------

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.history:
    role_class = "user" if msg["role"] == "user" else "bot"
    st.markdown(
        f"""
        <div class='msg {role_class}'>
            <div class='role'>{msg["role"].capitalize()}</div>
            <div>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 6. User Input Options (Text or Voice)
# -------------------------------

st.markdown("🎙️ **Record your voice question below:**")
audio = audiorecorder("🎧 Click to record", "Recording...")

question = None

# If audio recorded
if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    with st.spinner("Transcribing your voice..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            result = whisper_model.transcribe(tmp.name)
            question = result["text"]
        st.markdown(f"🗣️ You said: **{question}**")

# Text fallback
text_input = st.chat_input("💬 Type your question instead...")
if text_input:
    question = text_input

# -------------------------------
# 7. Output Mode Selection
# -------------------------------

output_mode = st.radio(
    "How do you want to receive the bot's response?",
    ["Text only", "Text + Voice"],
    horizontal=True
)

# -------------------------------
# 8. Process Question
# -------------------------------

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Analyzing your question..."):
        selected_context = None
        for keyword, section in sections.items():
            if keyword in question.lower():
                selected_context = section
                break
        time.sleep(1)

        if selected_context:
            result = qa_pipeline(question=question, context=selected_context)
            answer = result["answer"]
        else:
            answer = "I couldn’t match your question to a specific disease. Try including the name, like 'blast', 'blight', or 'tungro'."

    st.session_state.history.append({"role": "bot", "content": answer})

    # -------------------------------
    # 9. Optional Audio Output
    # -------------------------------

    if output_mode == "Text + Voice":
        with st.spinner("Generating voice reply..."):
            tts = gTTS(answer)
            audio_path = "bot_reply.mp3"
            tts.save(audio_path)
            st.audio(audio_path, autoplay=True)

    st.rerun()
