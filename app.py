import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import whisper
import time
import os
import base64

# -------------------------------
# 1. Load Models
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

qa_pipeline = load_qa_model()
whisper_model = load_whisper_model()

# -------------------------------
# 2. Knowledge Base
# -------------------------------

sections = {
    "bacterial leaf blight": "Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It affects leaves, leading to yellowing and wilting.",
    "bacterial leaf streak": "Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. It causes narrow, water-soaked streaks on leaves.",
    "bakanae": "Bakanae Disease, also known as 'Foolish Seedling', is caused by the fungus Fusarium fujikuroi and leads to elongated, thin seedlings.",
    "brown spot": "Brown Spot is caused by Bipolaris oryzae. It appears as brown circular spots on leaves, affecting photosynthesis.",
    "grassy stunt": "Rice Grassy Stunt Virus (RGSV) is spread by the brown planthopper and causes stunted growth.",
    "narrow brown spot": "Narrow Brown Spot is caused by the fungus Cercospora janseana, leading to narrow brown lesions on leaves.",
    "ragged stunt": "Rice Ragged Stunt Virus (RRSV), spread by brown planthoppers, causes ragged leaves and stunted plants.",
    "rice blast": "Rice Blast is caused by the fungus Magnaporthe oryzae, producing diamond-shaped lesions on leaves.",
    "false smut": "Rice False Smut, caused by Ustilaginoidea virens, produces greenish smut balls on grains.",
    "sheath blight": "Sheath Blight is caused by Rhizoctonia solani and forms lesions on the leaf sheath.",
    "sheath rot": "Sheath Rot is caused by Sarocladium oryzae, resulting in rotting of the uppermost leaf sheath.",
    "stem rot": "Stem Rot is caused by Sclerotium oryzae, which blackens the base of the stem.",
    "tungro": "Rice Tungro Virus is a dual infection (Rice Tungro Bacilliform and Spherical Viruses), causing yellow-orange leaf discoloration."
}

# -------------------------------
# 3. Page Setup
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

st.title("🌾 Rice Disease Q&A Assistant (with Voice)")

# -------------------------------
# 4. Chat History
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "bot", "content": "Hello 👋🏽! You can type or record your question about rice diseases — e.g., 'What causes rice blast?'."}
    ]

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
# 5. Voice Recording (Alternative Approach)
# -------------------------------

st.markdown("🎙️ **Record your voice question below:**")

# Option 1: Use st.audio_input (Streamlit's built-in audio recorder)
audio_file = st.audio_input("Record your question", label_visibility="collapsed")

question = None

if audio_file:
    st.audio(audio_file, format="audio/wav")
    
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing your voice..."):
        try:
            result = whisper_model.transcribe(tmp_path)
            question = result["text"]
            if question.strip():  # Only show if we got actual text
                st.markdown(f"🗣️ You said: **{question}**")
            else:
                st.warning("No speech detected in the recording. Please try again.")
                question = None
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            question = None
    
    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except:
        pass

# -------------------------------
# 6. Text Input Fallback
# -------------------------------

text_input = st.chat_input("💬 Or type your question here...")
if text_input:
    question = text_input

# -------------------------------
# 7. Output Mode Selection
# -------------------------------

output_mode = st.radio(
    "Choose how to receive the bot's response:",
    ["Text only", "Text + Voice"],
    horizontal=True
)

# -------------------------------
# 8. Generate Response
# -------------------------------

if question:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Analyzing your question..."):
        selected_context = None
        for keyword, section in sections.items():
            if keyword in question.lower():
                selected_context = section
                break

        time.sleep(1)  # simulate latency

        if selected_context:
            try:
                result = qa_pipeline(question=question, context=selected_context)
                answer = result["answer"]
            except Exception as e:
                answer = f"I encountered an error processing your question: {str(e)}"
        else:
            answer = "Hmm, I couldn't find that disease in my knowledge base. Try mentioning the name like 'blast', 'blight', or 'tungro'."

    st.session_state.history.append({"role": "bot", "content": answer})

    # Voice Output (if selected)
    if output_mode == "Text + Voice" and answer:
        with st.spinner("Generating voice reply..."):
            try:
                tts = gTTS(text=answer, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    # Read the audio file and display it
                    audio_bytes = open(tmp_audio.name, 'rb').read()
                    st.audio(audio_bytes, format="audio/mp3")
                # Clean up
                try:
                    os.unlink(tmp_audio.name)
                except:
                    pass
            except Exception as e:
                st.error(f"Error generating voice: {e}")

    st.rerun()

# -------------------------------
# 9. Clear Chat Button
# -------------------------------

if st.button("Clear Chat"):
    st.session_state.history = [
        {"role": "bot", "content": "Hello 👋🏽! You can type or record your question about rice diseases — e.g., 'What causes rice blast?'."}
    ]
    st.rerun()
