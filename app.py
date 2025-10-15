import streamlit as st
from transformers import pipeline
import time
import base64
from gtts import gTTS
import io

# -------------------------------
# 1. Load Q&A Model
# -------------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_pipeline = load_qa_model()

# -------------------------------
# 2. Text-to-Speech Function
# -------------------------------

def text_to_speech(text):
    """Convert text to speech and return audio data as base64"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode()
        return audio_base64
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# -------------------------------
# 3. Context Sections (Knowledge Base)
# -------------------------------

sections = {
    "bacterial leaf blight": """Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae...""",
    "bacterial leaf streak": """Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola...""",
    "bakanae": """Bakanae Disease, also known as "Foolish Seedling," is caused by the fungus Fusarium fujikuroi...""",
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
# 4. Page Config and Styling
# -------------------------------

st.set_page_config(page_title="🌾 Rice Disease Q&A Assistant", layout="centered")

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
.audio-controls {
    margin-top: 10px;
    padding: 8px;
    background-color: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e9ecef;
}
.audio-btn {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 5px;
}
.audio-btn:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 5. Initialize Chat History and Audio Settings
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "bot", "content": "Hello! I'm your Rice Disease Q&A Assistant. Ask me anything about rice diseases — e.g., 'What causes rice blast?' or 'How to control tungro?'"}
    ]

if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = True

# Audio settings in sidebar
with st.sidebar:
    st.header("🔊 Audio Settings")
    st.session_state.audio_enabled = st.checkbox("Enable Text-to-Speech", value=True)
    st.info("When enabled, you can listen to bot responses by clicking the audio button.")

# -------------------------------
# 6. Display Chat History with Audio
# -------------------------------

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.history):
    role_class = "user" if msg["role"] == "user" else "bot"
    st.markdown(
        f"""
        <div class='msg {role_class}'>
            <div class='role'>{msg["role"].capitalize()}</div>
            <div>{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Add audio controls for bot messages
    if msg["role"] == "bot" and st.session_state.audio_enabled:
        # Generate audio for this message
        audio_data = text_to_speech(msg["content"])
        
        if audio_data:
            # Create audio player
            audio_html = f"""
            <div class='audio-controls'>
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# 7. Chat Input (Pinned Below)
# -------------------------------

question = st.chat_input("Type your question about rice diseases...")

if question:
    # Store user message
    st.session_state.history.append({"role": "user", "content": question})

    # Process the question
    with st.spinner("Analyzing your question..."):
        selected_context = None
        for keyword, section in sections.items():
            if keyword in question.lower():
                selected_context = section
                break

        time.sleep(1)  # small delay for realism

        if selected_context:
            result = qa_pipeline(question=question, context=selected_context)
            answer = result["answer"]
        else:
            answer = "I couldn't match your question to a specific disease. Try including the name, like 'blast', 'blight', or 'tungro'."

    # Add bot response to history
    st.session_state.history.append({"role": "bot", "content": answer})

    # Rerun to refresh chat
    st.rerun()
