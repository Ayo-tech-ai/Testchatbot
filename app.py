import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import whisper
import time
import os
import base64
import requests

# -------------------------------
# 1. Load Models with Error Handling
# -------------------------------

@st.cache_resource
def load_qa_model():
    try:
        return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    except Exception as e:
        st.error(f"Error loading QA model: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.warning(f"Whisper model loading issue: {e}. Voice transcription will be disabled.")
        return None

qa_pipeline = load_qa_model()
whisper_model = load_whisper_model()

# Check if whisper is properly loaded
whisper_available = whisper_model is not None

# -------------------------------
# 2. Knowledge Base
# -------------------------------

sections = {
    "bacterial leaf blight": "Bacterial Leaf Blight is caused by the bacterium Xanthomonas oryzae pv. oryzae. It affects leaves, leading to yellowing and wilting. Management includes using resistant varieties and proper field sanitation.",
    "bacterial leaf streak": "Bacterial Leaf Streak is caused by Xanthomonas oryzae pv. oryzicola. It causes narrow, water-soaked streaks on leaves. Control methods include using certified seeds and avoiding field flooding.",
    "bakanae": "Bakanae Disease, also known as 'Foolish Seedling', is caused by the fungus Fusarium fujikuroi and leads to elongated, thin seedlings. Seed treatment with fungicides is recommended.",
    "brown spot": "Brown Spot is caused by Bipolaris oryzae. It appears as brown circular spots on leaves, affecting photosynthesis. Improve soil fertility and use resistant varieties.",
    "grassy stunt": "Rice Grassy Stunt Virus (RGSV) is spread by the brown planthopper and causes stunted growth. Control planthoppers with insecticides and use resistant varieties.",
    "narrow brown spot": "Narrow Brown Spot is caused by the fungus Cercospora janseana, leading to narrow brown lesions on leaves. Fungicide application can help control it.",
    "ragged stunt": "Rice Ragged Stunt Virus (RRSV), spread by brown planthoppers, causes ragged leaves and stunted plants. Vector control is essential.",
    "rice blast": "Rice Blast is caused by the fungus Magnaporthe oryzae, producing diamond-shaped lesions on leaves. Use resistant varieties and avoid excessive nitrogen.",
    "false smut": "Rice False Smut, caused by Ustilaginoidea virens, produces greenish smut balls on grains. Remove infected plants and use fungicides.",
    "sheath blight": "Sheath Blight is caused by Rhizoctonia solani and forms lesions on the leaf sheath. Proper spacing and fungicides help control it.",
    "sheath rot": "Sheath Rot is caused by Sarocladium oryzae, resulting in rotting of the uppermost leaf sheath. Avoid dense planting and use balanced fertilizers.",
    "stem rot": "Stem Rot is caused by Sclerotium oryzae, which blackens the base of the stem. Practice field sanitation and crop rotation.",
    "tungro": "Rice Tungro Virus is a dual infection (Rice Tungro Bacilliform and Spherical Viruses), causing yellow-orange leaf discoloration. Control green leafhopper vectors."
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
.warning {
    background-color: #FFF3CD;
    border: 1px solid #FFEAA7;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Rice Disease Q&A Assistant")

# Show warning if whisper is not available
if not whisper_available:
    st.markdown("""
    <div class='warning'>
    ⚠️ <b>Voice transcription is currently disabled</b><br>
    FFmpeg is required for voice processing. You can still use text input for questions.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# 4. Chat History
# -------------------------------

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "bot", "content": "Hello 👋🏽! You can type your question about rice diseases — e.g., 'What causes rice blast?' or 'How to treat bacterial leaf blight?'."}
    ]

# Display chat history
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
# 5. Voice Recording (Only if whisper is available)
# -------------------------------

if whisper_available:
    st.markdown("🎙️ **Record your voice question below:**")
    audio_file = st.audio_input("Record your question", label_visibility="collapsed")
else:
    st.info("💡 Voice input is disabled. Please use text input below.")
    audio_file = None

question = None

if audio_file and whisper_available:
    st.audio(audio_file, format="audio/wav")
    
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("Transcribing your voice..."):
        try:
            result = whisper_model.transcribe(tmp_path)
            question = result["text"].strip()
            if question and len(question) > 5:  # Only show if we got meaningful text
                st.markdown(f"🗣️ **You said:** {question}")
            else:
                st.warning("No clear speech detected in the recording. Please try again or use text input.")
                question = None
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            st.info("Please use text input instead.")
            question = None
    
    # Clean up temporary file
    try:
        os.unlink(tmp_path)
    except:
        pass

# -------------------------------
# 6. Text Input (Always Available)
# -------------------------------

text_input = st.chat_input("💬 Type your question about rice diseases here...")
if text_input:
    question = text_input

# -------------------------------
# 7. Output Mode Selection
# -------------------------------

output_mode = st.radio(
    "Choose how to receive the response:",
    ["Text only", "Text + Voice"],
    horizontal=True
)

# -------------------------------
# 8. Generate Response
# -------------------------------

if question and qa_pipeline:
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Analyzing your question..."):
        selected_context = None
        matched_disease = None
        
        # Find the best matching disease
        question_lower = question.lower()
        for keyword, section in sections.items():
            if keyword in question_lower:
                selected_context = section
                matched_disease = keyword
                break

        time.sleep(1)  # simulate processing time

        if selected_context:
            try:
                result = qa_pipeline(question=question, context=selected_context)
                answer = result["answer"]
                # Add some context about the disease
                answer = f"**About {matched_disease.title()}:**\n\n{answer}"
            except Exception as e:
                answer = f"I found information about {matched_disease}, but encountered an error processing it. Here's what I know:\n\n{selected_context}"
        else:
            # If no specific disease found, provide general help
            answer = """I couldn't find specific information about that disease in my knowledge base. 

Here are some rice diseases I can help with:
- Bacterial Leaf Blight
- Rice Blast  
- Tungro
- Brown Spot
- Sheath Blight
- False Smut
- Stem Rot
- Grassy Stunt

Try asking about one of these, or be more specific in your question!"""

    st.session_state.history.append({"role": "bot", "content": answer})

    # Voice Output (if selected)
    if output_mode == "Text + Voice" and answer:
        with st.spinner("Generating voice reply..."):
            try:
                tts = gTTS(text=answer, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
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

elif question and not qa_pipeline:
    st.error("The question-answering system is not available. Please try again later.")

# -------------------------------
# 9. Disease Quick Reference
# -------------------------------

with st.expander("📚 Quick Reference: Common Rice Diseases"):
    cols = st.columns(2)
    diseases = list(sections.keys())
    mid = len(diseases) // 2
    
    with cols[0]:
        for disease in diseases[:mid]:
            st.write(f"• {disease.title()}")
    
    with cols[1]:
        for disease in diseases[mid:]:
            st.write(f"• {disease.title()}")

# -------------------------------
# 10. Clear Chat Button
# -------------------------------

if st.button("🗑️ Clear Chat History"):
    st.session_state.history = [
        {"role": "bot", "content": "Hello 👋🏽! You can type your question about rice diseases — e.g., 'What causes rice blast?' or 'How to treat bacterial leaf blight?'."}
    ]
    st.rerun()
