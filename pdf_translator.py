import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import io
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI PDF Translator",
    page_icon="🌐",
    layout="wide"
)

# --------------------------------------------------
# Groq Client
# --------------------------------------------------

api_key=st.secrets["GROQ_API_KEY"]

client=Groq(api_key=api_key)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌐 AI PDF Translator")

st.write(
    "Upload a PDF document, select a target language, "
    "translate the document using AI, preview the result, "
    "and download the translated PDF."
)

# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your PDF document",
    type=["pdf"]
)

# --------------------------------------------------
# Language Selection
# --------------------------------------------------

languages = [
    "English",
    "Hindi",
    "Marathi",
    "Gujarati",
    "Tamil",
    "Telugu",
    "Bengali",
    "Kannada",
    "French",
    "German",
    "Spanish"
]

target_language = st.selectbox(
    "🌍 Select Target Language",
    languages
)

# --------------------------------------------------
# PDF to Text
# --------------------------------------------------

if uploaded_file is not None:

    try:

        reader = PdfReader(uploaded_file)

        extracted_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                extracted_text += text + "\n"

        # --------------------------------------------------
        # Validate Text
        # --------------------------------------------------

        if not extracted_text.strip():

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.stop()

        # --------------------------------------------------
        # Display Original Text
        # --------------------------------------------------

        st.subheader("📖 Original Document")

        st.text_area(
            "Extracted Text",
            extracted_text,
            height=300
        )

        # --------------------------------------------------
        # Translate Button
        # --------------------------------------------------

        if st.button("🔄 Translate Document"):

            with st.spinner(
                "🤖 Translating document using Groq AI..."
            ):

                prompt = f"""
Translate the following document into {target_language}.

Instructions:
- Preserve the original meaning.
- Preserve headings and paragraphs.
- Preserve lists where possible.
- Do not add explanations.
- Return only the translated document.

Document:

{extracted_text}
"""

                response = client.chat.completions.create(

                    model="llama-3.3-70b-versatile",

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.2
                )

                translated_text = (
                    response.choices[0]
                    .message.content
                )

            # --------------------------------------------------
            # Save Translation in Session
            # --------------------------------------------------

            st.session_state["translated_text"] = translated_text

            st.session_state["target_language"] = target_language

        # --------------------------------------------------
        # Display Translation
        # --------------------------------------------------

        if "translated_text" in st.session_state:

            translated_text = st.session_state[
                "translated_text"
            ]

            target_language = st.session_state[
                "target_language"
            ]

            st.subheader(
                f"🌍 Translated Document - {target_language}"
            )

            st.text_area(
                "Translated Text",
                translated_text,
                height=400
            )

            # --------------------------------------------------
            # Create PDF
            # --------------------------------------------------

            pdf_buffer = io.BytesIO()

            pdf = canvas.Canvas(
                pdf_buffer,
                pagesize=A4
            )

            width, height = A4

            pdf.setFont(
                "Helvetica-Bold",
                16
            )

            pdf.drawString(
                20 * mm,
                height - 20 * mm,
                f"Translated Document - {target_language}"
            )

            pdf.setFont(
                "Helvetica",
                11
            )

            y = height - 35 * mm

            # Split translated text into lines
            lines = translated_text.split("\n")

            for line in lines:

                # Wrap long lines
                words = line.split()
                current_line = ""

                for word in words:

                    test_line = (
                        current_line + " " + word
                    ).strip()

                    if pdf.stringWidth(
                        test_line,
                        "Helvetica",
                        11
                    ) < width - 40 * mm:

                        current_line = test_line

                    else:

                        pdf.drawString(
                            20 * mm,
                            y,
                            current_line
                        )

                        y -= 7 * mm

                        current_line = word

                        if y < 20 * mm:

                            pdf.showPage()

                            pdf.setFont(
                                "Helvetica",
                                11
                            )

                            y = height - 20 * mm

                if current_line:

                    pdf.drawString(
                        20 * mm,
                        y,
                        current_line
                    )

                    y -= 7 * mm

                # New paragraph spacing
                y -= 2 * mm

                if y < 20 * mm:

                    pdf.showPage()

                    pdf.setFont(
                        "Helvetica",
                        11
                    )

                    y = height - 20 * mm

            pdf.save()

            pdf_buffer.seek(0)

            # --------------------------------------------------
            # Download PDF
            # --------------------------------------------------

            st.download_button(

                label="⬇️ Download Translated PDF",

                data=pdf_buffer,

                file_name=(
                    f"translated_{target_language}.pdf"
                ),

                mime="application/pdf"
            )

    except Exception as e:

        st.error(
            f"❌ Error while processing PDF: {e}"
        )