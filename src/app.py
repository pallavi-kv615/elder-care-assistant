
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. Application setup
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
INDEX_FOLDER = PROJECT_FOLDER / "faiss_index"

st.set_page_config(
    page_title="Elder-Care Assistant",
    page_icon="E",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# 2. Senior-friendly design
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #fbf8f1;
        color: #26352e;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #244c3b !important;
    }

    h1 {
        font-size: 2.5rem !important;
        line-height: 1.3 !important;
    }

    h2 {
        font-size: 1.8rem !important;
    }

    h3 {
        font-size: 1.4rem !important;
    }

    p, li, .stMarkdown {
        font-size: 1.12rem;
        line-height: 1.75;
        color: #26352e;
    }

    .welcome-card {
        background-color: #edf3e9;
        border-left: 6px solid #47765b;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0 1.5rem 0;
        color: #26352e;
        font-size: 1.12rem;
        line-height: 1.8;
    }

    .info-card {
        background-color: #ffffff;
        border: 1px solid #ded9cb;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        color: #26352e;
        font-size: 1.12rem;
        line-height: 1.8;
    }

    /* Readable question box */
    div[data-testid="stTextArea"] label {
        color: #244c3b !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #26352e !important;
        -webkit-text-fill-color: #26352e !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        min-height: 6rem;
        border: 1px solid #b8c8bb !important;
        border-radius: 8px !important;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #66736b !important;
        -webkit-text-fill-color: #66736b !important;
        opacity: 1 !important;
    }

    /* Readable buttons */
    div.stButton > button {
        background-color: #315f48 !important;
        color: #ffffff !important;
        border: 1px solid #315f48 !important;
        border-radius: 8px;
        min-height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }

    div.stButton > button:hover {
        background-color: #244c3b !important;
        border-color: #244c3b !important;
        color: #ffffff !important;
    }

    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #ded9cb;
        border-radius: 8px;
    }

    div[data-testid="stAlert"] {
        font-size: 1.05rem;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. Header and introduction
# --------------------------------------------------

st.title("Elder-Care Assistant")

st.subheader("Senior Citizen Rights and Welfare Support")

st.markdown(
    """
    <div class="welcome-card">
        <strong>Welcome!</strong><br>
        This assistant helps you understand senior citizen rights
        in simple English. You can ask questions about the
        Maintenance and Welfare of Parents and Senior Citizens Act, 2007.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### How can we help you?")

st.write(
    "Type your question in the box below, then select "
    "**Get Answer**."
)

st.caption(
    "Example questions: Who can apply for maintenance? "
    "What is the appeal process? What does the Act say about tribunals?"
)

st.divider()


# --------------------------------------------------
# 4. Load document search index
# --------------------------------------------------

@st.cache_resource
def load_knowledge_base(index_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


# --------------------------------------------------
# 5. Connect to Gemini
# --------------------------------------------------

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# --------------------------------------------------
# 6. Question input
# --------------------------------------------------

question = st.text_area(
    "Your question",
    placeholder="For example: Who can apply for maintenance?",
    height=110,
)

ask_button = st.button(
    "Get Answer",
    use_container_width=True,
)


# --------------------------------------------------
# 7. Search documents and generate answer
# --------------------------------------------------

if ask_button:

    if not question.strip():
        st.warning("Please type a question before continuing.")

    elif not API_KEY:
        st.error(
            "The Gemini API key was not found. "
            "Please check your .env file."
        )

    elif not INDEX_FOLDER.exists():
        st.error(
            "The document search index was not found. "
            "Please check your faiss_index folder."
        )

    else:
        try:
            with st.spinner(
                "Searching the Act and preparing your answer..."
            ):

                vector_store = load_knowledge_base(
                    str(INDEX_FOLDER)
                )

                client = get_gemini_client(API_KEY)

                search_question = question.strip()
                question_lower = search_question.lower()

                # Improve retrieval for Section 5 questions.
                section_five_question = (
                    "who can apply" in question_lower
                    or "who may apply" in question_lower
                    or "application for maintenance" in question_lower
                    or (
                        "section 5" in question_lower
                        and (
                            "maintenance" in question_lower
                            or "apply" in question_lower
                        )
                    )
                )

                if section_five_question:
                    search_question = (
                        "Section 5 Application for maintenance. "
                        "Subsection (1), clauses (a), (b), and (c). "
                        "Application may be made by a senior citizen "
                        "or parent; if incapable, by another person "
                        "or organisation authorised by them; or the "
                        "Tribunal may take cognizance suo motu. "
                        "Definition of organisation in Section 5."
                    )

                # Retrieve six chunks instead of four.
                documents = vector_store.similarity_search(
                    search_question,
                    k=6,
                )

                if not documents:
                    st.warning(
                        "No relevant information was found in the "
                        "available documents. Please try rephrasing "
                        "your question."
                    )

                else:
                    context = "\n\n".join(
                        (
                            f"DOCUMENT: "
                            f"{document.metadata.get('source', 'Unknown')}\n"
                            f"PAGE: "
                            f"{document.metadata.get('page', 'Unknown')}\n"
                            f"TEXT:\n{document.page_content}"
                        )
                        for document in documents
                    )

                    # More precise answer instructions.
                    prompt = (
                        "You are an Elder-Care Assistant that explains "
                        "Indian senior citizen law in simple English.\n\n"

                        "IMPORTANT INSTRUCTIONS:\n"
                        "1. Answer the user's exact question directly.\n"
                        "2. Use only the provided document sections "
                        "for legal claims.\n"
                        "3. Read all supplied sections carefully.\n"
                        "4. If the question asks who may apply under "
                        "Section 5, include all three clauses:\n"
                        "(a) the senior citizen or parent;\n"
                        "(b) if they are incapable, another person "
                        "or organisation authorised by them;\n"
                        "(c) the Tribunal may take cognizance suo motu.\n"
                        "5. Explain legal terms in simple English.\n"
                        "6. If the supplied text contains the answer, "
                        "do not claim that the information is missing.\n"
                        "7. Do not invent legal rules or facts.\n"
                        "8. If the supplied text genuinely does not "
                        "contain the answer, say so clearly.\n"
                        "9. This is general information, not legal advice.\n\n"

                        f"DOCUMENT SECTIONS:\n{context}\n\n"
                        f"USER QUESTION:\n{question.strip()}\n\n"

                        "Write a clear answer with numbered points "
                        "when appropriate. Include every relevant "
                        "clause supported by the supplied text."
                    )

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                    )

                    st.divider()
                    st.markdown("## Answer")

                    if response.text:
                        st.markdown(
                            '<div class="info-card">',
                            unsafe_allow_html=True,
                        )

                        st.markdown(response.text)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    else:
                        st.warning(
                            "No answer was returned. Please try again."
                        )

                    # Display the exact passages used.
                     
                    # Display readable document sources.
                    with st.expander("View document sources"):
                        for number, document in enumerate(documents, start=1):
                            source_name = Path(
                                document.metadata.get("source", "Unknown")
                            ).name
                            page_number = document.metadata.get("page", "Unknown")
                            source_text = document.page_content.strip()

                            st.markdown(f"**Source {number}**")
                            st.caption(
                                f"Document: {source_name} | Page: {page_number}"
                            )
                            st.write(source_text)
                            st.divider()

                           
        except Exception as error:
            st.error(
                "We could not generate your answer. "
                "Please check your connection and try again."
            )

            st.caption(f"Technical details: {error}")


# --------------------------------------------------
# 8. Important information
# --------------------------------------------------

st.divider()

st.markdown("### Important information")

st.write(
    "This assistant provides general information from its "
    "available documents. It is not a substitute for advice "
    "from a qualified lawyer or an authorised government office."
)

st.caption(
    "Elder-Care Assistant | Senior Citizen Rights and Welfare Support"
)
