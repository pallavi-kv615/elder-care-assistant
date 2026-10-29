
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

    /* Chat input */
    div[data-testid="stChatInput"] textarea {
        background-color: #ffffff !important;
        color: #26352e !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #66736b !important;
        opacity: 1 !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #315f48 !important;
        color: #ffffff !important;
        border: 1px solid #315f48 !important;
        border-radius: 8px;
        min-height: 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.5rem 1rem;
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
        and welfare information in simple English. Ask questions about
        the Maintenance and Welfare of Parents and Senior Citizens Act,
        2007, and Atal Pension Yojana (APY).
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### How can we help you?")

st.write(
    "Ask a question below. You can continue asking questions "
    "without refreshing the page."
)

st.caption(
    "Example questions: Who can apply for maintenance? "
    "What is Atal Pension Yojana? What is the appeal process?"
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
# 6. Conversation state and New Chat button
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("### Your conversation")

with header_right:
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# 7. Display previous messages and their sources
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View document sources"):
                for number, source in enumerate(
                    message["sources"], start=1
                ):
                    st.markdown(f"**Source {number}**")
                    st.caption(
                        f"Document: {source['name']} | "
                        f"Page: {source['page']}"
                    )
                    st.write(source["text"])
                    st.divider()


# --------------------------------------------------
# 8. Question input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about senior citizen rights or APY..."
)


# --------------------------------------------------
# 9. Search documents and generate an answer
# --------------------------------------------------

if question:

    # Save and display the new user question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    answer = ""
    sources = []

    with st.chat_message("assistant"):

        if not API_KEY:
            answer = (
                "The Gemini API key was not found. "
                "Please check your application secrets or .env settings."
            )
            st.error(answer)

        elif not INDEX_FOLDER.exists():
            answer = (
                "The document search index was not found. "
                "Please check the faiss_index folder."
            )
            st.error(answer)

        else:
            try:
                with st.spinner(
                    "Searching your documents and preparing an answer..."
                ):
                    vector_store = load_knowledge_base(
                        str(INDEX_FOLDER)
                    )

                    client = get_gemini_client(API_KEY)

                    question_lower = question.lower()
                    search_question = question

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

                    # Improve retrieval for APY questions.
                    apy_question = (
                        "apy" in question_lower
                        or "atal pension" in question_lower
                        or "pension yojana" in question_lower
                    )

                    if apy_question:
                        search_question = (
                            "Atal Pension Yojana (APY), pension scheme, "
                            "scheme overview, eligibility, contributions, "
                            "pension benefits, subscriber, spouse, nominee, "
                            "account opening, and scheme features."
                        )

                    # Retrieve relevant document sections.
                    documents = vector_store.similarity_search(
                        search_question,
                        k=6,
                    )

                    if not documents:
                        answer = (
                            "I couldn't find relevant information in "
                            "the available documents. Please try "
                            "rephrasing your question."
                        )
                        st.warning(answer)

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

                        # Keep the last three exchanges for follow-up context.
                        previous_messages = (
                            st.session_state.messages[:-1][-6:]
                        )

                        history_text = "\n".join(
                            f"{item['role'].upper()}: {item['content']}"
                            for item in previous_messages
                        )

                        prompt = (
                            "You are an Elder-Care Assistant that explains "
                            "Indian senior citizen law and pension information "
                            "in simple English.\n\n"

                            "IMPORTANT INSTRUCTIONS:\n"
                            "1. Answer the current question directly.\n"
                            "2. Use only the supplied document sections "
                            "for factual and legal claims.\n"
                            "3. Do not invent facts, benefits, or legal rules.\n"
                            "4. If the documents do not contain the answer, "
                            "say so clearly.\n"
                            "5. Explain legal terms in simple English.\n"
                            "6. For Section 5 questions about who may apply, "
                            "include all three clauses when supported by "
                            "the supplied text: (a) the senior citizen or "
                            "parent; (b) an authorised person or organisation "
                            "if they are incapable; and (c) the Tribunal "
                            "taking cognizance suo motu.\n"
                            "7. Use previous messages only to understand "
                            "follow-up questions. Ground factual answers "
                            "in the supplied documents.\n"
                            "8. This is general information, not legal advice.\n"
                            "9. When asked what APY is, explain that APY "
                            "stands for Atal Pension Yojana and briefly "
                            "describe it as a pension scheme. Start with "
                            "this direct explanation, then describe relevant "
                            "features using the supplied documents. Do not "
                            "say its meaning is missing from the documents "
                            "when the scheme name is clear from the question "
                            "or supplied context.\n"
                            "10. For APY questions, focus on the document "
                            "sections about APY. Include only benefits, "
                            "eligibility rules, contributions, or other "
                            "details that are supported by those sections.\n\n"

                            f"RECENT CONVERSATION:\n"
                            f"{history_text or 'No previous questions.'}\n\n"

                            f"DOCUMENT SECTIONS:\n{context}\n\n"

                            f"CURRENT QUESTION:\n{question}\n\n"

                            "Write a clear, helpful answer. Use numbered "
                            "points when appropriate. Do not claim that "
                            "information is missing if it appears in "
                            "the supplied document sections."
                        )

                        response = client.models.generate_content(
                            model="gemini-3.5-flash-lite",
                            contents=prompt,
                        )

                        answer = response.text or (
                            "Sorry, I couldn't prepare an answer. "
                            "Please try asking again."
                        )

                        # Save the sources alongside this answer.
                        sources = [
                            {
                                "name": Path(
                                    document.metadata.get(
                                        "source", "Unknown"
                                    )
                                ).name,
                                "page": document.metadata.get(
                                    "page", "Unknown"
                                ),
                                "text": document.page_content.strip(),
                            }
                            for document in documents
                        ]

                        st.markdown(answer)

                        # Show sources for the current answer.
                        with st.expander("View document sources"):
                            for number, source in enumerate(
                                sources, start=1
                            ):
                                st.markdown(f"**Source {number}**")
                                st.caption(
                                    f"Document: {source['name']} | "
                                    f"Page: {source['page']}"
                                )
                                st.write(source["text"])
                                st.divider()

            except Exception as error:
                answer = (
                    "Sorry, I couldn't generate your answer. "
                    "Please check your connection and try again."
                )
                sources = []
                st.error(answer)
                st.caption(f"Technical details: {error}")

    # Save the assistant response so it remains visible on reruns.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


# --------------------------------------------------
# 10. Important information
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
