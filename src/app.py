import textwrap
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Elder-Care Assistant",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
INDEX_FOLDER = PROJECT_FOLDER / "faiss_index"


def render_html(content):
    """Render HTML without displaying it as code."""
    html = textwrap.dedent(content).strip()
    st.html(html)


# --------------------------------------------------
# 2. PROFESSIONAL PURPLE AND WHITE DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #faf8ff 0%,
            #f3edff 55%,
            #eee6ff 100%
        );
        color: #29145f;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #29145f !important;
    }

    p, li {
        color: #392765;
    }

    .topbar {
        background: linear-gradient(
            100deg,
            #241052,
            #422080
        );
        color: white;
        padding: 20px 28px;
        border-radius: 16px;
        margin-bottom: 35px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        box-shadow: 0 5px 18px rgba(54, 28, 112, 0.18);
    }

    .brand {
        color: white !important;
        font-size: 25px;
        font-weight: 750;
    }

    .brand-subtitle {
        color: #e4d9ff !important;
        font-size: 13px;
        margin-top: 5px;
    }

    .nav-label {
        color: #f1eaff !important;
        font-size: 14px;
        font-weight: 600;
    }

    .hero {
        padding: 25px 10px 35px 10px;
    }

    .hero-tag {
        display: inline-block;
        background: #e9ddff;
        color: #5426a8 !important;
        border-radius: 30px;
        padding: 9px 17px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 18px;
    }

    .hero-title {
        color: #241052 !important;
        font-size: clamp(38px, 5vw, 60px);
        font-weight: 800;
        line-height: 1.12;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #6336b4 !important;
        font-size: 27px;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .hero-description {
        color: #49396d !important;
        font-size: 18px;
        line-height: 1.8;
        max-width: 750px;
    }

    .section-heading {
        color: #29145f !important;
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 18px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 17px;
    }

    .feature-card {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e5d8ff;
        border-radius: 17px;
        padding: 23px 20px;
        min-height: 145px;
        box-shadow: 0 5px 15px rgba(76, 43, 139, 0.08);
        transition: 0.2s ease;
    }

    .feature-icon {
        color: #6735c3 !important;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .feature-title {
        color: #30166c !important;
        font-size: 17px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .feature-description {
        color: #655486 !important;
        font-size: 14px;
        line-height: 1.6;
    }

    .question-panel {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dfd0ff;
        border-radius: 20px;
        padding: 25px;
        margin-top: 28px;
        margin-bottom: 15px;
        box-shadow: 0 5px 20px rgba(76, 43, 139, 0.09);
    }

    .question-heading {
        color: #29145f !important;
        font-size: 27px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .question-description {
        color: #655486 !important;
        font-size: 16px;
        line-height: 1.7;
    }

    div[data-testid="stTextArea"] label {
        color: #30166c !important;
        font-weight: 700 !important;
        font-size: 17px !important;
    }

    div[data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        color: #30166c !important;
        -webkit-text-fill-color: #30166c !important;
        border: 1px solid #cbb7f5 !important;
        border-radius: 12px !important;
        font-size: 17px !important;
        line-height: 1.6 !important;
        min-height: 100px;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #8878a8 !important;
        -webkit-text-fill-color: #8878a8 !important;
        opacity: 1 !important;
    }

    div.stButton > button {
        background: linear-gradient(
            90deg,
            #5424b7,
            #8051d5
        ) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 48px;
        font-size: 17px !important;
        font-weight: 700 !important;
        padding: 10px 20px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #3d1a8b !important;
        color: white !important;
        border: none !important;
    }

    .answer-heading {
        color: #30166c !important;
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 15px;
    }

    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid #e3d6ff;
        border-radius: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    .trust-strip {
        background: #e9e0ff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: #432184 !important;
        font-size: 15px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 25px;
        line-height: 2;
    }

    .footer {
        background: linear-gradient(
            100deg,
            #241052,
            #422080
        );
        color: #eee6ff !important;
        text-align: center;
        padding: 23px;
        border-radius: 13px;
        margin-top: 20px;
        font-size: 14px;
        line-height: 1.8;
    }

    footer {
        visibility: hidden;
    }

    @media (max-width: 700px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .topbar {
            padding: 18px;
        }

        .hero-title {
            font-size: 38px;
        }

        .hero-subtitle {
            font-size: 22px;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }

        .question-panel {
            padding: 17px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 3. PROFESSIONAL HOMEPAGE HEADER
# --------------------------------------------------

render_html(
    """
    <div class="topbar">

        <div>
            <div class="brand">♡ Elder-Care Assistant</div>

            <div class="brand-subtitle">
                Senior Citizen Rights and Welfare Support
            </div>
        </div>

        <div class="nav-label">
            Home &nbsp; | &nbsp; About &nbsp; | &nbsp;
            Resources &nbsp; | &nbsp; Help
        </div>

    </div>
    """
)


# --------------------------------------------------
# 4. HERO SECTION AND FEATURE CARDS
# --------------------------------------------------

render_html(
    """
    <div class="hero">

        <div class="hero-tag">
            TRUSTED &nbsp; • &nbsp;
            INFORMATIVE &nbsp; • &nbsp;
            EASY TO USE
        </div>

        <div class="hero-title">
            Elder-Care Assistant
        </div>

        <div class="hero-subtitle">
            Ask. Learn. Empower.
        </div>

        <div class="hero-description">
            Your AI-powered guide for understanding senior
            citizen rights, government schemes and welfare
            benefits — all explained in simple English.
        </div>

    </div>

    <div class="section-heading">
        Explore Information
    </div>

    <div class="feature-grid">

        <div class="feature-card">
            <div class="feature-icon">⚖</div>

            <div class="feature-title">
                Maintenance Act, 2007
            </div>

            <div class="feature-description">
                Understand maintenance rights and legal
                protection for parents and senior citizens.
            </div>
        </div>

        <div class="feature-card">
            <div class="feature-icon">▤</div>

            <div class="feature-title">
                Pension Schemes
            </div>

            <div class="feature-description">
                Learn about NPS, APY and pension-related
                information.
            </div>
        </div>

        <div class="feature-card">
            <div class="feature-icon">♡</div>

            <div class="feature-title">
                Senior Citizen Welfare
            </div>

            <div class="feature-description">
                Explore healthcare, support and welfare
                information.
            </div>
        </div>

        <div class="feature-card">
            <div class="feature-icon">?</div>

            <div class="feature-title">
                FAQs and More
            </div>

            <div class="feature-description">
                Ask questions and get simple answers from
                available documents.
            </div>
        </div>

    </div>

    <div class="question-panel">

        <div class="question-heading">
            How can we help you?
        </div>

        <div class="question-description">
            Enter your question below and get an
            easy-to-understand answer.
        </div>

    </div>
    """
)


# --------------------------------------------------
# 5. LOAD DOCUMENT SEARCH INDEX
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
# 6. CONNECT TO GEMINI
# --------------------------------------------------

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# --------------------------------------------------
# 7. DOCUMENT RETRIEVAL
# --------------------------------------------------

def retrieve_documents(
    vector_store,
    search_queries,
    allowed_sources=None,
    k=4,
):

    retrieved_documents = {}

    for search_query in search_queries:

        if allowed_sources:

            for source in allowed_sources:

                results = vector_store.similarity_search_with_score(
                    search_query,
                    k=k,
                    filter={"source": source},
                    fetch_k=30,
                )

        else:

            results = vector_store.similarity_search_with_score(
                search_query,
                k=k,
            )

        for document, score in results:

            key = (
                document.metadata.get("source"),
                document.metadata.get("page"),
                document.page_content.strip(),
            )

            score = float(score)

            if (
                key not in retrieved_documents
                or score < retrieved_documents[key]["score"]
            ):

                retrieved_documents[key] = {
                    "document": document,
                    "score": score,
                }

    ranked_results = sorted(
        retrieved_documents.values(),
        key=lambda item: item["score"],
    )

    return [
        item["document"]
        for item in ranked_results
    ]

            
# --------------------------------------------------
# 8. QUESTION INPUT
# --------------------------------------------------

question = st.text_area(
    "Your question",
    placeholder=(
        "For example: Who can apply for maintenance? "
        "What is NPS? Who can join APY?"
    ),
    height=110,
    key="user_question",
)

ask_button = st.button(
    "Get Answer  →",
    use_container_width=True,
)


# --------------------------------------------------
# 9. SEARCH DOCUMENTS AND GENERATE ANSWER
# --------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning(
            "Please type your question before continuing."
        )

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
                "Searching documents and preparing your answer..."
            ):

                vector_store = load_knowledge_base(
                    str(INDEX_FOLDER)
                )

                client = get_gemini_client(API_KEY)

                original_question = question.strip()
                question_lower = original_question.lower()

                # ------------------------------------------
                # Detect question topics
                # ------------------------------------------

                section_five_question = any(
                    phrase in question_lower
                    for phrase in [
                        "section 5",
                        "section five",
                        "who can apply",
                        "who may apply",
                        "application for maintenance",
                        "under section 5",
                    ]
                )

                section_four_question = any(
                    phrase in question_lower
                    for phrase in [
                        "section 4",
                        "section four",
                        "maintenance obligation",
                    ]
                )

                apy_question = (
                    "apy" in question_lower
                    or "atal pension" in question_lower
                    or "pension yojana" in question_lower
                )

                nps_question = (
                    "nps" in question_lower
                    or "national pension system" in question_lower
                    or "national pension scheme" in question_lower
                )

                healthcare_question = any(
                    keyword in question_lower
                    for keyword in [
                        "healthcare",
                        "health care",
                        "medical",
                        "hospital",
                        "treatment",
                        "ayushman",
                        "pm-jay",
                        "pmjay",
                        "geriatric",
                        "health insurance",
                        "senior citizen health",
                    ]
                )

                # ------------------------------------------
                # Prepare retrieval queries
                # ------------------------------------------

                allowed_sources = None

                search_queries = [original_question]

                if section_five_question:

                    allowed_sources = [
                        "Senior_Citizen_Act_2007.pdf"
                    ]

                    search_queries = [
                        (
                            "Section 5 Application for maintenance. "
                            "Maintenance and Welfare of Parents and "
                            "Senior Citizens Act 2007. "
                            "Who may apply to the Maintenance Tribunal. "
                            "Application by senior citizen or parent. "
                            "Application by an authorised person or "
                            "registered organisation when the senior "
                            "citizen or parent is incapable. "
                            "Tribunal taking cognizance suo motu."
                        ),
                        (
                            "Section 5 subsection 1 clauses a b c. "
                            "Application for maintenance under Section 4. "
                            "Senior citizen parent authorised person "
                            "organisation Maintenance Tribunal."
                        ),
                    ]

                elif section_four_question:

                    allowed_sources = [
                        "Senior_Citizen_Act_2007.pdf"
                    ]

                    search_queries = [
                        (
                            "Section 4 Maintenance of parents and "
                            "senior citizens. Eligibility. "
                            "Parent or grandparent claiming maintenance "
                            "from children. Childless senior citizen "
                            "claiming maintenance from relative."
                        ),
                        original_question,
                    ]

                elif apy_question:

                    allowed_sources = ["APY_FAQs.pdf"]

                    search_queries = [
                        (
                            "Atal Pension Yojana APY official FAQ. "
                            "Eligibility, age requirements, contributions, "
                            "pension benefits, spouse and nominee benefits, "
                            "exit and withdrawal rules."
                        ),
                        original_question,
                    ]

                elif nps_question:

                    allowed_sources = [
                        "NPS-All citizen Model FAQs.pdf"
                    ]

                    search_queries = [
                        (
                            "National Pension System NPS official FAQ. "
                            "Eligibility, All Citizen Model, Tier I, "
                            "Tier II, PRAN, contributions, pension, "
                            "exit, withdrawal and annuity."
                        ),
                        original_question,
                    ]

                elif healthcare_question:

                    allowed_sources = [
                        "Senior_Citizen_Healthcare_FAQs.txt",
                        "Senior_Citizen_Act_2007.pdf",
                    ]

                    search_queries = [
                        (
                            "Senior citizen healthcare and medical support "
                            "in India. Ayushman Bharat PM-JAY benefits for "
                            "senior citizens aged 70 years and above, "
                            "eligibility, Aadhaar, e-KYC, Ayushman card, "
                            "coverage and enrolment."
                        ),
                        (
                            "Maintenance and Welfare of Parents and "
                            "Senior Citizens Act 2007 Section 20. "
                            "Medical support, government hospitals, "
                            "separate queues and geriatric care."
                        ),
                        original_question,
                    ]

                # ------------------------------------------
                # Retrieve relevant document sections
                # ------------------------------------------

                documents = retrieve_documents(
                    vector_store=vector_store,
                    search_queries=search_queries,
                    allowed_sources=allowed_sources,
                    k=8,
                )

                documents = documents[:6]

                if not documents:

                    st.warning(
                        "No relevant information was found in the "
                        "available documents. Please try rephrasing "
                        "your question."
                    )

                else:

                    # --------------------------------------
                    # Prepare document context
                    # --------------------------------------

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

                    # --------------------------------------
                    # Answer generation instructions
                    # --------------------------------------

                    prompt = (

                        "You are an Elder-Care Assistant that explains "
                        "Indian senior citizen rights, welfare and pension "
                        "information in simple English.\n\n"

                        "IMPORTANT INSTRUCTIONS:\n"

                        "1. Answer the user's exact question directly.\n"

                        "2. Use only the provided document sections "
                        "for factual and legal claims.\n"

                        "3. Read all supplied document sections carefully.\n"

                        "4. Clearly distinguish Section 4 and Section 5:\n"

                        "Section 4 explains the maintenance obligation.\n"

                        "Section 5 explains the procedure for making "
                        "an application for maintenance.\n"

                        "5. For a question about who may apply under "
                        "Section 5, include these three possibilities "
                        "only when supported by the supplied text:\n"

                        "(a) The senior citizen or parent.\n"

                        "(b) If they are incapable, another person "
                        "or organisation authorised by them.\n"

                        "(c) The Tribunal may take cognizance suo motu.\n"

                        "6. Explain legal terms in simple English.\n"

                        "7. Do not invent legal rules, eligibility "
                        "criteria, contribution amounts or benefits.\n"

                        "8. If the supplied documents do not contain "
                        "the answer, say so clearly.\n"

                        "9. For APY questions, explain Atal Pension "
                        "Yojana using only the supplied APY document.\n"

                        "10. For NPS questions, explain National Pension "
                        "System using only the supplied NPS document.\n"

                        "11. For healthcare questions, use the supplied "
                        "healthcare FAQ and Section 20 information.\n"

                        "12. This is general information, not legal advice.\n\n"

                        f"DOCUMENT SECTIONS:\n{context}\n\n"

                        f"USER QUESTION:\n{original_question}\n\n"

                        "Write a clear and helpful answer in simple English. "
                        "Use numbered points when appropriate. "
                        "Include relevant facts supported by the documents."
                    )

                    # --------------------------------------
                    # Generate answer using Gemini
                    # --------------------------------------

                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                    )

                    st.markdown("---")

                    st.markdown(
                        '<div class="answer-heading">Answer</div>',
                        unsafe_allow_html=True,
                    )

                    if response.text:

                        with st.container(border=True):
                            st.markdown(response.text)

                    else:

                        st.warning(
                            "No answer was returned. Please try again."
                        )

                    # --------------------------------------
                    # Display document sources
                    # --------------------------------------

                    with st.expander("View document sources"):

                        displayed_sources = set()
                        source_number = 0

                        for document in documents:

                            source_name = Path(
                                document.metadata.get(
                                    "source",
                                    "Unknown",
                                )
                            ).name

                            page_number = document.metadata.get(
                                "page",
                                "Unknown",
                            )

                            source_key = (
                                source_name,
                                page_number,
                            )

                            if source_key in displayed_sources:
                                continue

                            displayed_sources.add(source_key)
                            source_number += 1

                            st.markdown(
                                f"**Source {source_number}**"
                            )

                            st.caption(
                                f"Document: {source_name} | "
                                f"Page: {page_number}"
                            )

                            st.write(
                                document.page_content.strip()
                            )

                            st.divider()

        except Exception as error:

            st.error(
                "We could not generate your answer. "
                "Please check your connection and try again."
            )

            st.caption(
                f"Technical details: {error}"
            )


# --------------------------------------------------
# 10. TRUST INFORMATION AND FOOTER
# --------------------------------------------------

render_html(
    """
    <div class="trust-strip">
        ✓ Reliable document-based information
        &nbsp;&nbsp; | &nbsp;&nbsp;
        Quick and simple answers
        &nbsp;&nbsp; | &nbsp;&nbsp;
        Supporting senior citizens
    </div>
    """
)

st.markdown("### Important information")

st.write(
    "This assistant provides general information from its "
    "available documents. It is not a substitute for advice "
    "from a qualified lawyer or an authorised government office."
)

render_html(
    """
    <div class="footer">
        Elder-Care Assistant
        <br>
        Senior Citizen Rights and Welfare Support
        <br><br>
        Together for a better tomorrow.
    </div>
    """
)