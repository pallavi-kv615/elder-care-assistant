
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Find project folder
project_folder = Path(__file__).resolve().parent.parent

# Find documents folder
documents_folder = project_folder / "documents"

# Find PDF files
pdf_files = list(documents_folder.glob("*.pdf"))

if not pdf_files:

    print("No PDF files found.")

else:

    all_documents = []

    # Read each PDF
    for pdf_file in pdf_files:

        print("Reading:", pdf_file.name)

        reader = PdfReader(str(pdf_file))

        # Read each page separately
        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                document = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_number
                    }
                )

                all_documents.append(document)

    # Split pages while preserving metadata
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    print("\nTotal pages:", len(all_documents))
    print("Total chunks:", len(chunks))

    # Load embedding model
    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS index
    print("\nCreating FAISS search index...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Save index
    save_folder = project_folder / "faiss_index"

    vector_store.save_local(str(save_folder))

    print("\nFAISS index created successfully.")
    print("Saved to:", save_folder)

    # Test semantic search
    question = "Section 5 Application for maintenance: Who may apply for maintenance? senior citizen parent authorized person organization Tribunal suo motu"

    results = vector_store.similarity_search(
    question,
    k=6
    )
    
    print("\nTest question:", question)

    print("\nRelevant document sections:")

    for result in results:

        print("\n--------------------")

        print("Source:", result.metadata.get("source"))

        print("Page:", result.metadata.get("page"))

        print(result.page_content)
