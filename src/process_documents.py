
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. Find project and documents folders
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
documents_folder = project_folder / "documents"
save_folder = project_folder / "faiss_index"

pdf_files = sorted(documents_folder.glob("*.pdf"))


# --------------------------------------------------
# 2. Check for PDF documents
# --------------------------------------------------

if not pdf_files:
    print("No PDF files found in the documents folder.")

else:
    all_documents = []

    # --------------------------------------------------
    # 3. Read every PDF page
    # --------------------------------------------------

    for pdf_file in pdf_files:
        print(f"\nReading: {pdf_file.name}")

        try:
            reader = PdfReader(str(pdf_file))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    document = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file.name,
                            "page": page_number,
                        },
                    )

                    all_documents.append(document)

        except Exception as error:
            print(f"Unable to read {pdf_file.name}: {error}")

    if not all_documents:
        print(
            "No readable text was extracted from the PDFs. "
            "Scanned PDFs may require OCR."
        )

    else:
        # --------------------------------------------------
        # 4. Split pages into overlapping text chunks
        # --------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(all_documents)

        print(f"\nTotal readable pages: {len(all_documents)}")
        print(f"Total text chunks: {len(chunks)}")

        # --------------------------------------------------
        # 5. Load the embedding model
        # --------------------------------------------------

        print("\nLoading embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # --------------------------------------------------
        # 6. Create and save the FAISS index
        # --------------------------------------------------

        print("\nCreating FAISS search index...")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        vector_store.save_local(str(save_folder))

        print("\nFAISS index created successfully.")
        print(f"Saved to: {save_folder}")

        # --------------------------------------------------
        # 7. Test Section 5 retrieval
        # --------------------------------------------------

        question = (
            "Section 5 application for maintenance: "
            "Who may apply for maintenance? "
            "Senior citizen, parent, authorised person, "
            "organisation, and Tribunal taking cognizance suo motu."
        )

        print("\nTest question:")
        print(question)

        # Retrieve six chunks instead of four
        results = vector_store.similarity_search(
            question,
            k=6,
        )

        print("\nRelevant document sections:")

        if not results:
            print("No relevant sections were retrieved.")

        else:
            for number, result in enumerate(results, start=1):
                print("\n" + "=" * 60)
                print(f"Result {number}")
                print(f"Source: {result.metadata.get('source', 'Unknown')}")
                print(f"Page: {result.metadata.get('page', 'Unknown')}")
                print("-" * 60)
                print(result.page_content)

        print("\nRetrieval test completed.")
