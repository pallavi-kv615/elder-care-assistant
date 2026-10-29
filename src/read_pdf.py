
from pypdf import PdfReader
from pathlib import Path

# Find the project folder
project_folder = Path(__file__).resolve().parent.parent

# Find the documents folder
documents_folder = project_folder / "documents"

# Find PDF files
pdf_files = list(documents_folder.glob("*.pdf"))

if not pdf_files:
    print("No PDF file found in documents folder.")

else:
    pdf_path = pdf_files[0]

    print("PDF found:", pdf_path.name)

    reader = PdfReader(str(pdf_path))

    print("Total pages:", len(reader.pages))

    # Read the first page
    first_page = reader.pages[0]
    text = first_page.extract_text()

    print("\nFirst page content:\n")
    print(text)
