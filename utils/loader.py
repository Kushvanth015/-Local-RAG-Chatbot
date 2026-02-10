from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os


def load_uploaded_files(uploaded_files):
    """
    Loads multiple PDF/TXT files and returns LangChain Documents.
    Adds clean metadata:
    - source_file
    - page (for PDFs)
    """
    docs = []

    for file in uploaded_files:
        suffix = os.path.splitext(file.name)[1].lower()

        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Load based on file type
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()

            # Add metadata properly
            for d in file_docs:
                d.metadata["source_file"] = file.name

                # PyPDFLoader stores page number in metadata["page"]
                # Ensure it exists
                if "page" not in d.metadata:
                    d.metadata["page"] = "N/A"

        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            file_docs = loader.load()

            for d in file_docs:
                d.metadata["source_file"] = file.name
                d.metadata["page"] = "TXT"

        docs.extend(file_docs)

        # Cleanup temp file
        os.remove(tmp_path)

    return docs
