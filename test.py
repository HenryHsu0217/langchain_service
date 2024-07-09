import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_community.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile

def fetch_and_process_pdf(file_id):
    # Create a Drive API client
    # In Cloud Run, this will use the default credentials automatically
    drive_service = build('drive', 'v3')

    # Fetch the file
    request = drive_service.files().get_media(fileId=file_id)
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while done is False:
        _, done = downloader.next_chunk()

    # Save to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    # Use PyPDFLoader to load the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    return documents

def main():
    file_id = '1PI1pAOriyWQPOpLvUJcgKpCHTrDe12c7'
    documents = fetch_and_process_pdf(file_id)
    
    # Print some information about the documents
    print(f"Number of documents: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"Document {i + 1}:")
        print(f"  Page Content (first 100 chars): {doc.page_content[:100]}...")
        print(f"  Metadata: {doc.metadata}")
        print()

if __name__ == "__main__":
    main()