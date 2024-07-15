from langchain_google_cloud_sql_pg import PostgresEngine, PostgresLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
SERVEICE_ACCOUNT_KEY=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SCOPES = ['https://www.googleapis.com/auth/drive']
PROJECT_ID="langchain-service"
REGION = "asia-east1"  
INSTANCE = "google-drive-vector"  
DATABASE = "Google-drive-files"
TABLE_NAME = "document_test"
def create_db_tool():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    def connect_database(): 
        engine = PostgresEngine.from_instance(
        project_id=PROJECT_ID,
        region=REGION,
        instance=INSTANCE,
        database=DATABASE,
        user="Henry",
        password="2021ping"
        )
        return engine
    engine=connect_database()
    loader = PostgresLoader.create_sync(
        engine,
        table_name=TABLE_NAME
    )
    docs=loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    db=FAISS.from_documents(splits, embeddings)
    retriever=db.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "search_pdf",
        "Searches informations from PDF files",
    )
    tools = [tool]
    return tools
create_db_tool()