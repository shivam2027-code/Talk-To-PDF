from flask import Flask ,render_template , jsonify , request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI ,ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import os
load_dotenv()

app = Flask(__name__)

embeddings = download_embedding()

index_name = "pdf-bot"
#embed each chunk and upsert the embedding into your pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriver = docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(model,prompt)
rag_chain = create_retrieval_chain(retriver,question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET",'POST'])
def chat():
    msg = request.form["msg"]
    input=msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("response: ",response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)
