from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "you area medical assistent for qestion-answering tasks"
    "use the following pieces of retrived context to answer"
    "the question if you donot know the answer . say that you"
    "donot know. use three seneteces maximaum and keep the answer concise"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)
