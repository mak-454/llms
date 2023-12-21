from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

system_template = """Answer the user question using the provided context and chat history. 
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

"""

if __name__ == '__main__':

    loader = PyPDFLoader("./selfrag.pdf")
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    docs = text_splitter.split_documents(pages)

    embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    tech_store = Chroma.from_documents(
        docs, embeddings, collection_name="technology"
    )
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        #max_tokens=1024,
    )


    chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ])


    tech_qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = chat_memory,
        retriever = tech_store.as_retriever(),
        verbose = False,
        combine_docs_chain_kwargs={'prompt': prompt},
        get_chat_history = lambda h : h
    )

    """
    tech_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=tech_store.as_retriever(),
        verbose=False, condense_question_prompt=prompt,memory = chat_memory
    )
    """

    while True:
        query = input("Question> ")
        result = tech_qa({"question": query})
        print(result['answer'])




