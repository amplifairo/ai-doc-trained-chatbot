import streamlit as st
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
# import socket


openai_api_key = None
if 'openai' in st.secrets and 'api_key' in st.secrets.openai:
  openai_api_key = st.secrets.openai.api_key

if not openai_api_key:
  import constants
  openai_api_key = constants.APIKEY
  os.environ["OPENAI_API_KEY"] = openai_api_key

if 'langchain' in st.secrets and 'api_key' in st.secrets.langchain:
  os.environ["LANGCHAIN_TRACING_V2"] = "true"
  os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.langchain.endpoint
  os.environ["LANGCHAIN_API_KEY"] = st.secrets.langchain.api_key
  os.environ["LANGCHAIN_PROJECT"] = st.secrets.langchain.project_name

# Enable to save to disk & reuse the model (for repeated queries on the same data)

# CHARACTER_NAME = st.secrets.app.name

query = None

# Prompt
# PROMPT_TEMPLATE = """
#     Use the following pieces of context to answer the question at the end. 
#     You are role playing as {CHARACTER_NAME}. Always reply as a person. NEVER BREAK THE CHARACTER!
#     You have your own opinions and knowledge based on the information you have provided 
#     in the context. Never lie. If you dont know the answer then just say that you were not able to find that 
#     information in your database, don't try to make up an answer. You can use the chat history to answer some 
#     specific questions if required.
#     {context}
#     Question: {question}
# """

PROMPT_TEMPLATE = st.secrets.app.prompt

QA_CHAIN_PROMPT = PromptTemplate.from_template(PROMPT_TEMPLATE)

## getting the hostname by socket.gethostname() method
# hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
# ip_address = socket.gethostbyname(hostname)
# print(f"Hostname: {hostname}")
# print(f"IP Address: {ip_address}")

# if PERSIST and os.path.exists("persist"):
  # vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  # index = VectorStoreIndexWrapper(vectorstore=vectorstore)
# else:

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
persist_directory = 'db'
if not os.path.exists(persist_directory):
  text_loader_kwargs={'autodetect_encoding': True}
  loader = DirectoryLoader("data/", glob="**/*.txt", use_multithreading=True, loader_cls=TextLoader, silent_errors=True, loader_kwargs=text_loader_kwargs)
  raw_documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  documents = text_splitter.split_documents(raw_documents)
  # db = Chroma.from_documents(documents, embedding_function=OpenAIEmbeddings())
  vectordb = Chroma.from_documents(documents=documents, 
                                   embedding=embedding,
                                   persist_directory=persist_directory)
  # persiste the db to disk
  vectordb.persist()
  vectordb = None

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 1})

  # if PERSIST:
  #   # index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  #   index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_documents([documents])
  # else:
  #   # index = VectorstoreIndexCreator().from_loaders([loader])
  #   index = VectorstoreIndexCreator().from_documents(raw_documents)

@st.cache_resource
def init_chat_history():
  return ChatMessageHistory()

@st.cache_resource
def init_memory():
  return ConversationBufferMemory(
    memory_key='chat_history'
    , return_messages=True
    , output_key='answer'
  )

chat_history = init_chat_history()
history = []
memory = init_memory()

# model = "gpt-4-1106-preview"
model = st.secrets.openai.model
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model=model, temperature = 0.5),
  retriever=retriever,
  return_source_documents=True,
  combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
  # get_chat_history=lambda h :h,
  # memory=memory,
  chain_type="stuff"
)

def generate_response(query):
    history = chat_history.messages
    result = chain({
      "question": query['content']
      , "chat_history": history
      # , "CHARACTER_NAME": CHARACTER_NAME
      })
    print(result)

    return result['answer']

st.title(st.secrets.app.title) 
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": st.secrets.app.intro_message}]
    chat_history.add_ai_message(st.secrets.app.intro_message)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    usermsg = {"role": "user", "content": prompt}
    st.session_state.messages.append(usermsg)
    st.chat_message("user").write(prompt)
    chat_history.add_user_message(prompt)
    response = generate_response(usermsg)
    # msg = response.choices[0].message
    msg = {"role": "assistant", "content": response}
    print(msg['content'])
    st.session_state.messages.append(msg)
    # st.chat_message("assistant").write(msg)
    st.chat_message("assistant").write(msg['content'])