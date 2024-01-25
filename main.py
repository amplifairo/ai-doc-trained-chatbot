from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.retrievers import MultiQueryRetriever

# from langchain import hub

# #cachex
# import hashlib
# from gptcache import Cache
# from gptcache.adapter.api import init_similar_cache
# from langchain.cache import GPTCache
# from langchain.globals import set_llm_cache

# cache
from langchain.cache import SQLiteCache
import langchain

import os
import streamlit as st
from streamlit_pills import pills
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

# cachex
# def get_hashed_name(name):
#     return hashlib.sha256(name.encode()).hexdigest()

# def init_gptcache(cache_obj: Cache, llm: str):
#     hashed_llm = get_hashed_name(llm)
#     init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")

# set_llm_cache(GPTCache(init_gptcache))

# setup cache
langchain.llm_cache = SQLiteCache(database_path="./db/sqlite/growcentric.db")


loader = TextLoader('./data/growcentric_ro.txt')
# text_loader_kwargs={'autodetect_encoding': True}
# loader = DirectoryLoader("data/", glob="**/*.txt"
    # , use_multithreading=True
    # , loader_cls=TextLoader
    # , silent_errors=True
    # , loader_kwargs=text_loader_kwargs)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()

vector = FAISS.from_documents(documents, embeddings)

# add docs from website
webloader = WebBaseLoader("https://growcentric.ro/")
webloader.requests_kwargs = {'verify':False}
web_raw_documents = webloader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
web_documents = text_splitter.split_documents(web_raw_documents)
if len(web_documents) > 0:
  vector.add_documents(web_documents)

prompt_template = ChatPromptTemplate.from_template("""You are Tim, a top sales support specialist specializing in offering support to website visitors and converting them into leads as part of the GrowCentric team. 
Your ultimate goal is to get a firm order from the user with the following details:
- Name and surname
- Phone number
- Email address
- Company name
- Job title
Answer the following question based only on the provided context that represents information about a romanian company GrowCentric.:

<context>
{context}
</context>

Question: {input}""")
model = "gpt-4-1106-preview"
llm = ChatOpenAI(model=model)

document_chain = create_stuff_documents_chain(llm, prompt_template)

retriever = vector.as_retriever()
advanced_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
retrieval_chain = create_retrieval_chain(advanced_retriever, document_chain)

def generate_response(query):
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

# generate_response("what is the status of the vessel?")

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

selected = pills(
    "Cele mai frecvente intrebari",
    [
        "Cand a fost infiintata compania?",
        "Ce servicii ofera compania GrowCentric pentru piata din Romania?",
        "Care este durata minima a contractului?",
    ],
    clearable=True,
    index=None,
)

def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state["messages"].append(
        message
    )  # Add response to message history

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": 'Salut! Eu sunt Tim, cu ce te pot ajuta?'}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# To avoid duplicated display of answered pill questions each rerun
if selected and selected not in st.session_state.get(
    "displayed_pill_questions", set()
):
    st.session_state.setdefault("displayed_pill_questions", set()).add(selected)
    add_to_message_history("user", selected)
    st.chat_message("user").write(selected)
    response = generate_response(selected)
    add_to_message_history("assistant", response)
    st.chat_message("assistant").write(response)

if prompt := st.chat_input():
    add_to_message_history("user", prompt)
    st.chat_message("user").write(prompt)
    response = generate_response(prompt)
    add_to_message_history("assistant", response)
    st.chat_message("assistant").write(response)

