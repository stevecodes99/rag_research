import os
from pprint import pprint
from typing import List
from typing_extensions import TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START


# Environment Setup
def setup_environment():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "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"


# Load Documents
def load_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    return docs_list


# Split Documents
def split_documents(docs_list):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


# Setup VectorStore
def setup_vectorstore(doc_splits):
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
    )
    return vectorstore


# Setup LLM and Prompts
def setup_llm_and_prompts(local_llm):
    llm = ChatOllama(model=local_llm, format="json", temperature=0)

    retrieval_prompt = PromptTemplate(
        template="""system You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        user
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n assistant
        """,
        input_variables=["question", "document"],
    )
    retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

    answer_prompt = PromptTemplate(
        template="""system You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise user
        Question: {question} 
        Context: {context} 
        Answer: assistant""",
        input_variables=["question", "document"],
    )
    answer_llm = ChatOllama(model=local_llm, temperature=0)
    rag_chain = answer_prompt | answer_llm | StrOutputParser()

    hallucination_prompt = PromptTemplate(
        template=""" system You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. user
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  assistant""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_prompt | llm | JsonOutputParser()

    answer_grade_prompt = PromptTemplate(
        template="""system You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        user Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} assistant""",
        input_variables=["generation", "question"],
    )
    answer_grader = answer_grade_prompt | llm | JsonOutputParser()

    router_prompt = PromptTemplate(
        template="""system You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explanation. Question to route: {question} assistant""",
        input_variables=["question"],
    )
    question_router = router_prompt | llm | JsonOutputParser()

    return retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_router


# State Management Classes
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]


# Workflow Nodes
def retrieve(state, retriever):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state, rag_chain):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state, retrieval_grader):
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state, web_search_tool):
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def route_question(state, question_router):
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source["datasource"] == "web_search":
        return "websearch"
    elif source["datasource"] == "vectorstore":
        return "vectorstore"


def decide_to_generate(state):
    web_search = state["web_search"]
    if web_search == "Yes":
        return "websearch"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    if grade == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


# Build and Compile Workflow
def build_workflow(retriever, rag_chain, retrieval_grader, hallucination_grader, answer_grader, question_router, web_search_tool):
    workflow = StateGraph(GraphState)

    workflow.add_node("websearch", lambda state: web_search(state, web_search_tool))
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, retrieval_grader))
    workflow.add_node("generate", lambda state: generate(state, rag_chain))

    workflow.add_conditional_edges(
        START,
        lambda state: route_question(state, question_router),
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    workflow.add_node("grade_generation", lambda state: grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader))
    workflow.add_edge("generate", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        lambda state: state["grade"],
        {
            "useful": END,
            "not supported": "retrieve",
            "not useful": "retrieve",
        },
    )

    return workflow


# Main
if __name__ == "__main__":
    setup_environment()

    # URLs to load documents from
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",]
    print('-'*40)
    print('Loading Documents')
    docs_list = load_documents(urls)
    print('Documents Loaded')
    print('-'*40)
    print('Splitting Documents')
    doc_splits = split_documents(docs_list)
    print('Documents Split')
    print('-'*40)
    print('Setting up VectorStore')
    vectorstore = setup_vectorstore(doc_splits)
    print('VectorStore setup')
    print('-'*40)
    print('Setting up Graph')

    # Setup LLM and Prompts
    local_llm = "llama3"
    retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_router = setup_llm_and_prompts(local_llm)

    # Setup Web Search Tool
    web_search_tool = TavilySearchResults(num_results=3)

    # Build Workflow
    
    workflow = build_workflow(vectorstore.as_retriever(), rag_chain, retrieval_grader, hallucination_grader, answer_grader, question_router, web_search_tool)
    print('Graph setup')
    print('-'*40)

    # Example State
    # example_state = GraphState(question="What is prompt Engineering?", generation="", web_search="", documents=[])

    print('WorkFlow started')

    app = workflow.compile()

    # Test

    inputs = {"question": "What are the types of agent memory?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])

    print('-'*40)
    print('Workflow completed')

    # # Run Workflow
    # result = workflow.invoke(example_state)
    # pprint(result)
