from langgraph.graph import MessagesState, StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# 1. Cargar vector store persistido
vstore = Chroma(
    persist_directory="vector_store",
    embedding_function=OpenAIEmbeddings(),
)
retriever = vstore.as_retriever(k=3)

#2. tool para el agente
retriever_tool = create_retriever_tool(
    retriever,
    name="retrieve_tickets",
    description="Retrieve similar past tickets with description and resolution",
)

# modelo
chat_model = init_chat_model("openai:gpt-4.1", temperature=0)

# principal node
def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    system_message = (
        "You are a help-desk assistant. "
        "If resolving the user's issue requires consulting past tickets, "
        "call the 'retrieve_tickets' tool with {'query': user_question}. "
        "Otherwise, answer the question directly."
    )

    response = (
        chat_model
        .bind_tools([retriever_tool]).invoke(state["messages"], 
            system_message=system_message)
    )
    return {"messages": [response]}

#
GRADE_PROMPT = (
    "You are an evaluator determining if a retrieved ticket is relevant to a support request.\n"
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

# Define the model for grading documents (only a binary score yes o no)
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

#node to grade the retrieved documents
# This node will determine whether the retrieved documents are relevant to the question.
def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]: #literal means that the return value can only be one of the specified strings
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    ticket_content = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=ticket_content)
    response = (
        chat_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"

#nodo 3
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = chat_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}

#nodo 4
GENERATE_PROMPT = (
    "You are a help-desk assistant. Using ONLY the retrieved ticket context,  "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "provide a priority level (1=High, 2=Medium, 3=Low) and a concise resolution.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    messages = state["messages"]
    question = messages[0].content
    context = messages[-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = chat_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# Create the workflow graph
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision
    tools_condition,
    {
        #edge:node
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents
)
workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

# Compile
agent = workflow.compile()

if __name__ == "__main__":
    while True:
        request = input("🔹 Describe your issue (enter to exit): ").strip()
        if not request:
            break
        for update in agent.stream({"messages": [{"role": "user", "content": request}]}):
            for node, result in update.items():
                print(f"--- {node} ---")
                result["messages"][-1].pretty_print()
                print()
