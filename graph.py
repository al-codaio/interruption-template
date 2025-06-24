import os
import json
from typing import List, Optional, TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.errors import GraphInterrupt
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from dotenv import load_dotenv

load_dotenv()

def log(msg):
    print(f"[GRAPH LOG] {msg}")

# Post-processing function to extract only the user-friendly question
def extract_question(text):
    import re
    # Remove all code blocks (```...```), aggressively
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Return the last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""

llm = ChatOpenAI(model="gpt-4o-2024-05-13")

MISSING_FIELDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Ask the user for the missing fields. ONLY return the question, no preamble, no code blocks, no extra text."),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "I am missing the following fields: {missing_fields}. Please generate a question to ask the user for this information."),
    ]
)

NODE_REQUIREMENTS = {
    "start_analysis": ["user_email", "document_id"],
    "performs_step_2": ["analysis_result"],
}

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    user_email: Optional[str] = Field(None, description="The user's email address.")
    document_id: Optional[str] = Field(None, description="The ID of the document to be analyzed.")
    analysis_result: Optional[str] = Field(None, description="The result of the analysis from start_analysis.")
    next_node: Optional[str] = Field(None, description="The next node to execute.")
    step_2_done: Optional[bool] = False

# Define response schemas for StructuredOutputParser
response_schemas = [
    ResponseSchema(name="user_email", description="The user's email address."),
    ResponseSchema(name="document_id", description="The ID of the document to be analyzed."),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

def extract_fields(state, missing_fields):
    conversation = "\n".join(f"{type(m).__name__}: {getattr(m, 'content', m)}" for m in state.messages)
    prompt = (
        f"You are an expert at extracting information.\n"
        f"Here is the conversation so far:\n{conversation}\n\n"
        f"Extract the following fields if present: {', '.join(missing_fields)}. "
        f"Return a JSON object with the extracted values. If a value is not found, do not include its key."
    )
    response = llm.invoke(prompt)
    try:
        return parser.parse(response.content)
    except Exception:
        return {}

def ask_for_missing_fields(state, missing_fields):
    prompt = MISSING_FIELDS_PROMPT.format(
        messages=state.messages,
        missing_fields=", ".join(missing_fields),
    )
    response = llm.invoke(prompt)
    question = extract_question(response.content)
    state.messages.append(AIMessage(content=question))
    raise GraphInterrupt()

def orchestrator(state: AgentState) -> dict:
    if not state.user_email or not state.document_id:
        return {"next_node": "start_analysis"}
    if not state.analysis_result:
        return {"next_node": "start_analysis"}
    if not getattr(state, "step_2_done", False):
        return {"next_node": "performs_step_2"}
    return {"next_node": "end"}

def validation_and_processing_node(state: AgentState) -> dict:
    next_node = state.next_node
    if not next_node:
        return {"messages": state.messages}
    if next_node == "end":
        return {"messages": [AIMessage(content="Analysis complete. Thank you!")]}
    required_fields = NODE_REQUIREMENTS.get(next_node, [])
    missing_fields = [f for f in required_fields if not getattr(state, f, None)]
    if missing_fields:
        extracted = extract_fields(state, missing_fields)
        for k, v in extracted.items():
            setattr(state, k, v)
        still_missing = [f for f in required_fields if not getattr(state, f, None)]
        if extracted:
            return extracted
        if still_missing:
            ask_for_missing_fields(state, still_missing)
    return {"messages": state.messages}

def route_after_validation(state: AgentState) -> str:
    log(f"[route_after_validation] next_node={state.next_node}")
    # If validation decided we are done, just end.
    if not state.next_node or state.next_node == "end":
        log("[route_after_validation] Routing to end")
        return "end"
    
    # Check if we have the required fields for the next node. If not, re-run orchestrator.
    required_fields = NODE_REQUIREMENTS.get(state.next_node, [])
    if any(not getattr(state, field, None) for field in required_fields):
        log(f"[route_after_validation] Missing required fields for node '{state.next_node}', routing to orchestrator")
        return "orchestrator"
        
    log(f"[route_after_validation] Routing to {state.next_node}")
    return state.next_node

def start_analysis(state: AgentState) -> dict:
    processing_msg = AIMessage(content="Processing analysis...")
    result = f"Analysis of document '{state.document_id}' for user '{state.user_email}' is complete. "
    return {
        "analysis_result": result,
        "messages": [processing_msg, AIMessage(content=result)]
    }

def performs_step_2(state: AgentState) -> dict:
    processing_msg = AIMessage(content="Running step 2...")
    result = f"Step 2 processed the analysis result: '{state.analysis_result}'. This is the final step."
    return {
        "messages": [processing_msg, AIMessage(content=result)],
        "step_2_done": True
    }

# Define the graph
builder = StateGraph(AgentState)

builder.add_node("orchestrator", orchestrator)
builder.add_node("validation_and_processing", validation_and_processing_node)
builder.add_node("start_analysis", start_analysis)
builder.add_node("performs_step_2", performs_step_2)

builder.add_edge(START, "orchestrator")
builder.add_edge("orchestrator", "validation_and_processing")

builder.add_conditional_edges(
    "validation_and_processing",
    route_after_validation,
    {
        "orchestrator": "orchestrator",
        "start_analysis": "start_analysis",
        "performs_step_2": "performs_step_2",
        "end": END,
    },
)

builder.add_edge("start_analysis", "orchestrator")
builder.add_edge("performs_step_2", "orchestrator")

# Uncomment the next two lines to serve the app at index.html
# memory = MemorySaver()
# graph = builder.compile(checkpointer=memory) 
graph = builder.compile() 