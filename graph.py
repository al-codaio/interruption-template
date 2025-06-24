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

llm = ChatOpenAI(model="gpt-4o-2024-05-13")

MISSING_FIELDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Your role is to ask the user for the information you are missing. Based on the user's request and the missing fields, generate a single, user-friendly question to ask the user. For example, if 'user_email' is missing, ask 'What is your email address?'. If multiple fields are missing, ask for them all in one question."),
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

# Define response schemas for StructuredOutputParser
response_schemas = [
    ResponseSchema(name="user_email", description="The user's email address."),
    ResponseSchema(name="document_id", description="The ID of the document to be analyzed."),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

def orchestrator(state: AgentState) -> dict:
    log(f"[orchestrator] user_email={state.user_email}, document_id={state.document_id}, analysis_result={state.analysis_result}")
    if not state.user_email or not state.document_id:
        log("[orchestrator] Routing to start_analysis")
        return {"next_node": "start_analysis"}
    elif not state.analysis_result:
        log("[orchestrator] Routing to performs_step_2")
        return {"next_node": "performs_step_2"}
    else:
        log("[orchestrator] Routing to end")
        return {"next_node": "end"}

def validation_and_processing_node(state: AgentState) -> dict:
    log(f"[validation_and_processing_node] Entered with state: user_email={state.user_email}, document_id={state.document_id}, analysis_result={state.analysis_result}, next_node={state.next_node}")
    last_message = state.messages[-1]

    # If the last message is from the user, try to process it for missing info
    if isinstance(last_message, HumanMessage):
        prompt = (
            f"You are an expert at extracting information.\n"
            f"Extract 'user_email' and 'document_id' from the user's message:\n\n"
            f"'{last_message.content}'\n\n"
            f"Return a JSON object with the extracted values. If a value is not found, do not include its key."
        )
        response = llm.invoke(prompt)
        try:
            extracted_data = parser.parse(response.content)
        except Exception as e:
            log(f"[validation_and_processing_node] Failed to parse extracted data: {response.content} | Error: {e}")
            extracted_data = {}
        log(f"[validation_and_processing_node] Extracted data from user: {extracted_data}")
        if extracted_data:
            return extracted_data

    # Validation logic (runs after any extracted fields have been merged)
    next_node = state.next_node
    if not next_node:
        log("[validation_and_processing_node] No next_node set, re-running orchestrator.")
        return {}

    if next_node == "end":
        log("[validation_and_processing_node] Reached end node.")
        return {"messages": [AIMessage(content="Analysis complete. Thank you!")]}

    required_fields = NODE_REQUIREMENTS.get(next_node, [])
    missing_fields = [field for field in required_fields if not getattr(state, field, None)]

    if missing_fields:
        log(f"[validation_and_processing_node] Missing fields for node '{next_node}': {missing_fields}")
        # Ask the user for the missing information
        prompt = MISSING_FIELDS_PROMPT.format(
            messages=state.messages,
            missing_fields=", ".join(missing_fields),
        )
        response = llm.invoke(prompt)
        # Add the AI's question to the message list and interrupt
        state.messages.append(response)
        log(f"[validation_and_processing_node] Interrupting to ask user: {response.content}")
        raise GraphInterrupt()

    # If all fields are present, proceed
    log(f"[validation_and_processing_node] All required fields present for node '{next_node}'. Proceeding.")
    return {}

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
    log(f"[start_analysis] user_email={state.user_email}, document_id={state.document_id}")
    email = state.user_email
    doc_id = state.document_id
    result = f"Analysis of document '{doc_id}' for user '{email}' is complete. The result is 42."
    log(f"[start_analysis] Result: {result}")
    return {"analysis_result": result, "messages": [AIMessage(content=result)]}

def performs_step_2(state: AgentState) -> dict:
    log(f"[performs_step_2] analysis_result={state.analysis_result}")
    analysis_result = state.analysis_result
    result = f"Step 2 processed the analysis result: '{analysis_result}'. This is the final step."
    log(f"[performs_step_2] Result: {result}")
    return {"messages": [AIMessage(content=result)]}

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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory) 