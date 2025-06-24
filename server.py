from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import json
import uuid

from graph import graph, AgentState

app = FastAPI()

# Mount a directory for static files (like index.html)
app.mount("/static", StaticFiles(directory="public"), name="static")


class InvokeRequest(BaseModel):
    input: str
    thread_id: str


@app.get("/")
async def get_index():
    with open("public/index.html", "r") as f:
        return HTMLResponse(f.read())


async def stream_events(thread_id: str, input_message: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # The initial input to the graph is a list of messages.
    # For the first message, it's just the user's input.
    # For subsequent messages, the checkpointer will load the history.
    initial_input = {"messages": [("human", input_message)]}

    async for event in graph.astream_events(initial_input, config=config, version="v1"):
        event_type = event["event"]
        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content})}\n\n"
        elif event_type == "on_tool_start":
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
        elif event_type == "on_tool_end":
            output = event['data']['output']
            # Try to extract the last AI message content if present
            message_content = (
                output.get('messages', [{}])[-1].get('content')
                if isinstance(output, dict) and 'messages' in output and output['messages']
                and isinstance(output['messages'][-1], dict)
                else output if isinstance(output, str) else None
            )
            if message_content:
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': message_content})}\n\n"

    # After the event loop, send the last AI message in the state (if any)
    final_state = graph.invoke(initial_input, config=config)
    if 'messages' in final_state and final_state['messages']:
        ai_message_content = None
        for msg in reversed(final_state['messages']):
            # If using dicts
            if isinstance(msg, dict) and msg.get('type', '').lower() == 'ai':
                ai_message_content = msg.get('content')
                break
            # If using AIMessage objects
            elif hasattr(msg, 'type') and getattr(msg, 'type', '').lower() == 'ai':
                ai_message_content = getattr(msg, 'content', None)
                break
        if ai_message_content:
            yield f"data: {json.dumps({'type': 'final', 'output': ai_message_content})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/graph/invoke")
async def invoke_graph(request: InvokeRequest):
    thread_id = request.thread_id
    return StreamingResponse(
        stream_events(thread_id, request.input),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 