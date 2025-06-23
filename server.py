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
            yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name'], 'output': event['data']['output']})}\n\n"

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