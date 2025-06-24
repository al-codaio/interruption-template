from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from graph import graph

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


@app.post("/graph/invoke")
async def invoke_graph(request: InvokeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    initial_input = {"messages": [("human", request.input)]}
    result = await graph.ainvoke(initial_input, config=config)

    def get_new_ai_messages(messages):
        # Return all AI messages after the last HumanMessage
        last_human_idx = max(i for i, m in enumerate(messages) if getattr(m, "type", None) == "human")
        return [m.content for m in messages[last_human_idx+1:] if getattr(m, "type", None) == "ai"]

    async def event_stream():
        for content in get_new_ai_messages(result["messages"]):
            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000) 