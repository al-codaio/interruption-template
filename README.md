# LangGraph Interruption For Missing Fields Template

<img src="https://github.com/user-attachments/assets/0eb2bc14-c8a0-4809-8785-c313ed419668" height="600" />

This template shows how an AI application built on LangGraph can dynamically check for required information, interrupt the process to ask the human for missing details, and then resume from where it left off. The required fields in this example are `email` and `document ID` (as shown in the screenshot above). Without this required information, the graph won't continue. 

## 🚀 Template Features

- **Dynamic Interruption**: The graph can be paused at any point to ask the user for required information.
- **Stateful Resumption**: The graph maintains its state and can resume correctly after an interruption.
- **Centralized Validation**: A single validation node checks for prerequisites before executing a node.
- **Chat Interface**: A simple web interface showing the chatbot in [index.html](index.html)

## How it Works

The core idea is to use a validation node (`validate_state_and_interrupt`) that checks if all required fields are present before proceeding to the next step in the graph. The architecture lpan used to generate the code is in [planning.md](planning.md)

1.  An `orchestrator` node determines the next action (e.g., `start_analysis`).
2.  It sets `next_node_after_validation` in the state and routes to the `validate_state_and_interrupt` node.
3.  The `validate_state_and_interrupt` node looks up the required fields for the target node in a `NODE_REQUIREMENTS` dictionary.
4.  If fields are missing, the graph is interruped and asks the user for the required info
5.  If all fields are present, it routes the graph to the intended node (e.g., `start_analysis`).
6.  When the user provides the missing information, the `process_user_input` node updates the state.
7.  The graph then loops back to the `validate_state_and_interrupt` node to re-check the state and continue the process.

## ⚡ How To Use

**1. Clone the repo**
    ```bash
    git clone https://github.com/your-username/interruption-template.git
    cd interruption-template
    ```

**2. Setup and run virtual Python environment**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install requirements**
    ```bash
    poetry install
    ```

**4. Add your OpenAI API key**
Get your OpenAI API key [here](https://platform.openai.com/api-keys) and your LangSmith API key (follow instructions [here](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key)). Before running these commands in terminal, add the keys to the `OPEN_API_KEY` and `LANGCHAIN_API_KEY` variables below:
```bash
touch .env
echo 'OPENAI_API_KEY="PASTE-YOUR-OPENAI-API-KEY-HERE"' > .env
echo 'LANGCHAIN_API_KEY="PASTE-YOUR-LANGCHAIN-API-KEY-HERE"' >> .env
echo 'LANGCHAIN_TRACING_V2="true"' >> .env
echo 'LANGCHAIN_PROJECT="Customer Support Assistant Template"' >> .env
```

**5. Run the applicaiton in your browser**
```bash
poetry run uvicorn server:app --host 0.0.0.0 --port 8000
```

Then, open your browser and navigate to `http://127.0.0.1:8000`.

**6. Run the application in LangGraph Studio**

You need to comment out these two lines at the bottom of `graph.py`:
```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```
And then uncomment the last line in `graph.py`
```python
graph = builder.compile()
```
Then run:
```bash
poetry run langgraph dev
```

This will launch the studio, but I'm still working on getting the studio to work since the first human message input requires a JSON object which isn't user-friendly. This is what the graph looks like in Studio showing all the conditional edges:
<img src="https://github.com/user-attachments/assets/7c3e8715-86a1-484d-a8af-85ee5eaeeb08" width="700" />
