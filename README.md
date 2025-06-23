# LangGraph Interruption Example

This project demonstrates how to use LangGraph to build an AI application that can dynamically check for required information, interrupt the process to ask the user for missing details, and then resume from where it left off.

This example is designed to be a template for creating more complex agents that require robust user interaction and state management.

## Features

- **Dynamic Interruption**: The graph can be paused at any point to ask the user for required information.
- **Stateful Resumption**: The graph maintains its state and can resume correctly after an interruption.
- **Centralized Validation**: A single validation node checks for prerequisites before executing a node.
- **LLM-Generated Prompts**: The language model generates user-friendly prompts for missing information.
- **Chat Interface**: A simple web interface is provided to interact with the application.
- **LangGraph Studio Ready**: The project is configured to be used with LangGraph Studio for development and debugging.

## How it Works

The core idea is to use a validation node (`validate_state_and_interrupt`) that checks if all required fields are present in the `AgentState` before proceeding to the next step in the graph.

1.  An `orchestrator` node determines the next action (e.g., `start_analysis`).
2.  It sets `next_node_after_validation` in the state and routes to the `validate_state_and_interrupt` node.
3.  The `validate_state_and_interrupt` node looks up the required fields for the target node in a `NODE_REQUIREMENTS` dictionary.
4.  If fields are missing, it uses an LLM to generate a prompt for the user, and then interrupts the graph's execution.
5.  If all fields are present, it routes the graph to the intended node (e.g., `start_analysis`).
6.  When the user provides the missing information, the `process_user_input` node updates the state.
7.  The graph then loops back to the `validate_state_and_interrupt` node to re-check the state and continue the process.

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management.
- An [OpenAI API Key](https://platform.openai.com/api-keys) for the language model.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/interruption-template.git
    cd interruption-template
    ```

2.  **Install dependencies:**
    ```bash
    poetry install
    ```

3.  **Set up environment variables:**

    Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Open the `.env` file and add your OpenAI API key:
    ```
    OPENAI_API_KEY=your-api-key-here
    ```

### Running the Application

You can run the application in two ways: through the provided web interface or using LangGraph Studio.

#### 1. Web Interface

To run the web server and interact with the chatbot:

```bash
poetry run uvicorn server:app --host 0.0.0.0 --port 8000
```

Then, open your browser and navigate to `http://127.0.0.1:8000`.

#### 2. LangGraph Studio

To use LangGraph Studio for development:

```bash
poetry run langgraph dev
```

This will launch the studio, and you can interact with your graph, view its state, and debug its execution.

## Project Structure

- `graph.py`: Contains the definition of the LangGraph, including the state, nodes, and edges.
- `server.py`: A FastAPI server that exposes the graph as an API and serves the web interface.
- `public/index.html`: The frontend for the chatbot.
- `pyproject.toml`: Defines the project dependencies.
- `langgraph.json`: Configuration file for LangGraph Studio.
- `planning.md`: The initial planning document for the project.
- `README.md`: This file. 