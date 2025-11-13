# YojnaPath - Multi-Stage Conversational Agent
<p align="center">
  <img src="assets/Yojanaपथ.png"/>
</p>


A clean and simple conversational AI agent built with LangGraph that guides users through government scheme information using multi-stage conversation flow with LLM structured output for intent detection.

## Demo Video

<video src="https://github.com/user-attachments/assets/034d1564-abcf-45dd-a801-310f800f4112" controls width="600"></video>


## Features

-  **Multi-Stage Conversation Flow**: Uses `stage_config.json` to define conversation stages and transitions
-  **Intent Detection**: LLM structured output determines next conversation stage
-  **Dynamic Stage Routing**: Efficiently moves between stages based on user intent
-  **Grok LLM Integration**: Uses Grok API for conversational responses
-  **LangGraph Implementation**: Clean graph-based conversation management

## Architecture

### Core Components

1. **Stage Configuration** (`stage_config.json`): Defines conversation stages and possible transitions
2. **Graph Builder** (`langgraph_app/graph_builder.py`): Creates LangGraph with stage management
3. **Grok LLM Integration**: Custom implementation for Grok API calls with structured output
4. **Conversation State**: Tracks current stage, messages, and user input

### Stage Flow

The conversation flows through predefined stages:
- `start_greet` → Initial greeting and route detection
- `scheme_doubt_solving` → Answer scheme-related questions
- `gather_info` → Collect user profile information
- `preference` → Understand scheme preferences
- `recommend_scheme` → Suggest relevant schemes
- `kb_tool_call` → Provide application assistance
- `farewell` → End conversation

## Setup

### 1. Clone and Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Grok API

Create a `.env` file in the project root:

```env
GROK_API_KEY=your_grok_api_key_here
```

### 3. Run the Application

```bash
# Run demo conversation
python -m langgraph_app.app

# Run interactive mode
python -m langgraph_app.app --interactive
```

## Usage

### Demo Mode
```bash
python -m langgraph_app.app
```
Runs a predefined conversation flow to demonstrate the multi-stage system.

### Interactive Mode
```bash
python -m langgraph_app.app --interactive
```
Allows real-time conversation with the agent.

## Configuration

### Stage Configuration (`stage_config.json`)

Each stage defines:
- `id`: Unique stage identifier
- `name`: Human-readable stage name
- `type`: Stage type (START, NORMAL, END)
- `prompt`: Stage-specific instructions
- `nextStages`: Possible transitions with conditions

Example:
```json
{
  "id": "scheme_doubt_solving",
  "name": "Scheme Doubt Solving",
  "type": "NORMAL",
  "prompt": "Sure! Please ask your question about the scheme — eligibility, documents, benefits, or how to apply.",
  "nextStages": [
    {
      "nextStageId": "scheme_doubt_solving",
      "condition": "User has another doubt or follow-up question"
    },
    {
      "nextStageId": "kb_tool_call",
      "condition": "User asks how to apply or wants external help"
    }
  ]
}
```

## Project Structure

```
YOJNAPATH/
├── langgraph_app/
│   ├── __init__.py           # Module exports
│   ├── app.py               # Main application
│   ├── graph_builder.py     # LangGraph construction
│   └── tool_executor.py     # Tool execution (unused in simplified version)
├── models.py                # Pydantic models
├── stage_config.json        # Stage definitions
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Key Features

### 1. LLM Structured Output
The system uses structured output to get both conversational response and next stage determination:

```python
class LLMResponse(BaseModel):
    response: str
    next_stage: str
    confidence: float
```

### 2. Stage Validation
Only allows transitions to valid next stages as defined in the configuration:

```python
if current_stage.nextStages:
    allowed_stages = [ns.nextStageId for ns in current_stage.nextStages]
    if next_stage_id not in allowed_stages:
        next_stage_id = allowed_stages[0]  # Default to first allowed
```

### 3. Conversation Context
Maintains conversation history and provides context to the LLM for better responses.

## Development

### Adding New Stages

1. Add stage definition to `stage_config.json`
2. Update existing stages' `nextStages` to include new transitions
3. The graph will automatically adapt to the new configuration

### Customizing LLM Behavior

Modify the prompt building in `build_stage_prompt()` function in `graph_builder.py`.

## Error Handling

- Graceful fallback responses for API failures
- JSON parsing error handling for structured output
- Stage validation and default routing

## Dependencies

- `langgraph` - Graph-based conversation flow
- `langchain-core` - Core LangChain functionality
- `pydantic` - Data validation and modeling
- `requests` - HTTP client for Grok API
- `python-dotenv` - Environment variable management

---

Built  for rural citizens to access government scheme information easily. 
