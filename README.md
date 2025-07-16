# YojnaPath - Government Scheme Assistant

YojnaPath is a conversational AI assistant that helps rural citizens navigate and understand government schemes. It uses LangGraph to manage the conversation flow through different stages.

## Project Structure

- `langgraph_app/`: Contains the main application code
  - `graph_builder.py`: Defines the LangGraph state graph
  - `app.py`: Demo application that shows how to use the graph
- `stage_manager.py`: Manages conversation stages and prompts
- `prompt_manager.py`: Manages prompt templates
- `models.py`: Defines data models
- `stage_config.json`: Configuration for conversation stages
- `tools/`: Contains tool implementations
  - `kb_tool.py`: Knowledge base tool
  - `scheme_tool.py`: Government scheme tool

## How It Works

1. The system uses a stage-based conversation flow defined in `stage_config.json`.
2. Each stage has a prompt and potential next stages.
3. The LLM response determines which stage to transition to next.
4. The active stage is stored in the LangGraph state.
5. The stage manager provides the appropriate prompt template for the current stage.

## Conversation Flow

1. **Greeting Statement**: Initial greeting and understanding user intent
2. **Scheme Doubt Solving**: Answering questions about specific schemes
3. **Gather Information**: Collecting user information for personalized recommendations
4. **Preference**: Understanding user preferences for scheme categories
5. **Recommend Scheme**: Suggesting relevant schemes based on user profile
6. **KB Tool Call**: Providing external help for application process
7. **Farewell**: Ending the conversation

## Running the Demo

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the demo application:
   ```
   python -m langgraph_app.app
   ```

## Extending the System

To add new stages:
1. Add the stage definition to `stage_config.json`
2. Update the tool implementations if needed

To integrate with a real LLM:
1. Replace the mock LLM responses in `app.py` with actual LLM calls
2. Implement the structured output parsing to extract the next stage

## Requirements

See `requirements.txt` for the list of required packages. 