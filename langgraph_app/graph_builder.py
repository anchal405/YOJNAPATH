import os
import json
import sys
from typing import Dict, Any, TypedDict, Annotated, Literal, Optional, cast, List, Sequence
from datetime import datetime
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from models import StageType, Stage, NextStage, LLMResponse

# Load stage configuration
with open("stage_config.json", "r") as f:
    stages_data = json.load(f)

# Create stage objects from configuration
stages: Dict[str, Stage] = {}
for stage_data in stages_data:
    stage = Stage(**stage_data)
    stages[stage.id] = stage

# Initialize Groq LLM with structured output
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
structured_llm = llm.with_structured_output(LLMResponse)

# Custom reducer for string values
def last_value(a: Any, b: Any) -> Any:
    """Return the last value."""
    return b

# Custom reducer for lists
def append_list(a: list, b: list) -> list:
    """Append lists."""
    return a + b

# Define the simplified state schema
class State(TypedDict, total=False):
    conversation_id: Annotated[str, last_value]
    current_stage: Annotated[str, last_value]
    messages: Annotated[List[HumanMessage | AIMessage], append_list]
    user_input: Annotated[str, last_value]

def get_start_stage() -> Stage:
    """Get the start stage from configuration"""
    for stage in stages.values():
        if stage.type == StageType.START:
            return stage
    raise ValueError("No start stage found in configuration")

def build_stage_prompt(stage: Stage, messages: List[HumanMessage | AIMessage]) -> str:
    """Build the prompt for the current stage including context"""
    
    # Get conversation history (last 3 exchanges)
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    conversation_context = ""
    
    if recent_messages:
        conversation_context = "\n\nRecent conversation:\n"
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                conversation_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_context += f"Assistant: {msg.content}\n"
    
    # Get possible next stages for context
    next_stages_info = ""
    if stage.nextStages:
        next_stages_info = "\n\nPossible next stages:\n"
        for next_stage in stage.nextStages:
            next_stage_obj = stages.get(next_stage.nextStageId)
            if next_stage_obj:
                next_stages_info += f"- {next_stage.nextStageId} ({next_stage_obj.name}): {next_stage.condition}\n"
    
    prompt = f"""You are YojnaPath, a helpful government scheme assistant for rural citizens.

Current Stage: {stage.name}
Stage Description: {stage.prompt}

{conversation_context}

{next_stages_info}

Instructions:
1. Respond helpfully to the user's message
2. Choose the most appropriate next stage based on the user's intent
3. If user wants to end conversation or says goodbye, choose 'farewell'
4. If user has scheme-related doubts, choose 'scheme_doubt_solving'
5. If user needs application help, choose 'kb_tool_call'

Respond with:
- response: Your helpful response to the user
- next_stage: The ID of the next appropriate stage
- confidence: Your confidence in the stage choice (0.0 to 1.0)"""

    return prompt

def process_stage(state: State) -> State:
    """Process the current stage and generate LLM response with next stage"""
    
    current_stage_id = state.get("current_stage")
    if not current_stage_id:
        # Start with the initial stage
        start_stage = get_start_stage()
        current_stage_id = start_stage.id
        state["current_stage"] = current_stage_id
    
    current_stage = stages[current_stage_id]
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    # Add user input to messages if provided
    if user_input:
        messages.append(HumanMessage(content=user_input))
    
    # For END stages, don't call LLM, just return a final message
    if current_stage.type == StageType.END:
        final_message = current_stage.prompt or "Thank you for using YojnaPath. Have a great day!"
        messages.append(AIMessage(content=final_message))
        return {
            **state,
            "messages": messages,
            "current_stage": current_stage_id,
            "user_input": ""
        }
    
    # Build stage-specific prompt
    prompt = build_stage_prompt(current_stage, messages)
    
    # Get structured response from LLM
    try:
        response = structured_llm.invoke(prompt)
        
        # Ensure we have an LLMResponse object
        if isinstance(response, dict):
            llm_response = LLMResponse(**response)
        else:
            llm_response = response
        
        # Add AI response to messages
        messages.append(AIMessage(content=llm_response.response))
        
        # Determine next stage
        next_stage_id = llm_response.next_stage
        
        # Validate next stage is allowed from current stage
        if current_stage.nextStages:
            allowed_stages = [ns.nextStageId for ns in current_stage.nextStages]
            if next_stage_id not in allowed_stages:
                print(f"⚠️  Invalid stage transition: {current_stage_id} -> {next_stage_id}")
                print(f"Allowed stages: {allowed_stages}")
                # Default to first allowed stage if invalid
                next_stage_id = allowed_stages[0] if allowed_stages else "farewell"
        else:
            # If no next stages defined, go to farewell
            next_stage_id = "farewell"
        
        # Make sure the next stage exists
        if next_stage_id not in stages:
            print(f"⚠️  Stage {next_stage_id} not found, defaulting to farewell")
            next_stage_id = "farewell"
        
        print(f"🔄 Stage transition: {current_stage_id} -> {next_stage_id}")
        
        return {
            **state,
            "messages": messages,
            "current_stage": next_stage_id,
            "user_input": ""  # Clear user input after processing
        }
        
    except Exception as e:
        print(f"Error in LLM call: {e}")
        # Fallback response
        messages.append(AIMessage(content="I apologize, but I'm having trouble processing your request. Could you please try again?"))
        return {
            **state,
            "messages": messages,
            "current_stage": "farewell",  # Go to farewell on error
            "user_input": ""
        }

def should_continue(state: State) -> str:
    """Determine if conversation should continue or end"""
    current_stage_id = state.get("current_stage")
    user_input = state.get("user_input", "")
    
    print(f"🎯 Current stage: {current_stage_id}, User input: '{user_input}'")
    
    if current_stage_id:
        stage = stages.get(current_stage_id)
        if stage and stage.type == StageType.END:
            print("🏁 Ending conversation - reached END stage")
            return END
    
    # If there's no user input, also end (to prevent infinite loops)
    if not user_input.strip():
        print("🏁 Ending conversation - no user input")
        return END
        
    print("➡️  Continuing conversation")
    return "continue"

def build_yojnapath_graph():
    """Build a simplified LangGraph for conversational flow"""
    
    # Create the graph
    builder = StateGraph(State)
    
    # Add a single conversation node
    builder.add_node("conversation", process_stage)
    
    # Set entry point
    builder.set_entry_point("conversation")
    
    # Add conditional edges
    builder.add_conditional_edges(
        source="conversation",
        path=should_continue,
        path_map={
            "continue": "conversation",
            END: END
        }
    )
    
    # Compile the graph
    return builder.compile()

def init_conversation(conversation_id: str = "default") -> State:
    """Initialize a new conversation"""
    start_stage = get_start_stage()
    
    return {
        "conversation_id": conversation_id,
        "current_stage": start_stage.id,
        "messages": [],
        "user_input": ""
    }

def add_user_input(state: State, user_input: str) -> State:
    """Add user input to the state"""
    return {
        **state,
        "user_input": user_input
    }
