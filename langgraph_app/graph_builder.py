import os
import json
import sys
from typing import Dict, Any, TypedDict, Annotated, Literal, Optional, cast, List, Sequence
from datetime import datetime
import operator

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage

from stage_manager import Boto3StageManager
from tools.kb_tool import kb_tool
from tools.scheme_tool import scheme_tool
from models import StageType
from langgraph_app.tool_executor import ToolExecutor

# Load stage configuration
with open("stage_config.json", "r") as f:
    stages_data = json.load(f)

GENERIC_PROMPT = "You are a helpful government scheme assistant for rural citizens."
stage_manager = Boto3StageManager(
    generic_prompt=GENERIC_PROMPT,
    stages_info=json.dumps(stages_data),
)

tool_executor = ToolExecutor(tools=[kb_tool, scheme_tool])

# Custom reducer for string values
def last_value(a: Any, b: Any) -> Any:
    """Return the last value."""
    return b

# Custom reducer for lists
def append_list(a: list, b: list) -> list:
    """Append lists."""
    return a + b

# Define the state schema
class State(TypedDict, total=False):
    conversation_id: Annotated[str, last_value]  # Use last_value to handle updates
    active_stage: Annotated[str, last_value]  # Use last_value to handle updates
    messages: Annotated[list, append_list]  # Use append_list to handle updates
    next_stage: Annotated[str, last_value]  # Use last_value to handle updates
    response: Annotated[Dict[str, Any], last_value]  # Use last_value to handle updates
    prompt: Annotated[str, last_value]  # Use last_value to handle updates

# Define a function to get the prompt for the current active stage
def run_stage_node(state: State) -> State:
    """
    Gets the prompt for the current active stage and adds it to the state.
    """
    conversation_id = state.get("conversation_id", "default")
    active_stage = state.get("active_stage")
    
    # If active_stage is not in state, get the start stage
    if not active_stage:
        start_stage = stage_manager.get_start_stage()
        active_stage = start_stage.id
        state["active_stage"] = active_stage
    
    # Get the prompt for the active stage
    prompt = stage_manager.get_chain_for_current_active_stage(conversation_id)
    
    # Update the state with the prompt
    return {
        **state,
        "prompt": prompt,
        "active_stage": active_stage
    }

# Define a function to process the LLM response and extract the next stage
def process_llm_response(state: State) -> State:
    """
    Process the LLM response to extract the next stage and update the state.
    This function would parse the structured output from the LLM.
    """
    # In a real implementation, you would parse the structured output
    # For now, we'll assume the LLM response contains a next_stage field
    response = state.get("response", {})
    next_stage = response.get("next_stage") if isinstance(response, dict) else None
    
    # If next_stage is provided in the response, update the state
    if next_stage:
        # Update the active stage in the stage manager
        conversation_id = state.get("conversation_id", "default")
        stage_manager.set_active_stage(conversation_id, next_stage)
        
        return {
            **state,
            "next_stage": next_stage
        }
    
    # If no next_stage is provided, keep the current active stage
    return state

# Define a function to route to the next node based on the next_stage in state
def route_next_stage(state: State) -> str:
    """
    Route to the next node based on the next_stage in state.
    """
    # Get the next stage from state
    next_stage = state.get("next_stage")
    
    # If next_stage is not provided, get the current active stage
    if not next_stage:
        conversation_id = state.get("conversation_id", "default")
        active_stage = stage_manager.get_active_stage(conversation_id)
        if active_stage and active_stage.type == StageType.END:
            return END
        return active_stage.id if active_stage else stage_manager.get_start_stage().id
    
    # Check if the next stage is an end stage
    next_stage_obj = stage_manager.stage_id_2_stage.get(next_stage)
    if next_stage_obj and next_stage_obj.type == StageType.END:
        return END
    
    # Return the next stage ID for routing
    return next_stage

# Define a function for the stage mover tool
def stage_mover_tool(state: State, next_stage_name: str) -> State:
    """
    Tool for moving to the next stage.
    """
    # Find the stage by name
    stage = stage_manager.find_stage_by_name(next_stage_name)
    
    if stage:
        conversation_id = state.get("conversation_id", "default")
        stage_manager.set_active_stage(conversation_id, stage.id)
        
        return {
            **state,
            "active_stage": stage.id,
            "next_stage": stage.id
        }
    
    # If stage not found, return the current state
    return state

# Build the graph
def build_yojnapath_graph():
    """
    Build the LangGraph for YojnaPath.
    """
    # Create a new graph
    builder = StateGraph(State)
    
    # Add nodes for each stage
    for stage in stage_manager.stage_id_2_stage.values():
        builder.add_node(stage.id, run_stage_node)
    
    # Add a node for processing LLM response
    builder.add_node("process_llm_response", process_llm_response)
    
    # Set the entry point to the start stage
    builder.set_entry_point(stage_manager.get_start_stage().id)
    
    # Add edges between stages based on the stage configuration
    for stage in stage_manager.stage_id_2_stage.values():
        if stage.type == StageType.END:
            continue
        
        # Add edge from stage to process_llm_response
        builder.add_edge(stage.id, "process_llm_response")
        
        # Add conditional edge from process_llm_response to next stages
        if stage.nextStages:
            for next_stage in stage.nextStages:
                builder.add_edge("process_llm_response", next_stage.nextStageId)
    
    # Set the finish point
    builder.add_edge("process_llm_response", END)
    
    # Add conditional edge from process_llm_response to determine next stage
    builder.add_conditional_edges(
        source="process_llm_response",
        path=route_next_stage
    )
    
    # Compile and return the graph
    return builder.compile()

# Function to initialize the state
def init_state(conversation_id: str = "default") -> State:
    """
    Initialize the state for a new conversation.
    """
    start_stage = stage_manager.get_start_stage()
    stage_manager.set_active_stage(conversation_id, start_stage.id)
    
    return {
        "conversation_id": conversation_id,
        "active_stage": start_stage.id,
        "messages": [],
        "next_stage": "",
        "response": {}
    }

# Function to add a user message to the state
def add_user_message(state: State, message: str) -> State:
    """
    Add a user message to the state.
    """
    messages = state.get("messages", [])
    messages.append(HumanMessage(content=message))
    
    return {
        **state,
        "messages": messages
    }

# Function to add an AI message to the state
def add_ai_message(state: State, message: str) -> State:
    """
    Add an AI message to the state.
    """
    messages = state.get("messages", [])
    messages.append(AIMessage(content=message))
    
    return {
        **state,
        "messages": messages
    }
