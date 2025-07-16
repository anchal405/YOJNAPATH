import os
import json
import sys
from datetime import datetime
from typing import Dict, Any, cast

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langgraph_app.tool_executor import ToolExecutor

from langgraph_app.graph_builder import (
    build_yojnapath_graph,
    init_state,
    add_user_message,
    add_ai_message,
    stage_manager,
    State
)

def main():
    """
    Main function to demonstrate the YojnaPath graph.
    """
    # Build the graph
    graph = build_yojnapath_graph()
    
    # Initialize the state
    conversation_id = "demo-conversation"
    state = init_state(conversation_id)
    
    # Print the initial state
    print("Initial state:")
    if "active_stage" in state:
        active_stage = state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")
    print("\n" + "-" * 50 + "\n")
    
    # Add a user message
    user_input = "Hello, I want to know about government schemes for farmers."
    state = add_user_message(state, user_input)
    print(f"User: {user_input}")
    
    # Mock LLM response (in a real application, this would come from an LLM)
    # Here we're simulating the LLM determining the next stage
    mock_llm_response = {
        "response": "Sure, I can help you with information about government schemes for farmers. Would you like to know about a specific scheme or should I recommend schemes based on your profile?",
        "next_stage": "scheme_doubt_solving"
    }
    
    # Update the state with the LLM response
    state["response"] = mock_llm_response
    
    # Invoke the graph with the state
    result = graph.invoke(state)
    result_state = cast(State, result)
    
    # Print the result
    print("\nAI: " + mock_llm_response["response"])
    print("\n" + "-" * 50 + "\n")
    
    # Print the new active stage
    print("New state:")
    if "active_stage" in result_state:
        active_stage = result_state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")
    
    # Add another user message
    user_input = "I want to know about PM Kisan Samman Nidhi Yojana."
    state = add_user_message(result_state, user_input)
    print("\n" + "-" * 50 + "\n")
    print(f"User: {user_input}")
    
    # Mock another LLM response
    mock_llm_response = {
        "response": "The PM Kisan Samman Nidhi Yojana provides income support of ₹6,000 per year to all farmer families across the country in three equal installments of ₹2,000 each every four months. The scheme is fully funded by the Government of India. Would you like to know about eligibility criteria or how to apply?",
        "next_stage": "scheme_doubt_solving"
    }
    
    # Update the state with the LLM response
    state["response"] = mock_llm_response
    
    # Invoke the graph with the state
    result = graph.invoke(state)
    result_state = cast(State, result)
    
    # Print the result
    print("\nAI: " + mock_llm_response["response"])
    print("\n" + "-" * 50 + "\n")
    
    # Print the new active stage
    print("New state:")
    if "active_stage" in result_state:
        active_stage = result_state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")
    
    # Add another user message
    user_input = "How do I apply for this scheme?"
    state = add_user_message(result_state, user_input)
    print("\n" + "-" * 50 + "\n")
    print(f"User: {user_input}")
    
    # Mock another LLM response that moves to kb_tool_call stage
    mock_llm_response = {
        "response": "To apply for PM Kisan Samman Nidhi Yojana, you need to register on the PM Kisan portal or visit your nearest Common Service Center (CSC). Would you like me to provide more details on the application process?",
        "next_stage": "kb_tool_call"
    }
    
    # Update the state with the LLM response
    state["response"] = mock_llm_response
    
    # Invoke the graph with the state
    result = graph.invoke(state)
    result_state = cast(State, result)
    
    # Print the result
    print("\nAI: " + mock_llm_response["response"])
    print("\n" + "-" * 50 + "\n")
    
    # Print the new active stage
    print("New state:")
    if "active_stage" in result_state:
        active_stage = result_state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")
    
    # Add another user message
    user_input = "Yes, please provide more details."
    state = add_user_message(result_state, user_input)
    print("\n" + "-" * 50 + "\n")
    print(f"User: {user_input}")
    
    # Mock another LLM response that moves to farewell stage
    mock_llm_response = {
        "response": "No problem! You can submit your query here and our team will follow up with the right steps: [Google Form Link]. Is there anything else you would like to know?",
        "next_stage": "farewell"
    }
    
    # Update the state with the LLM response
    state["response"] = mock_llm_response
    
    # Invoke the graph with the state
    result = graph.invoke(state)
    result_state = cast(State, result)
    
    # Print the result
    print("\nAI: " + mock_llm_response["response"])
    print("\n" + "-" * 50 + "\n")
    
    # Print the new active stage
    print("New state:")
    if "active_stage" in result_state:
        active_stage = result_state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")
    
    # Add another user message
    user_input = "No, that's all. Thank you!"
    state = add_user_message(result_state, user_input)
    print("\n" + "-" * 50 + "\n")
    print(f"User: {user_input}")
    
    # Mock another LLM response that stays in farewell stage
    mock_llm_response = {
        "response": "Thank you for using YojnaPath. Have a great day! If you need help again, just say 'YojnaPath'.",
        "next_stage": "farewell"
    }
    
    # Update the state with the LLM response
    state["response"] = mock_llm_response
    
    # Invoke the graph with the state
    result = graph.invoke(state)
    result_state = cast(State, result)
    
    # Print the result
    print("\nAI: " + mock_llm_response["response"])
    print("\n" + "-" * 50 + "\n")
    
    # Print the final state
    print("Final state:")
    if "active_stage" in result_state:
        active_stage = result_state["active_stage"]
        print(f"Active stage: {active_stage}")
        print(f"Stage name: {stage_manager.stage_id_2_stage[active_stage].name}")
        print(f"Stage prompt: {stage_manager.stage_id_2_stage[active_stage].prompt}")

if __name__ == "__main__":
    main() 