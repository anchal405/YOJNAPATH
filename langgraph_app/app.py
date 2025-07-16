import os
import sys
from typing import cast
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_app.graph_builder import (
    build_yojnapath_graph,
    init_conversation,
    add_user_input,
    stages,
    State
)

def print_separator():
    """Print a separator line"""
    print("\n" + "="*50 + "\n")

def print_stage_info(state: State):
    """Print current stage information"""
    current_stage_id = state.get("current_stage")
    if current_stage_id and current_stage_id in stages:
        stage = stages[current_stage_id]
        print(f"Current Stage: {stage.name} ({current_stage_id})")
        print(f"Stage Type: {stage.type}")

def main():
    """
    Main function to demonstrate the simplified YojnaPath conversational flow.
    """
    print("🚀 Starting YojnaPath Conversational Agent")
    print_separator()
    
    # Check if Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Warning: GROQ_API_KEY environment variable not set!")
        print("Please set your Groq API key in the .env file to use the LLM functionality.")
        print("For now, the app will still run but may encounter errors.")
        print()
    
    # Build the graph
    print("📊 Building LangGraph...")
    graph = build_yojnapath_graph()
    print("✅ Graph built successfully!")
    
    # Initialize conversation
    conversation_id = "demo-conversation"
    state = init_conversation(conversation_id)
    
    print(f"💬 Starting conversation: {conversation_id}")
    print_stage_info(state)
    print_separator()
    
    # Simulate conversation flow
    conversations = [
        "Hello, I want to know about government schemes for farmers.",
        "I want to know about PM Kisan Samman Nidhi Yojana.",
        "What are the eligibility criteria for this scheme?",
        "How do I apply for this scheme?",
        "Yes, please provide more details about the application process.",
        "Thank you, that's all I needed to know!"
    ]
    
    for i, user_input in enumerate(conversations, 1):
        print(f"👤 User ({i}): {user_input}")
        
        # Add user input to state
        state = add_user_input(state, user_input)
        
        # Process through the graph
        try:
            print("🤖 Processing...")
            result = graph.invoke(state, config={"recursion_limit": 10})
            state = cast(State, result)
            
            # Get the latest AI message
            messages = state.get("messages", [])
            if messages:
                latest_message = messages[-1]
                if hasattr(latest_message, 'content'):
                    print(f"🤖 YojnaPath: {latest_message.content}")
            
            print()
            print_stage_info(state)
            
            # Check if conversation ended
            current_stage_id = state.get("current_stage")
            if current_stage_id and current_stage_id in stages:
                stage = stages[current_stage_id]
                if stage.type.value == "END":
                    print("🏁 Conversation ended.")
                    break
                    
        except Exception as e:
            print(f"❌ Error processing conversation: {e}")
            print("This might be due to missing Groq API key or network issues.")
            print("You can set GROQ_API_KEY in a .env file or as an environment variable.")
        
        print_separator()
    
    print("✨ Demo completed!")

def interactive_mode():
    """
    Interactive mode for testing the conversational flow.
    """
    print("🎯 YojnaPath Interactive Mode")
    print("Type 'quit' to exit")
    print_separator()
    
    # Build the graph
    graph = build_yojnapath_graph()
    
    # Initialize conversation
    state = init_conversation("interactive-session")
    print_stage_info(state)
    print_separator()
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user input and process
            state = add_user_input(state, user_input)
            result = graph.invoke(state, config={"recursion_limit": 10})
            state = cast(State, result)
            
            # Display AI response
            messages = state.get("messages", [])
            if messages:
                latest_message = messages[-1]
                if hasattr(latest_message, 'content'):
                    print(f"🤖 YojnaPath: {latest_message.content}")
            
            print()
            print_stage_info(state)
            
            # Check if conversation ended
            current_stage_id = state.get("current_stage")
            if current_stage_id and current_stage_id in stages:
                stage = stages[current_stage_id]
                if stage.type.value == "END":
                    print("🏁 Conversation completed. Type anything to start a new conversation.")
                    state = init_conversation("interactive-session")
            
            print_separator()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YojnaPath Conversational Agent")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    else:
        main() 