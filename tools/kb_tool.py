from langchain_core.tools import BaseTool, tool

@tool
def kb_tool(query: str) -> str:
    """
    A tool to query the knowledge base for information about government schemes.
    
    Args:
        query: The query to search for in the knowledge base.
        
    Returns:
        str: The information retrieved from the knowledge base.
    """
    # In a real implementation, this would query a database or API
    # For now, we'll just return a placeholder response
    return f"Here is information about your query: {query}. Please submit this query through our form for detailed assistance."

# Create a tool instance
kb_tool = kb_tool 