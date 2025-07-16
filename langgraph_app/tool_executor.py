from typing import Dict, Any, List

class ToolExecutor:
    """
    A simple tool executor that can run tools.
    """
    
    def __init__(self, tools: List[Any]):
        """
        Initialize the tool executor with a list of tools.
        
        Args:
            tools: A list of tools to execute.
        """
        self.tools = {}
        for tool in tools:
            # Handle different types of tools
            if hasattr(tool, "name"):
                self.tools[tool.name] = tool
            elif hasattr(tool, "__name__"):
                self.tools[tool.__name__] = tool
            else:
                # For tools without a name attribute, use the function itself as the key
                self.tools[str(tool)] = tool
    
    def invoke(self, tool_invocation: Dict[str, Any]) -> Any:
        """
        Invoke a tool with the given input.
        
        Args:
            tool_invocation: A dictionary with 'tool' and 'tool_input' keys.
            
        Returns:
            The result of the tool execution.
        """
        tool_name = tool_invocation.get("tool")
        tool_input = tool_invocation.get("tool_input", {})
        
        if not tool_name or tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        
        if isinstance(tool_input, dict):
            return tool(**tool_input)
        else:
            return tool(tool_input) 