from langchain_core.tools import BaseTool, tool
from typing import Dict, List, Optional

@tool
def scheme_tool(
    query: Optional[str] = None, 
    filters: Optional[Dict[str, str]] = None, 
    user_profile: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    A tool to search for and recommend government schemes based on query or user profile.
    
    Args:
        query: Optional query string to search for specific schemes.
        filters: Optional filters to narrow down scheme search (e.g., category, state).
        user_profile: Optional user profile data for personalized recommendations.
        
    Returns:
        List[Dict[str, str]]: A list of relevant schemes with their details.
    """
    # In a real implementation, this would query a database or API
    # For now, we'll just return placeholder schemes
    schemes = [
        {
            "name": "PM Kisan Samman Nidhi Yojana",
            "description": "Income support of ₹6,000 per year to farmer families",
            "eligibility": "All farmer families with cultivable land",
            "benefits": "₹6,000 per year in three equal installments",
            "application_process": "Online registration on PM Kisan portal or through CSC"
        },
        {
            "name": "Pradhan Mantri Awas Yojana (PMAY)",
            "description": "Housing for all by 2022",
            "eligibility": "Economically weaker sections and low income groups",
            "benefits": "Financial assistance for house construction",
            "application_process": "Apply through local municipal office or online portal"
        },
        {
            "name": "Pradhan Mantri Fasal Bima Yojana",
            "description": "Crop insurance scheme",
            "eligibility": "All farmers including sharecroppers and tenant farmers",
            "benefits": "Insurance coverage and financial support in case of crop failure",
            "application_process": "Apply through local agriculture office or online portal"
        }
    ]
    
    # Return all schemes for this demo
    return schemes

# Create a tool instance
scheme_tool = scheme_tool
