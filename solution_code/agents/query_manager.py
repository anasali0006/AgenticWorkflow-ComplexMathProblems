import json
import os
from utils import get_json_completion
from pydantic import BaseModel


# Define a structure to get response from AI Agent
class QueryManagerResponse(BaseModel):
    organization_found: bool
    organization_name: str
    organization_abbreviation: str
    query_without_organization: str
    is_math_question: bool


class QueryManagerAgent:
    
    def __init__(self):

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.PROJECT_DIR = os.path.dirname(self.BASE_DIR)
        self.data_file_path = os.path.join(self.PROJECT_DIR, "data", "new_combinations.json")    
        self.combinations_data = self.load_combinations_data()
        
        self.system_message = """You are a helpful AI Assistant, who is at first step of agentic AI workflow.
The workflow is to solve the mathematical questions related to some known organizations. 
You will be provided a user query. Your task is to:
1. Identify which organization this query is talking about (using list provided below), and give it at output
2. If the question does not contain information or hint about the organization, set the "ogranization_found" flag\
to False. Otherwise give it as True
3. Give the original user query at the output without the name of the origanization. Just ensure no mathematical details are lost.
4. If the query is not a mathematical question, set the "is_math_question" flag to False, otherwise True.

Output should be JSON with these fields:
{
    "organization_found" : True/False,
    "organization_name" : name of organization if found, otherwise empty string,
    "orgnization_abbreviation" : abbreviation of the organization using the list given to you
    "query_without_organization" : query,
    "is_math_question" : True/False
}

The user query might be complex, so do not change any essense of original mathematical query. It might include\
multiple questions, you need to make sure not to lose any information.

Here is the list of the organizations to chose from. It gives Abbreviations and corresponding Names:
    """

        self.system_prompt = self.system_message + str(self.combinations_data)

    def load_combinations_data(self):
        with open(self.data_file_path, "r") as f:
            return json.load(f)
    
    def execute(self, client, model, user_query):

        self.user_message = user_query
        return get_json_completion(client, model, self.system_prompt, self.user_message, QueryManagerResponse)
