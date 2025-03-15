from utils import get_json_completion
from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional


# JSON schema for output
class QueryBreakerResponse(BaseModel):
    answer_not_in_context: bool
    cannot_solve_due_to_limited_functions: bool
    direct_answer_found: str
    math_executor_needed: bool
    solution_steps: Optional[Dict[str, str]] = Field(default=None)


class QueryBreakerAgent:
    
    def __init__(self):

        self.system_prompt = """ You are a helpful AI Assistant. 
Your job is to take in a user query and give the correspoding mathematical functions that will solve the query step by step. 
Functions signatures of currently available functions. Every function takes exactly two inputs. Not more, not less:

- add(a,b)
- subtract(a,b)
- multiply(a,b)
- divide(a,b)

Follow these steps:
1. Understand the user query and context. Take your time to reason. Think hard to solve the problem
2. Check if the answer can be calculated from the given context. Try hard, most probably it will be there. Think again.
3. If so, see some examples given to you on how to break queries. They are just for your understanding. Learn from them. 
4. Break the user query into simple steps corresponding to functions given above. Recheck your reasoning
[Be careful about percentages. They need be to in range 0-100, so always multiply fraction with 100 for percentage]
5. If query needs additional functions, set the cannot_solve_due_to_limited_functions
6. Otherwise, give the steps with function signature to solve the user query. Recheck, make sure they are correct
7. Revise all steps above to ensure high accuracy.

Make sure to use double quotes for key-values and single quotes for referring steps inside the functions.
Strictly follow the single and double quotes. Output JSON
{   
    "answer_not_in_context" : true
    "cannot_solve_due_to_limited_functions": false,
    "solution_steps": {
                "step1": "add(5,3)",
                "step2": "subtract('step1',2)",
                "step3": "divide('step2',3)"
            }
        
}

Your output will go to math engine, so it is important that you follow the output strucutre strictly.
Always provide steps if both previous flags are false.

"""
        
    def execute(self, client, model, query_without_organization, organization_name, results_from_retriever):

        complete_context = results_from_retriever['Context'].str.cat(sep = " ")

        # select some questions and dialogues for few-shot learning
        contexts, questions, dialogues = [], [], []
        for idx, row in results_from_retriever.iterrows():
            contexts.append(row['Context'])
            questions.append(row['Question'])
            dialogues.append(row['Dialogue'])
            
            # Stop after selecting two rows
            if len(contexts) == 3:
                break
                
        # Assign variables dynamically based on available data
        question1 = questions[0] if len(questions) > 0 else None
        dialogue1 = dialogues[0] if len(dialogues) > 0 else None

        question2 = questions[1] if len(questions) > 1 else None
        dialogue2 = dialogues[1] if len(dialogues) > 1 else None


        user_query = f"""
Organization: ```{organization_name}```
User Query: ```{query_without_organization}```
Context: ```{complete_context}```
Example Question 1: ```{question1}```
Dialogue Breakdown 1: ```{dialogue1}```
Example Question 2: ```{question2}```
Dialogue Breakdown 2: ```{dialogue2}```
        """

        result = get_json_completion(client, model, self.system_prompt, user_query, QueryBreakerResponse)
        
        return result
    
