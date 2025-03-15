from config import initialize_client
from agents.query_manager import QueryManagerAgent
from agents.query_breaker import QueryBreakerAgent
from retrieval.retriever import RetrievalEngine
from math_excutor.math_executor import MathFunctionExecutor

class Driver:
    
    def __init__(self):
        self.client = initialize_client()
        self.chat_model = 'gpt-4o-mini'
        self.breaking_model = 'gpt-4o-mini'
        self.query_manager_agent =  QueryManagerAgent()
        self.retrieval_engine = RetrievalEngine()
        self.query_breaker_agent = QueryBreakerAgent()
        self.math_executor = MathFunctionExecutor()

    def main_flow(self, input_message):

        # call the query manager agent, which gets the name of the organization
        query_manager_results = self.query_manager_agent.execute(self.client, self.chat_model, input_message)

        organization_found= query_manager_results.organization_found
        organization_name= query_manager_results.organization_name
        organization_abbreviation =  query_manager_results.organization_abbreviation
        query_without_organization= query_manager_results.query_without_organization
        is_math_question= query_manager_results.is_math_question
        

        # Giving the name of the organziation is important, otherwise, it does not make much sense. 
        if not organization_found:
            return {"message": "Organization information not provided",
                    "answer": None}
         
        
        # For now, I am limiting the scope to only mathematical questions, but this can be enhanced. 
        # Simple retrievl questions can be answered, but for this prototype, only mathematical questions are allowed:
        # Those questions, which are solveable by using math functions like add, subtract, multiply and divide
        if not is_math_question:
            return {"message": "Not a mathematical question",
                    "answer": None}
        
        
        # call the retrieval engine
        retriever_df = self.retrieval_engine.run_retrieval_engine(self.client, 
                                                         query_without_organization, 
                                                         organization_abbreviation)
       
        
        # call the query_breaker
        query_breaker_result = self.query_breaker_agent.execute(self.client, 
                                                                self.breaking_model, 
                                                                query_without_organization, 
                                                                organization_name, 
                                                                retriever_df)

        answer_not_in_context = query_breaker_result.answer_not_in_context
        cannot_solve_due_to_limited_functions = query_breaker_result.cannot_solve_due_to_limited_functions
        solution_steps = query_breaker_result.solution_steps

        if answer_not_in_context:
            return {"message": "Not able to find answer in the provided context",
                    "answer": None}

        if cannot_solve_due_to_limited_functions:
            return {"message": "Only four operations supported yet. Add, Sub, Mul, Div",
                    "answer": None}
        
        if solution_steps is None:
            return {"message": "No steps produced",
                    "answer": None}

        # call the math_executor
        try:

            final_answer = self.math_executor.execute_steps(solution_steps)

        except Exception as e:
            return {"message": f"Problem in executor",
                    "answer": None}
            

        return {"message": "Success",
                "answer": final_answer,
                "debug": solution_steps
                }
            



if __name__ == "__main__":
    
    driver = Driver()
    
    while True:
        input_message = input("Question: ")
        print(driver.main_flow(input_message))