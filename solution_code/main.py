from config import initialize_client
from agents.query_manager import QueryManagerAgent
from retrieval.retriever import RetrievalEngine


class Driver:
    
    def __init__(self):
        self.client = initialize_client()
        self.chat_model = 'gpt-4o-mini'
        self.query_manager_agent =  QueryManagerAgent()
        self.retrieval_engine = RetrievalEngine()

    def main_flow(self, input_message):

        # call the query manager agent, which gets the name of the organization
        query_manager_results = self.query_manager_agent.execute(self.client, self.chat_model, input_message)

        organization_found= query_manager_results.organization_found
        organization_name= query_manager_results.organization_name
        organization_abbreviation =  query_manager_results.organization_abbreviation
        query_without_organization= query_manager_results.query_without_organization
        is_math_question= query_manager_results.is_math_question
        
        print(organization_found)
        print(organization_name)
        print(organization_abbreviation)
        print(query_without_organization)
        print(is_math_question)

        if not organization_found:
            return "Organization information not provided"
        if not is_math_question:
            return "Not a mathematical question"
        
        # call the retrieval engine
        result = self.retrieval_engine.run_retrieval_engine(self.client, 
                                                         query_without_organization, 
                                                         organization_abbreviation).iloc[0]
       
        
        # call the query_breaker


        # call the math_executor


        # call the results formattor/do result formatting


        # give back the result found. 
    
        pass



if __name__ == "__main__":
    
    driver = Driver()
    
    while True:
        input_message = input("Question: ")
        driver.main_flow(input_message)