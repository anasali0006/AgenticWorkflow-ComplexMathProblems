from math_excutor.math_functions import add, subtract, multiply, divide, power


class MathFunctionExecutor:
    def __init__(self):
        pass

    
    def execute_steps(self, steps_dict):
        computed_values = {}  # Stores results of each step

        try: 

            for step, expression in steps_dict.items():

                # Extract function name and arguments
                func_name, args_str = expression.split('(')
                args_str = args_str.rstrip(')')  # Remove closing parenthesis
                args = args_str.split(',')  # Split arguments

                # Convert arguments: replace step references with computed values
                evaluated_args = []
                for arg in args:
                    arg = arg.strip().strip("'")  # Remove surrounding single quotes if any
                    if arg in computed_values.keys():  # If it's a reference to a previous step
                        evaluated_args.append(computed_values[arg])
                    else:
                        evaluated_args.append(float(arg))  # Convert to integer
                
                # Execute the function dynamically
                if func_name == "add":
                    result = add(*evaluated_args)
                elif func_name == "subtract":
                    result = subtract(*evaluated_args)
                elif func_name == "multiply":
                    result = multiply(*evaluated_args)
                elif func_name == "divide":
                    result = divide(*evaluated_args)
                elif func_name == "power":
                    result = power(*evaluated_args)
                else:
                    raise ValueError(f"Unsupported function: {func_name}")

                # Store the computed result
                computed_values[step] = result
        
        except Exception as e:
            raise ValueError(f"Problem in Executor. {e}")
        
        # The last step would be the output, so return that. 
        final_answer = list(computed_values.values())[-1]
        return final_answer
    


