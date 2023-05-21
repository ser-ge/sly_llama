from langchain.utilities import PythonREPL
from langchain.agents import Tool

from mrkl_agent import mrlkl_agent

def test_mrkl_fib():

    python_repl = PythonREPL()

    def strip_code(input_code):
        """
        Strips the code of any backticks or code identifiers
        And runs the repl
        """
        input_code = input_code.strip('\n').strip('`').strip('python')
        print(input_code)
        return python_repl.run(input_code)

    repl_tool = Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. Do not add backticks ``` or code identifers",
            func=strip_code,
            )

    query = "use python to find the first 10 fib numbers"

    tools = [repl_tool]

    output = mrlkl_agent(query, tools, max_iters=10, max_retries=4)

    expected_output = '0, 1, 1, 2, 3, 5, 8, 13, 21, 34'

    assert expected_output in output, "no fib!"



if __name__ == '__main__':
    test_mrkl_fib()
