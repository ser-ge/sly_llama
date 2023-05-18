from json import JSONDecodeError
from typing import List, Optional

from langchain.llms.fake import FakeListLLM
from pydantic import BaseModel, root_validator, validator


# %%

responses=[
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4"
]
# llm = FakeListLLM(responses=responses)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# from langchain.chains import LLMChain
# chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
# print(chain.run("colorful socks"))



# %%
from langchain.agents import Tool
from langchain.utilities import PythonREPL

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run
)


# def format_docstring(doc, **input_variables):


from langchain.llms import OpenAI
llm = OpenAI()




@llm_call(llm)
def mrkl(current_observation, tools, request, agent_scratchpad):

    """
    You are a helpful assistant designed to answer quesitons.

    You have access to the following tools:

    {tools}

    Use the following format:

    Request: the question you must answer if you can
    Thought: you should always think about what to do
    Action: name of the tool to use, should be one of [{tool_names}]
    Action Input: the input to the tool
    Observation: the result of the tool
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: [
        "is_possible": <boolean indicating if the request has been successfully answered>,
        "explanation": <a description of why and how the request can or cannot be answered>,
    ]

    Begin!

    Request: {request}
    {agent_scratchpad}

    """


class MrklLexer(Lexer):

    tokens = {ACTION_INPUT, ACTION, THOUGHT, FINAL_ANSWER}

    @_(r'.+Action Input.+[$\n]')
    def ACTION_INPUT(self, token):
        token.value = token.value.replace('Action Input', "")
        token.value = token.value.replace(':', '')
        token.value = token.value.strip('\n ')
        return token

    @_(r'.+Action.+[$\n]')
    def ACTION(self, token):
        token.value = token.value.replace('Action', "")
        token.value = token.value.replace(':', '')
        token.value = token.value.strip('\n ')
        return token

    @_(r'(.+)?Thought.+[$\n]')
    def THOUGHT(self, token):
        token.value = token.value.replace('Thought', "")
        token.value = token.value.replace(':', '')
        token.value = token.value.strip('\n ')
        return token

    @_(r'(.+)?Final Answer.+[$\n]')
    def FIANAL_ANSWER(self, token):
        token.value = token.value.replace('Final Answer', "")
        token.value = token.value.replace(':', '')
        token.value = token.value.strip('\n ')
        return token

    ignore_rest = '.+'
    ignore_newline ='\n'

    def __call__(self, input_string):
        """
        return the token in a dict:
        {token_name : value}
        """
        tokens = list(self.tokenize(input_string))
        output_dict = {t.type.lower() : t.value for t in tokens  }
        return output_dict


class MrklOutput(BaseModel):
    action: str
    action_input : Optional[str]
    final_answer : Optional[str]
    tools: dict =  {'python_repl' : repl_tool}


    @root_validator
    def validate_tool(cls, values):
        action = values['action']
        tools = values['tools']
        if action not in tools.keys():
            raise LlmException(f"{action} is not a valid tool, try again")
        return

    @root_validator
    def check_action_or_answer(cls, values):
        action = 'action_input' in vlaues and 'action' in values
        answer = 'final_answer' in values
        if not any(action, answer):
            raise LlmException("You must either choose an action or give the final answer")


# def retry(llm_call):



output_parser = MrklLexer()
output_dict = output_parser(mrkl.__doc__)
mrkl_output = MrklOutput(**output_dict)

# %%

output = mrkl(current_observation, tools, request, agent_scratchpad)
output_dict = output_parser(test_string)




# %%


##
from pydantic import BaseModel

# %%

import re


json_regex = r"[\s\n]([{\[].*?[}\]])[\s\n]"
# json_regex = r".+({.+[:,].+}|\[.+[,:].+\]).+"
json_fidner = re.compile(json_regex)

# %%

json_string = re.search(json_regex, result)


# %%
class JsonLLMReader(BaseModel):

    @classmethod
    def from_llm_output(cls, input_string):
        """
        TODO search the output to extract JSON
        """
        try:
            return cls.parse_raw(input_string)
        except JSONDecodeError as e:
            raise LlmException(f"The output was not valide JSON, be sure to only provide JSON")











