"""
Using sly and sly_llama to implement MRKL
"""

from typing import List, Optional
from pydantic import root_validator
from sly import Lexer

from sly_llama import llm_call, SlyBaseModel, LlmException

from langchain import OpenAI
from langchain.agents import Tool

# TODO set model name in .env
llm = OpenAI(model_name="gpt-3.5-turbo")


class MrklLexer(Lexer):
    """
    Lexer for Mrkl Output
    This class is used to parse the raw output into tokens before being vlaidated by
    pydantic class.
    """

    # define the tokens we are going to match
    tokens = {ACTION_INPUT, ACTION, THOUGHT, FINAL_ANSWER}

    # ignore new lines so that we can match multiline tokens
    ignore = "\t \n"

    # define regex for each token and clean the captured string value in token.value

    @_(r"Thought(.|\n)*?(?=Final Answer|Action)")
    def THOUGHT(self, token):
        token.value = token.value.replace("Thought", "").replace(":", "").strip("\n ")
        return token

    @_(r"Action Input(.|\n)*?(?=Observation)")
    def ACTION_INPUT(self, token):
        token.value = token.value.replace("Action Input:", "")
        return token

    @_(r"Action(.|\n)*(?=Action Input)")
    def ACTION(self, token):
        token.value = token.value.replace("Action", "").replace(":", "").strip("\n ")
        return token

    @_(r"Final Answer(.|\n)*")
    def FINAL_ANSWER(self, token):
        token.value = token.value.replace("Final Answer", "").strip("\n ")

        return token

    # this is a strict lexer, we must ignore unmatched tokens else error.
    ignore_rest = ".+"


class MrklOutput(SlyBaseModel):
    """
    Model to validate the output of the Mrkl llm call
    """

    action: Optional[str] = None
    action_input: Optional[str] = None
    thought: Optional[str] = None
    final_answer: Optional[str] = None

    @staticmethod
    def lexer():
        """
        Lexer for the Mrkl output, this tokenises the raw output before
        MrklOutput is built form the resulting dict. See SlyBaseModel for
        more info.
        """
        return MrklLexer()

    @root_validator(pre=True)
    def check_action_or_answer(cls, values):
        """
        Ensure that either an action or a final answer is given.
        """
        action = "action_input" in values and "action" in values
        answer = "final_answer" in values

        if not any([action, answer]):
            raise LlmException(
                "You must either choose an action or give the final answer"
            )

        return values


@llm_call(
    llm,
    stop_sequence="Observation",
    verbose=False,
    return_prompt=True,
    return_llm_output=True,
)
def mrkl_start(tools, tool_names, request) -> MrklOutput:
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
        "answer" : <the answer directly addressing the request>
    ]

    Begin!

    Request: {request}

    """


@llm_call(
    llm,
    stop_sequence="Observation",
    verbose=False,
    return_prompt=True,
    return_llm_output=True,
)
def mrkl_step(history, current_observation) -> MrklOutput:
    """
    {history}{current_observation}
    """


def insert_newline_after_match(string, pattern: str = "Action Input:"):
    """
    Inserts a newline after the given pattern in the given string.
    """
    return string.replace(pattern, pattern + "\n")


def mrlkl_agent(
    query: str, tools_list: List[Tool], max_iters: int, max_retries: int
) -> str | None:
    """
    Runs the MRLKL agent with the given query, tools list, and maximum number of iterations.

    Parameters
    ----------
    query : str
        The query to be used by the MRLKL agent.
    tools_list : List[Tool]
        A list of tools to be used by the MRLKL agent.
    max_iters : int
        The maximum number of iterations to run the MRLKL agent for.
    max_retries : int
        The maximum number of retries to attempt before giving up if LlmException is thrown
    """

    tools = {tool.name: tool for tool in tools_list}
    tool_info = str(({t.name: t.description for t in tools.values()}))
    tool_names = str(tools.keys())

    # Start the MRLKL agent with the initial conditions
    mrlkl_output, first_prompt, raw_output = mrkl_start(tool_info, tool_names, query)

    last_output = insert_newline_after_match(raw_output, "Action Input:")
    history = first_prompt + last_output

    print(history)

    for _ in range(max_iters):
        if mrlkl_output.action in tools:
            current_observation = tools[mrlkl_output.action](mrlkl_output.action_input)
        else:
            current_observation = (
                f"{mrkl_start.action} not a valid tool, try another one"
            )

        for i in range(max_retries):
            try:
                mrlkl_output, last_prompt, raw_output = mrkl_step(
                    history, current_observation
                )
                break

            except LlmException as e:
                current_observation = e.message
                mrlkl_output, last_prompt, raw_output = mrkl_step(
                    history, current_observation
                )

        # the llm one shot learns better if it can see last action separated by new line, esp code indent
        last_output = insert_newline_after_match(raw_output, "Action Input:")

        history = last_prompt + last_output
        print(last_prompt)
        print(last_output)

        if mrlkl_output.final_answer:
            return mrlkl_output.final_answer
