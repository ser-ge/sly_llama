from typing import Callable
from inspect import signature
from functools import wraps
from typing import Optional
import unicodedata
import re

from pydantic import BaseModel, root_validator, validator
from sly import Lexer, Parser



class LlmException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self) -> str:
        return self.message

class RetryException(Exception):

    def __init__(self, message: str):
        self.message = message

    def __str__(self) -> str:
        return self.message


def llm_call(llm, verbose=False, stop_sequence: Optional[str]=None,return_prompt=True, return_llm_output=True)-> Callable:
    def decorator(func):
        docs = func.__doc__
        func_signature = signature(func)
        input_variables = list(iter(func_signature.parameters))
        target_type = func_signature.return_annotation

        cast_func = target_type.from_llm_output if hasattr(target_type, 'from_llm_output') else target_type

        @wraps(func)
        def decorated(*args):
            format_args = {arg : val for arg, val in zip(input_variables, args)}
            prompt = docs.format(**format_args)

            raw_output = llm(prompt)

            if stop_sequence is not None:
                trunc_output = re.split(stop_sequence, raw_output)[0]
                trunc_output += stop_sequence
            else:
                trunc_output = raw_output

            if verbose: print(trunc_output)

            values_to_return = (cast_func(trunc_output), prompt, trunc_output)
            filters = [True, return_prompt, return_llm_output]

            return tuple(v for (v, f) in zip(values_to_return, filters) if f)

        return decorated
    return decorator



# Models

class JsonBaseModel(BaseModel):

    @classmethod
    def from_llm_output(cls, text : str):
        """
        TODO search the output to extract JSON

        """
        # Remove all \n and \t from the text
        text = re.sub(r"[\n\t]", "", text)

        # Find the first { and the last }
        # this needs to be done AFTER removing the above because (for some reason)
        # when removing the \n and \t, the resultant text sometimes ends up with spaces
        first = text.find("{")
        last = text.rfind("}")

        text = unicodedata.normalize("NFKD", text[first : last + 1]).encode(
                "ascii", "ignore"
            )
        try:
            return cls.parse_raw(text)
        except Exception as e:
            raise LlmException(f"{text} \n The output was not valid JSON, be sure to only provide JSON. Error: {e}")


class SlyBaseModel(BaseModel):
    """
    BaseModel to be used with sly lexer. Implements from_llm_output which calls
    the provided lexer to tokenise llm_output before building the pydantic model.

    Any subclass should implement the lexer() static method which returns the Lexer to be used
    to tokenise the raw output before being validated.

    TODO currently assumes unique values for each token, handle multiple.
    """

    @staticmethod
    def lexer() -> Lexer:
        raise NotImplementedError

    @classmethod
    def from_llm_output(cls, llm_output):

        tokens = list(cls.lexer().tokenize(llm_output))
        output_dict = {t.type.lower(): t.value for t in tokens}

        return cls.parse_obj(output_dict)

