from inspect import signature
from functools import wraps
import unicodedata
import re

from pydantic import BaseModel, root_validator, validator
from sly import Lexer, Parser

from json import JSONDecodeError
import json


class LlmException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def llm_call(llm, verbose=False):
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

            if verbose: print(raw_output)

            return cast_func(llm(prompt))

        return decorated
    return decorator



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
            raise LlmException(f"{text} \n The output was not valid JSON, be sure to only provide JSON")


