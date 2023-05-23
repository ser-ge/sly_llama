from collections import namedtuple
from typing import List, OrderedDict
from pydantic.main import ModelMetaclass
from pydantic import BaseModel
from sly import Lexer

import re


class PatternError(Exception):
    """
    Exception raised if there's some kind of problem with the specified
    regex patterns in the lexer.
    """

    pass


class MetaLLmOutput(ModelMetaclass):
    @classmethod
    def __prepare__(mcs, name, bases, **kwargs):
        namespace = OrderedDict()

        def _(pattern, *extra):
            patterns = [pattern, *extra]

            def decorate(func):
                pattern = "|".join(f"({pat})" for pat in patterns)
                if hasattr(func, "pattern"):
                    func.pattern = pattern + "|" + func.pattern
                else:
                    func.pattern = pattern
                return func

            return decorate

        namespace["_"] = _

        return namespace

    def __new__(mcs, name, bases, namespace, **kwargs):
        del namespace["_"]

        # construct the namespace for pydanticn __new__ excluding the parsing funcs
        new_namespace = OrderedDict()
        parse_funcs = {}
        # extract the parsing functions to dict
        for key, value in namespace.items():
            if hasattr(value, "pattern"):
                parse_funcs[key] = value
            else:
                new_namespace[key] = value

        # TODO validate parse func names against namespace
        # get cls from pydantic __new__
        cls = super().__new__(mcs, name, bases, new_namespace, **kwargs)

        # add the parse_funcs back to cls attrs
        cls.parse_funcs = parse_funcs

        return cls


class LlmOutput(BaseModel, metaclass=MetaLLmOutput):
    @classmethod
    def from_llm_output(cls, llm_output: str):
        # construct the dict from wich to initilise the pydantic object
        args_dict = {}

        for field_name, parse_func in cls.parse_funcs.items():
            field_name = field_name.lower()
            pattern = parse_func.pattern

            try:
                # TODO define re flags
                cpat = re.compile(pattern)
            except Exception as e:
                raise PatternError(f"Invalid regex for field {field_name}") from e

            # Verify that the pattern doesn't match the empty string
            if cpat.match(""):
                raise PatternError(f"Regex for field {field_name} matches empty input")

            re_matches = cpat.findall(llm_output)

            # only run parse func if matches are found
            if re_matches:
                field_value = parse_func(re_matches)
                args_dict[field_name] = field_value

        return cls(**args_dict)
