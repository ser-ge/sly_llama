## Sly Llama


Sly LLama is for quickly prototyping large language model (llm) applications and is not intended for serious use.

Sly Llama provides the `@llm_call` decorator for turning python functions into llm calls and utility classes to parse the llm output with the [Sly](https://sly.readthedocs.io/en/latest/sly.html#introduction) Lexer class and pydantic BaseModel.

Be warned both sly-llama and sly abuse python syntax, your IDE will not be happy.


### Example:

    from sly_llama import llm_call

    from langchain.llms import OpenAI

    llm = OpenAI()

    @llm_call(llm)
    def add(x: str, y: str) -> int:
        """
        calculate {x} + {y}
        only return the number and nothing else
        """

    add(40, 2)
    #> 42


Note the doc string of the function is an f-string with inputs matching the function parameters.
The `@llm_call(lm)` decorator:
    - takes the `llm` function as input ( this can be any function that takes and returns strings)
    - formats the doc string of the function with its parameters to generate the prompt
    - calls `llm(promtp)` to generate an output
    - attempts to coerce the output into the type in function return annotation, in this case an `int`


    from pydantic import BaseModel

    class AddOutput(BaseModel):
        answer: int

    @classmethod
    def from_llm_output(cls, llm_output: str):
        output = {'answer' : int(llm_output.strip())}
        return cls.parse_obj(output)


    @llm_call(llm)
    def add(x: str, y: str) -> AddOutput:
        """
        calculate {x} + {y}
        only return the number and nothing else
        """

    add(40, 2)
    #> AddOutput(answer=42)

The return type can be any python class. If the class has a `from_llm_output` method `@llm_call` will
pass the output to this method to construct the target class, otherwise it will call the class directly on the output.


For a more complex example using the Sly lexer for token parsing see `mrkl_agent.py`.




