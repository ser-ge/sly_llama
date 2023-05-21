
def test_mrkl_lexer():
    from scratch import MrklLexer
    test_output = """
     Thought: I need to use python to generate the fibonacci numbers
        Action: python_repl
        Action Input:
        def fibonacci(n):
            a = 0
            b = 1
            if n < 0:
                print("Incorrect input")
            elif n == 0:
                return a
            elif n == 1:
                return b
            else:
                for i in range(2,n):
                    c = a + b
                    a = b
                    b = c
                return b

        for i in range(10):
            print(fibonacci(i))
        Observation:
        0
        1
        1
        2
        3
        5
        8
        13
        21
        34
        Thought: I now know the first 10 fib
        numbers

        Final Answer: {
            "is_possible": true,
            "explanation": "Using the python_repl tool, a function was defined to generate the fibonacci sequence and then the first 10 values were printed

        Observation
        dfsdff
    """



