import llama_cpp
grammar_str = r"""
root ::= (text | compute)+
text ::= [a-zA-Z0-9.,?!' \n]+
compute ::= "COMPUTE: " [0-9]+ " " [+\-*/] " " [0-9]+ " = " [0-9]+ "\n"
"""
try:
    g = llama_cpp.LlamaGrammar.from_string(grammar_str)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
