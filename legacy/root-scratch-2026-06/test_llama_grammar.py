import llama_cpp
grammar = llama_cpp.LlamaGrammar.from_string("root ::= \"COMPUTE: \" [0-9]+ \" + \" [0-9]+ \" = \" [0-9]+\n")
print(grammar)
