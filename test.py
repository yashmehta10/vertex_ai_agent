from vertexai.preview import reasoning_engines

# print(reasoning_engines.ReasoningEngine.list())
ENGINE_PATH = ""

remote_app = reasoning_engines.ReasoningEngine(ENGINE_PATH)

response = remote_app.query(input="What is the weather in Sydney on 24th June? Should I go to the office?")
print(response)