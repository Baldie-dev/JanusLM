from llama_cpp import Llama

model_path = "C:/Users/micha/llama-2-13b-chat.Q5_K_M.gguf"

llm = Llama(
      model_path=model_path,
)
output = llm(
      "Q: Name the planets in the solar system? A: ",
      max_tokens=32,
      stop=["Q:", "\n"],
      echo=True
)
print(output)