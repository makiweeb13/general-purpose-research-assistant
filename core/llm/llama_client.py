from langchain_ollama import OllamaLLM

class LlamaClient:
    def __init__(self, model_name: str = "llama3.1:8b", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OllamaLLM(model=model_name, temperature=temperature)

    def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLaMA model given a prompt"""
        response = self.client.invoke(prompt)
        return response.strip()