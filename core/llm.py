import openai
import requests
import json
from core.config import settings

class LLMInterface:
    def __init__(self, api_key: str = settings.OPENAI_API_KEY):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        self.client = openai.OpenAI(api_key=api_key)

    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> list[float]:
        """
        Generates an embedding for the given text using a specified OpenAI model.
        """
        print(f"--- Attempting to get embedding for text: '{text[:100]}...'") # <-- ADD THIS
        text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            print(f"--- Successfully generated embedding with dimension: {len(embedding)}") # <-- ADD THIS
            return embedding
        except Exception as e:
            print(f"--- ERROR getting embedding: {e}") # <-- ADD THIS
            return []
    def _chat_completion(self, prompt: str, system_prompt: str, model: str = "gpt-4.1-mini") -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during OpenAI chat completion: {e}")
            return f"Error: Could not get response from LLM. Details: {e}"

    def generate_thought(self, prompt: str, context: str) -> str:
        """Generates an intermediate reasoning step."""
        system_prompt = f"You are a reasoning engine. The user will provide you with context and a task. Perform the task and output only the result.\n\n--- CONTEXT ---\n{context}"
        return self._chat_completion(prompt, system_prompt, model="gpt-4.1-nano") # Use a cheaper model for thoughts

    def generate_response(self, prompt: str, context: str) -> str:
        """Generates the final user-facing response."""
        system_prompt = f"You are a helpful assistant. The user will provide you with context and a final instruction. Fulfill the instruction based *only* on the context provided.\n\n--- CONTEXT ---\n{context}"
        return self._chat_completion(prompt, system_prompt, model="gpt-4.1-mini")

# A global instance that can be imported and used throughout the application
llm_interface = LLMInterface()

class SLMInterface:
    """
    Interface for communicating with a local Small Language Model (SLM)
    served via Ollama.
    """
    def __init__(self, host: str = "http://ollama:11434"):
        self.host = host
        self.model_name = "qwen3:1.7b"
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """
        Generates a standard text response from the local SLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": "5m" # Keep the model loaded
                },
                timeout=60
            )
            response.raise_for_status()
            
            response_data = response.json()
            message_content = response_data.get("message", {}).get("content", "")
            
            if not message_content:
                print("Warning: SLM returned an empty message for text generation.")
            
            return message_content.strip()

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return None
    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict | None:
        """
        Generates a JSON response from the local SLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "format": "json", # This tells Ollama to ensure the output is valid JSON
                    "stream": False,
                    "keep_alive": "1h"
                },
                timeout=60 # 60-second timeout for the generation
            )
            response.raise_for_status()
            
            response_data = response.json()
            message_content = response_data.get("message", {}).get("content", "")
            
            if not message_content:
                print("Error: SLM returned an empty message.")
                return None

            return json.loads(message_content)

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from SLM response: {e}")
            print(f"Raw response: {message_content}")
            return None

# Global instance for the SLM planner
slm_planner = SLMInterface()