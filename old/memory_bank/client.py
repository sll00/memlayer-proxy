from .storage.sqlite import SQLiteStorage
from .wrappers.openai import OpenAIWrapper

class Memory:
    def __init__(self, api_key: str = None, mode: str = "embedded", storage_path: str = "./memory_bank_data"):
        if mode == "embedded":
            self.storage = SQLiteStorage(storage_path)
            # Other components like SearchService and Consolidator will be initialized here.
        elif mode == "self-hosted":
            # TODO: Initialize API client to connect to Docker container
            pass
        elif mode == "cloud":
            # TODO: Initialize API client to connect to managed service
            pass
        else:
            raise ValueError("Invalid mode specified. Choose from 'embedded', 'self-hosted', or 'cloud'.")

    def wrap(self, llm_client):
        # This is the factory that applies the correct adapter.
        if "openai" in str(type(llm_client)).lower():
            print("OpenAI client detected. Applying memory wrapper.")
            # We pass our initialized components to the wrapper.
            return OpenAIWrapper(llm_client, memory_storage=self.storage)
        # TODO: Add adapters for Anthropic, Ollama, etc.
        else:
            raise TypeError("Unsupported LLM client type. Supported clients: OpenAI.")

    # We can add direct memory access methods here later
    # def search(...)
    # def add(...)