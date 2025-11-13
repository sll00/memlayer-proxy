import requests
import json

OLLAMA_HOST = "http://ollama:11434"
MODEL_NAME = "qwen3:1.7b"

def simple_ollama_test():
    """
    Sends a very simple prompt to Ollama to check for basic functionality.
    """
    print("--- Starting Simple Ollama Test ---")
    try:
        print(f"Pinging Ollama at {OLLAMA_HOST}...")
        response = requests.get(OLLAMA_HOST, timeout=10)
        response.raise_for_status()
        print(f"Ollama is running: {response.text.strip()}")

        print(f"\nSending simple chat request to model '{MODEL_NAME}'...")
        
        # A very simple, non-JSON request
        chat_payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, who are you?"}],
            "stream": False
        }

        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json=chat_payload,
            timeout=180  # Generous timeout
        )
        response.raise_for_status()
        
        response_data = response.json()
        content = response_data.get("message", {}).get("content", "")
        
        print("\n--- TEST SUCCESS ---")
        print(f"Received response from Ollama: {content}")
        print("--------------------")

    except requests.exceptions.Timeout:
        print("\n--- TEST FAILED ---")
        print("The request to Ollama timed out. The service might be stuck or too slow.")
        print("-------------------")
    except requests.exceptions.RequestException as e:
        print("\n--- TEST FAILED ---")
        print(f"An error occurred while communicating with Ollama: {e}")
        print("-------------------")

if __name__ == "__main__":
    simple_ollama_test()