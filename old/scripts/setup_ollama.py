import requests
import json
import time

OLLAMA_HOST = "http://ollama:11434"
MODEL_NAME = "qwen3:1.7b" # As of Nov 2025, this is a hypothetical but plausible name

def check_ollama_ready():
    """Waits for the Ollama service to be responsive."""
    print("Checking if Ollama service is ready...")
    for _ in range(30): # Wait for up to 30 seconds
        try:
            response = requests.get(OLLAMA_HOST)
            if response.status_code == 200:
                print("Ollama service is ready.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Error: Ollama service did not become ready in time.")
    return False

def pull_model_if_not_exists():
    """Pulls the specified model from Ollama if it's not already present."""
    if not check_ollama_ready():
        return

    try:
        # Check if the model already exists
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        
        model_exists = any(model['name'] == MODEL_NAME for model in models)

        if model_exists:
            print(f"Model '{MODEL_NAME}' already exists in Ollama. Skipping pull.")
        else:
            print(f"Model '{MODEL_NAME}' not found. Pulling from Ollama Hub...")
            # This is a streaming request, so we process it line by line
            with requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": MODEL_NAME}, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        body = json.loads(line)
                        if "status" in body:
                            print(body["status"])
                        if body.get("error"):
                            raise Exception(body["error"])
            print(f"Successfully pulled model '{MODEL_NAME}'.")

    except Exception as e:
        print(f"An error occurred while setting up Ollama model: {e}")

if __name__ == "__main__":
    pull_model_if_not_exists()