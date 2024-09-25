import requests
from utils import log_error

class OpenAIClient:
    openai_api_base = "https://localhost.com/v1/chat/completions"  # Update the URL for Ollama
    headers = {
        "Content-Type": "application/json"
    }
    
    @staticmethod
    def generate_llama_response(messages: list, model: str = "codellama:34b") -> str:
        """Call Llama 2 API using the specified format and return the response."""
        try:
            # Prepare the payload for the API request
            payload = {
                "model": model,
                "messages": messages
            }

            # Make the API request to the local instance
            response = requests.post(OpenAIClient.openai_api_base, headers=OpenAIClient.headers, json=payload)
            
            # Raise an error if the request failed
            response.raise_for_status()
            
            # Parse the response and return the generated content
            return response.json()["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            log_error(f"Error calling Code Llama API: {str(e)}")
            raise RuntimeError("Failed to get a response from Llama 2 API.")

