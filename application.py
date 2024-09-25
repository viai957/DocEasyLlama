from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from query_processor import QueryProcessor
from qdrant_client import QdrantClient
import gradio as gr
import logging
from openai_api import OpenAIClient  # Assuming you have an OpenAI API client as before
import json

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for Gradio UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Request body for query processing
class QueryRequest(BaseModel):
    query: str
    collection_name: str
    limit: int = 10

# API Endpoint to process the query
@app.post("/query")
def query_data(request: QueryRequest) -> Dict:
    query_processor = QueryProcessor(client, request.collection_name)
    results = query_processor.process_query(request.query, request.limit)
    return {"results": results}

def fetch_and_generate_response(query: str, collection_name: str, limit: int = 200):
    try:
        # Step 1: Fetch context from Qdrant using the QueryProcessor
        query_processor = QueryProcessor(client, collection_name)
        results = query_processor.process_query(query, limit)  # Increase limit to fetch more chunks

        if not results:
            return "No context available for the provided query."

        # Combine the context
        contexts = [result['content'] for result in results]
        combined_context = " --- ".join(contexts)

        # Ensure the combined context fits within the token limit for the LLM
        combined_context = trim_context_to_fit_limit(contexts)

        # Prepare the system and user messages
        system_prompt = "You are a helpful assistant specialized in API documentation for the given user Query. Build a detailed API help document in Markdown format. MAKE SURE to generate your response in MARKDOWN format for API documentation: List a detailed description of what the API does, its intended use, provide a detailed payload with parameter name, description, type, and provide a sample response for a successful API call along with implementation code."
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"User Query: {query}",
            },
            {
                "role": "system",
                "content": f"Context: {combined_context}",
            }
        ]

        # Step 2: Generate the LLM response
        response = OpenAIClient.generate_llama_response(messages)

        # Log the response for debugging
        logging.info(f"LLM Response: {response}")

        # Check if response contains choices
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["text"]  # Assuming the first choice is the desired one
        else:
            # If the response is not as expected, log and return an error
            logging.error(f"Unexpected LLM response format: {response}")
            return {"error": "Unexpected response format from LLM."}, combined_context

    except Exception as e:
        # Log and return more detailed error information
        logging.error(f"Error while generating LLM response: {e}")
        return {"error": f"Error in LLM response generation: {str(e)}"}
    
def validate_response(response):
    if "choices" not in response or not isinstance(response, dict):
        logging.error(f"Unexpected LLM response format: {response}")
        return {"error": "Unexpected response format from LLM. Please check the query and format."}
    return response

def validate_json(response_text):
    try:
        response_json = json.loads(response_text)
        return response_json
    except ValueError as e:
        print("Response is not valid JSON:", e)
        return None

# Function to trim or summarize the context to fit within a token limit
def trim_context_to_fit_limit(contexts: List[str], token_limit: int = 3500) -> str:
    combined_context = ""
    current_token_count = 0

    for context in contexts:
        token_count = len(context.split())  # Rough estimate of tokens
        if current_token_count + token_count > token_limit:
            break
        combined_context += context + " --- "
        current_token_count += token_count

    return combined_context.strip(" --- ")

# Function to wrap FastAPI into Gradio and use LLM for response generation
def query_gradio(query: str, collection_name: str, limit: int = 10):
    try:
        # Fetch results using query processor
        results, context = fetch_and_generate_response(query, collection_name, limit)

        # Check if the result is a dict (JSON response) and convert it to a formatted string
        if isinstance(results, dict):
            results = json.dumps(results, indent=2)
        elif isinstance(results, list):
            results = "\n".join(results)
        
        return results
    except Exception as e:
        return f"Error in LLM response generation: {str(e)}"

# Gradio UI setup with Markdown support for better formatting
def create_gradio_interface():
    interface = gr.Interface(
        fn=query_gradio,
        inputs=[
            gr.Textbox(label="Query"),
            gr.Dropdown(choices=["test-v12-hybrid", "other-collection"], label="Select Collection Name"),
            gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Limit")
        ],
        outputs=gr.Markdown(label="LLM Response"),
        title="Vector DB Query Interface with LLM Response",
        description="A simple interface to query a Qdrant collection and generate an LLM-based response."
    )
    return interface

def run_gradio():
    gradio_interface = create_gradio_interface()
    gradio_interface.launch(server_name="0.0.0.0", server_port=7860, share=False)

# Start the FastAPI app and Gradio interface
if __name__ == "__main__":
    gradio_interface = create_gradio_interface()
    gradio_interface.launch(server_name="0.0.0.0", server_port=7860)
