from openai import OpenAI
import anthropic
from google.cloud import vertexai

def query_gpt(prompt, model="gpt-3.5-turbo"):
    '''
    prompt: string
    model: string, other examples are "gpt-4o", "gpt-4o-turbo", "gpt-3.5-turbo",
    returns: string
    '''
    client = OpenAI(api_key=OPENAI_KEY)

    chat_completion_det = client.chat.completions.create(
        messages = [
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        model=model,
        logprobs = False,
        temperature = 0.9,
        max_tokens = 1024 # MUST BE MUCH HIGHER TO GENERATE 6000 SAMPLES
    )
    return chat_completion_det.choices[0].message.content

def query_claude(prompt, model="claude-2"):
    # Set your API key
    # CLAUDE_API_KEY = "YOUR_API_KEY" 
    client = anthropic.Client(CLAUDE_KEY)

    # Send a request to Claude
    response = client.completion(
        prompt=prompt,
        model=model,
        max_tokens_to_sample=1024,
    )

    return response.completion


def query_gemini(prompt, model_name="gemini-1.5-flash-001"):
    # Replace with your project ID and location
    location = "us-central1"
    project_id = "synthetic-data-432701"

    # Initialize Vertex AI client
    client = vertexai.Client(project=project_id, location=location)

    # Create a model instance
    model = vertexai.GenerativeModel(model_name=model_name)

    # Generate content using the model
    response = model.generate_content(prompt=prompt)

    return response

