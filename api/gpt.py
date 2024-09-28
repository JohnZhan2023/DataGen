import os
import requests
import base64
import json

# Configuration
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
def gpt_4o(img_path, text):
        
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Payload for the request
    payload = {
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
            },
            {
            "type": "text",
            "text": text
            },
        ]
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
    }

    ENDPOINT = "https://zhaohanggpt4v.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"

    # Send request
    # retry logic can be added here
    i = 0
    # if the time is too long, retry
    while i<20:
        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=30)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            break
        except requests.RequestException as e:
            print(f"Failed to make the request. Error: {e}")
            i += 1
            print("Retrying...")
            continue
    
    # Handle the response as needed (e.g., print or process)
    data = response.json()
    print(data["usage"])
    return data['choices'][0]['message']['content']

def gpt_4o_text(text):
        

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Payload for the request
    payload = {
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": text
            },
        ]
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
    }

    ENDPOINT = "https://zhaohanggpt4v.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    # Handle the response as needed (e.g., print or process)
    data = response.json()
    print(data["usage"])
    return data['choices'][0]['message']['content']

