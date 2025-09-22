import requests
import json
import argparse

def read_prompt_file(file_path):
    """Read prompt from a text file, preserving all characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Make API request with prompt from a file')
    parser.add_argument('--model', default="Qwen2.5-7B-Instruct", help='Model to use')
    parser.add_argument('--max_tokens', type=int, default=10, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0, help='Sampling temperature')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--port', type=str)
    args = parser.parse_args()

    # Read prompt from file
    prompt = read_prompt_file(args.prompt)
    
    # API endpoint
    url = f"http://127.0.0.1:{args.port}/v1/completions"

    # Headers
    headers = {
        "Content-Type": "application/json"
    }

    # Request payload
    data = {
        "model": args.model,
        "prompt": prompt,  # Use the prompt from file
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "offline": args.offline
    }

    # Make the request with streaming enabled
    response = requests.post(url, headers=headers, json=data, stream=True)

    # Handle streaming response
    if response.status_code == 200:
        for chunk in response.iter_lines():
            if chunk:
                # Process each chunk
                decoded_chunk = chunk.decode('utf-8')
                # Handle Server-Sent Events format if applicable
                if decoded_chunk.startswith('data: '):
                    content = decoded_chunk[6:]  # Remove 'data: ' prefix
                    try:
                        json_content = json.loads(content)
                        print(json_content)
                    except json.JSONDecodeError:
                        print(content)
                else:
                    print(decoded_chunk)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()

