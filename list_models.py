import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print("✓ API Key is valid!")
    print("\nAvailable models that support generateContent:\n")
    for model in data.get('models', []):
        if 'generateContent' in model.get('supportedGenerationMethods', []):
            print(f"  - {model['name']}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
