"""
Test Gemini API connection
"""
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {API_KEY[:20]}..." if API_KEY else "No API key found")

try:
    import google.generativeai as genai
    print("✓ google.generativeai package imported successfully")
    
    if API_KEY:
        genai.configure(api_key=API_KEY)
        print("✓ API configured")
        
        # Test with a simple query
        model = genai.GenerativeModel('gemini-pro')
        print("✓ Model created: gemini-pro")
        
        response = model.generate_content("Say hello in 5 words")
        print(f"✓ API Response: {response.text}")
        print("\n✅ Gemini AI is working correctly!")
    else:
        print("❌ No API key found in .env file")
        
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {str(e)}")
