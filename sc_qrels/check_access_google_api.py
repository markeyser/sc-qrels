from dotenv import load_dotenv
import os
from google import genai

# Load the .env file
load_dotenv()

# Read the key from env
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file.")

# Use legacy Client-based initialization
client = genai.Client(api_key=api_key)

# Make test call
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works in a few words"
)

print(response.text)
