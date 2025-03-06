import requests
import sys
import os
from dotenv import load_dotenv

# Load API credentials from .env file
#load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# Fetch API credentials from the environment
API_KEY = os.getenv('DICT_API')

# Check if the API_KEY is loaded correctly
if not API_KEY:
    print("❌ Error: API key not found. Please check your .env file.")
    sys.exit(1)

# Merriam-Webster API endpoint
MW_API_URL = "https://www.dictionaryapi.com/api/v3/references/sd3/json/"

def get_definition(word):
    """Fetch word definition from Merriam-Webster API."""
    url = f"{MW_API_URL}{word.lower()}?key={API_KEY}"
    print(f"✨🌙 Looking up the word: {word}... 🌙✨")
    
    # Make the request to the Merriam-Webster API
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        print(f"Response Status Code: {response.status_code}")

        # Check if the response is valid JSON
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            word_data = data[0]  # Use the first definition set
            if 'shortdef' in word_data:
                print(f"\n✨🌙 Definitions for '{word}': 🌙✨")
                for i, definition in enumerate(word_data['shortdef'], 1):
                    print(f"{i}. {definition}")
            else:
                print("❌ No short definitions found.")
        else:
            print("❌ No definitions found.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: Could not decode JSON response. The API might be down or the response is malformed.")
        sys.exit(1)

def main():
    """Extract word from command-line arguments and fetch definition."""
    if len(sys.argv) > 1:
        word = sys.argv[1]  # First argument after the script name
    else:
        print("❌ Error: You need to provide a word to define.")
        sys.exit(1)
    
    get_definition(word)

if __name__ == "__main__":
    main()
