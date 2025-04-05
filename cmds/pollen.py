#!/usr/bin/env python3

import requests
import sys
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


API_KEY = os.getenv("POLLEN_API_KEY")
if not API_KEY:
    print("❌ API key not found. Please add 'POLLEN_API_KEY' in your .env file.")
    sys.exit(1)

GEO_URL = "https://nominatim.openstreetmap.org/search"
POLLEN_URL = "https://api.ambeedata.com/latest/pollen/by-lat-lng"


def get_coordinates(city_name):
    """Convert city name to coordinates using Nominatim with required User-Agent."""
    headers = {
        # Adding User-Agent header to avoid rate-limiting issues
        "User-Agent": "cli-tools/1.0 (Lila James; pollen@local.test)"
    }
    params = {"q": city_name, "format": "json"}
    response = requests.get(GEO_URL, params=params, headers=headers)

    try:
        results = response.json()
    except Exception as e:
        print(f"❌ Failed to decode response from geocoder: {e}")
        print(f"↪ Response content: {response.text}")
        return None, None

    if not results:
        print(f"❌ Could not find coordinates for {city_name}")
        return None, None

    return float(results[0]["lat"]), float(results[0]["lon"])


def fetch_pollen_data(lat, lng):
    """Fetch pollen data from Ambee API."""
    headers = {"x-api-key": API_KEY}
    params = {"lat": lat, "lng": lng}
    response = requests.get(POLLEN_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error fetching pollen data: {response.status_code}")
        return {}


def display_pollen_data(data):
    """Display pollen data in a user-friendly format."""
    # DEBUG: show full data structure
    # <-- You can remove this after debugging
    # print(json.dumps(data, indent=2))

    pollen_info = data.get("data", [])[0] if data.get("data") else {}
    if not pollen_info:
        print("❌ No pollen data available.")
        return

    # Accessing the city from the pollen data, if available
    city = pollen_info.get('location', {}).get('city', 'Unknown')
    updated_at = pollen_info.get('updatedAt', 'N/A')

    # Getting the pollen levels for each type
    levels = pollen_info.get("Risk", {})
    counts = pollen_info.get("Count", {})

    print(f"\n📍 Location: {city}")
    print(f"📅 Date: {updated_at}")
    print("\n🌿 Pollen Levels:")
    for pollen_type in ["grass_pollen", "tree_pollen", "weed_pollen"]:
        risk = levels.get(pollen_type, "N/A")
        count = counts.get(pollen_type, "N/A")
        print(
            f"   - {pollen_type.replace('_', ' ').title()}: {count} grains/m³ ({risk})")


def main():
    if len(sys.argv) < 2:
        print("Usage: pollen <City Name>")
        return

    city = " ".join(sys.argv[1:])
    lat, lng = get_coordinates(city)
    if lat is not None and lng is not None:
        data = fetch_pollen_data(lat, lng)
        display_pollen_data(data)


if __name__ == "__main__":
    main()
