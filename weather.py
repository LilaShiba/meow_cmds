#!/usr/bin/env python3

import requests
import sys

# Weather condition to emoji mapping
WEATHER_EMOJIS = {
    "Clear": "🌙✨",
    "Cloudy": "☁️🌫️",
    "Partly Cloudy": "⛅🌙",
    "Overcast": "🌥️💜",
    "Rain": "🌧️💙",
    "Showers": "🌦️🌊",
    "Thunderstorm": "⛈️⚡",
    "Snow": "❄️☃️",
    "Fog": "🌫️👀",
    "Haze": "🌫️💨",
    "Mist": "🌫️✨",
    "Drizzle": "🌦️💖",
    "Windy": "💨🌪️",
}

def get_weather_emoji(description):
    """Get a corresponding Sailor Moon emoji based on weather description."""
    for condition, emoji in WEATHER_EMOJIS.items():
        if condition.lower() in description.lower():
            return emoji
    return "🌙🌌"  # Default magical moonlight

def fetch_weather_data(station):
    """Fetch and display the latest weather data for the given station with magical styling."""
    try:
        station_url = f"https://api.weather.gov/stations/{station}/observations/latest"
        headers = {"User-Agent": "WeatherScript (your.email@example.com)"}
        response = requests.get(station_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if 'properties' in data:
            properties = data['properties']
            description = properties.get('textDescription', 'Unknown')
            emoji = get_weather_emoji(description)
            temp = properties.get('temperature', {}).get('value', 'N/A')

            print("\n🌙✨ Magic Weather Report ✨🌙\n")
            print(f"📍  Station: {station}")
            print(f"⏳  Timestamp: {properties.get('timestamp', 'N/A')}")
            print(f"🌤️  Weather: {description} {emoji}")
            print(f"🔥  Temperature: {temp} °C")
            print(f"🔥  F: {(temp * 9/5) + 32} °F")

            print(f"💦  Dewpoint: {properties.get('dewpoint', {}).get('value', 'N/A')}°C")
            print(f"💨  Wind Speed: {properties.get('windSpeed', {}).get('value', 'N/A')} km/h")
            print(f"🧭  Wind Direction: {properties.get('windDirection', {}).get('value', 'N/A')}°")
            print(f"💧  Humidity: {properties.get('relativeHumidity', {}).get('value', 'N/A')}%")
            print(f"⚖️   Pressure: {properties.get('barometricPressure', {}).get('value', 'N/A')} Pa")
            print("\n🌟 Stay magical! 🌟\n")
        else:
            print("Error: 'properties' key not found in the response.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather data: {e}")
        sys.exit(1)

def main():
    """Extract station code from command-line arguments."""
    if len(sys.argv) > 1:
        station = sys.argv[1].lstrip("-")  # Strip leading '-'
    else:
        station = "KJFK"  # Default to JFK if no station is provided

    print(f"🔮 Fetching weather data for station: {station}...")
    fetch_weather_data(station)

if __name__ == "__main__":
    main()
