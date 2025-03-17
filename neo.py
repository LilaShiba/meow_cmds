#!/usr/bin/env python3

import requests

# Fetching Sentry data
def fetch_sentry_data():
    url = "https://ssd-api.jpl.nasa.gov/sentry.api"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx responses
        return response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching Sentry data: {e}")
        return []

# Get the top closest asteroids with impact probability
def get_top_closest_asteroids(data, top_n=5):
    asteroids = []
    for entry in data:
        try:
            impact_probability = float(entry.get("ip", -1))
            asteroid = {
                "id": entry.get("id", "N/A"),
                "designation": entry.get("des", "N/A"),
                "name": entry.get("fullname", "N/A"),
                "last_observed": entry.get("last_obs", "N/A"),
                "diameter_km": entry.get("diameter", "N/A"),
                "velocity_km_s": entry.get("v_inf", "N/A"),
                "impact_probability": impact_probability,
                "risk_period": entry.get("range", "N/A")
            }
            asteroids.append(asteroid)
        except ValueError:
            continue  # If we can't parse impact probability, skip this entry
    
    asteroids.sort(key=lambda x: x["impact_probability"], reverse=True)
    return asteroids[:top_n]

# Printing the top closest asteroids
def print_asteroids(asteroids):
    print("\n🚀🌑 Top Closest Asteroids to Earth 🪐💫\n")
    for i, asteroid in enumerate(asteroids, start=1):
        print(f"{i}. 🪐 {asteroid['name']} ({asteroid['designation']})")
        print(f"   📌 ID: {asteroid['id']}")
        print(f"   📅 Last Observed: {asteroid['last_observed']}")
        print(f"   🔭 Diameter: {asteroid['diameter_km']} km")
        print(f"   💨 Velocity: {asteroid['velocity_km_s']} km/s")
        print(f"   🎯 Impact Probability: {asteroid['impact_probability']}")
        print(f"   🌍 Risk Period: {asteroid['risk_period']}")
        print("   ───────────────────────────")

# Main function to fetch and display data
def main():
    print("🌟 Fetching the latest asteroid data from NASA... 🌟")
    data = fetch_sentry_data()
    if data:
        top_asteroids = get_top_closest_asteroids(data)
        print_asteroids(top_asteroids)
    else:
        print("❌ No asteroid data available!")

if __name__ == "__main__":
    main()
