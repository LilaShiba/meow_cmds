#!/usr/bin/env python3

import requests
import json

def fetch_sentry_data():
    url = "https://ssd-api.jpl.nasa.gov/sentry.api"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print("Failed to retrieve data:", response.status_code)
        return []

def get_top_closest_asteroids(data, top_n=5):
    min_heap = []
    
    for entry in data:
        try:
            impact_probability = float(entry.get("ip", "N/A"))
            asteroid = {
                "id": entry.get("id", "N/A"),
                "designation": entry.get("des", "N/A"),
                "name": entry.get("fullname", "N/A"),
                "last_observed": entry.get("last_obs", "N/A"),
                "diameter_km": entry.get("diameter", "N/A"),
                "velocity_km_s": entry.get("v_inf", "N/A"),
                "impact_probability": entry.get("ip", "N/A"),
                "number_of_impacts": entry.get("n_imp", "N/A"),
                "hazard_scale": entry.get("ps_cum", "N/A"),
                "risk_period": entry.get("range", "N/A")
            }
            min_heap.append((impact_probability * -1, asteroid))
        except ValueError:
            continue
    
    min_heap.sort(key=lambda x: x[0])
    return [asteroid for _, asteroid in min_heap[:top_n]]

def print_asteroids(asteroids):
    print("\n🚀 Top Closest Asteroids to Earth 🪐\n")
    for i, asteroid in enumerate(asteroids, start=1):
        print(f"{i}. 🪐 {asteroid['name']} ({asteroid['designation']})")
        print(f"   📌 ID: {asteroid['id']}")
        print(f"   📅 Last Observed: {asteroid['last_observed']}")
        print(f"   🔭 Diameter: {asteroid['diameter_km']} km")
        print(f"   💨 Velocity: {asteroid['velocity_km_s']} km/s")
        print(f"   🎯 Impact Probability: {asteroid['impact_probability']}")
        print(f"   🌍 Risk Period: {asteroid['risk_period']}")
        print("   ───────────────────────────")

def main():
    data = fetch_sentry_data()
    top_asteroids = get_top_closest_asteroids(data)
    print_asteroids(top_asteroids)
    
if __name__ == "__main__":
    main()