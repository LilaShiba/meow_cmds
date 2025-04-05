import os
import argparse
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOVEE_API_KEY")

BASE_URL = "https://developer-api.govee.com/v1"
HEADERS = {
    "Govee-API-Key": API_KEY,
    "Content-Type": "application/json"
}

COLOR_SCHEMES = {
    "cyberpunk": [(255, 0, 255), (0, 255, 255), (0, 0, 255), (255, 20, 147)],
    "candle": [(255, 147, 41), (255, 160, 82), (255, 197, 143), (255, 174, 111)],
    "sunset": [(252, 84, 108), (255, 153, 102), (255, 94, 77), (153, 102, 255)]
}


def get_devices():
    response = requests.get(f"{BASE_URL}/devices", headers=HEADERS)
    response.raise_for_status()
    return response.json().get("data", {}).get("devices", [])


def turn_on(device):
    body = {
        "device": device["device"],
        "model": device["model"],
        "cmd": {"name": "turn", "value": "on"}
    }
    return requests.put(f"{BASE_URL}/devices/control", headers=HEADERS, json=body).ok


def set_color(device, rgb):
    body = {
        "device": device["device"],
        "model": device["model"],
        "cmd": {"name": "color", "value": {"r": rgb[0], "g": rgb[1], "b": rgb[2]}}
    }
    return requests.put(f"{BASE_URL}/devices/control", headers=HEADERS, json=body).ok


def apply_scene(scene):
    if scene not in COLOR_SCHEMES:
        raise ValueError(f"Unknown scene: {scene}")

    devices = get_devices()
    colors = COLOR_SCHEMES[scene]
    print(f"🎨 Applying '{scene}' scene to {len(devices)} light(s).")

    for i, device in enumerate(devices):
        turn_on(device)
        color = colors[i % len(colors)]  # wrap if more lights than colors
        print(f"  → {device['deviceName']} gets {color}")
        set_color(device, color)


def main():
    parser = argparse.ArgumentParser(
        description="Control Govee lights with scenes.")
    parser.add_argument("-s", "--scene", choices=list(COLOR_SCHEMES.keys()), required=True,
                        help="Scene to apply: cyberpunk, candle, sunset")
    args = parser.parse_args()
    apply_scene(args.scene)
