import board
import digitalio
import time
import subprocess

# Set up the GPIO pin connected to the toggle switch (GPIO17 in this case)
switch_pin = digitalio.DigitalInOut(board.D17)  # GPIO 17 (pin 11 on Pi)
switch_pin.direction = digitalio.Direction.INPUT
switch_pin.pull = digitalio.Pull.UP  # Use internal pull-up resistor

# Function to run the "up" script


def run_up_script():
    print("Switch is up! Running the 'up' script...")
    # Adjust path to your script
    subprocess.run(['python3', 'lights.py -s on'])

# Function to run the "down" script


def run_down_script():
    print("Switch is down! Running the 'down' script...")
    # Adjust path to your script
    subprocess.run(['python3', 'lights.py -s off'])


try:
    last_state = switch_pin.value  # Store the initial state of the switch
    while True:
        current_state = switch_pin.value

        # Check if the state has changed
        if current_state != last_state:
            if current_state == False:  # Switch is pressed down (LOW state)
                run_down_script()  # Run the 'down' script
            else:  # Switch is up (HIGH state)
                run_up_script()  # Run the 'up' script

            last_state = current_state  # Update the last state

        time.sleep(0.1)  # Small delay to avoid excessive CPU usage

except KeyboardInterrupt:
    print("Program interrupted")
