from utils.neo import main as neo_process
from utils.define import main as define_process
from utils.net_log import main as net_log_process
from utils.sky import main as sky_process
from utils.weather_logger import main as weather_process
import time

def main():
    while True:
        # Call all the processes
        print("Running near earth objects process...")
        neo_process()

        print("Running net_log_process...")
        net_log_process()

        print("Running sky_process...")
        sky_process()

        print("Running weather_process...")
        weather_process()

        # TODO: add sensor data via I2C


        # Wait for 1 hour (3600 seconds) before the next cycle
        print("Waiting for the next cycle (1 hour)...")
        time.sleep(3600)  # 3600 seconds = 1 hour

if __name__ == "__main__":
    main()


