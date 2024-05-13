import threading
import time
import subprocess
import os

# Define the function to run the first code
def run_first_code():
    while True:
        print("Running first code...")
        os.chdir('C:\\Works\\AIs\\Driver Model')
        subprocess.run(['python', 'Drive_model.py'])
        print("First code completed.")
        time.sleep(5)  # Sleep for 1.5 minutes

# Define the function to run the second code
def run_second_code():
    while True:
        os.chdir('C:\\Works\\AIs\\Driver Model\\game')
        print("Running second code...")
        subprocess.run(['python', 'main_myrace_1,9.py'])
        print("Second code completed.")
        os.chdir('C:\\Works\\AIs\\Driver Model')
        time.sleep(5)  # Sleep for 1.5 minutes

# Create two threads, one for each code
thread_first_code = threading.Thread(target=run_first_code)
thread_second_code = threading.Thread(target=run_second_code)

# Start the threads
thread_first_code.start()
thread_second_code.start()

# Join the threads to the main thread
thread_first_code.join()
thread_second_code.join()
