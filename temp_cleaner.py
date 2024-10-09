import os
import time
from datetime import datetime, timedelta

# Define the temp directory
temp_dir = r"C:\Users\Matt\AppData\Local\Temp"

# Define the start time as October 7th, 2024 at 00:00:00
start_time = datetime(2024, 10, 7, 0, 0, 0)

# Define the cutoff time as one hour in the past
cutoff_time = datetime.now() - timedelta(hours=1)

c=0
# Iterate over files in the temp directory
for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)

    # Get the file's creation time
    try:
        file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
        
        # Check if the file was created between the start time and the cutoff time
        if (start_time < file_creation_time < cutoff_time) and os.path.isfile(file_path):
            print(f"Deleting: {file_path}")
            os.remove(file_path)  # Delete the file
            c+=1
    except Exception as e:
        print(f"Could not process file {file_path}. Reason: {e}")