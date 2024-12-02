import os
import subprocess
import time
from datetime import datetime

def get_recent_folders(directory, limit=100):
    """
    Get the most recent folders in a directory, limited to a specified number.
    """
    try:
        items = [(os.path.join(directory, item), os.path.getmtime(os.path.join(directory, item)))
                 for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:limit]]
    except Exception as e:
        print(f"Error while getting recent folders: {e}")
        return []

def cleanup_old_folders(directory, limit=100):
    """
    Ensure the number of folders in a directory does not exceed the limit.
    Delete the oldest folders if necessary.
    """
    try:
        items = [(os.path.join(directory, item), os.path.getmtime(os.path.join(directory, item)))
                 for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
        sorted_items = sorted(items, key=lambda x: x[1])  # Oldest first
        while len(sorted_items) > limit:
            oldest_folder = sorted_items.pop(0)[0]
            print(f"Deleting old folder: {oldest_folder}")
            os.rmdir(oldest_folder)  # Deletes the folder (ensure it is empty)
    except Exception as e:
        print(f"Error while cleaning up folders: {e}")

def process_folders(folders, target_main_folder):
    """
    Check each folder for a subfolder named 'Post processing images' and run postPrep.py if found.
    """
    for folder in folders:
        subfolder_path = os.path.join(folder, "Post processing images")
        if os.path.exists(subfolder_path):
            try:
                # Create a unique target directory in the main folder
                target_directory = os.path.join(target_main_folder, os.path.basename(folder))
                os.makedirs(target_directory, exist_ok=True)
                
                print(f"Processing folder: {subfolder_path} -> Saving to {target_directory}")
                # Call postPrep.py with source folder and target directory as arguments
                subprocess.run(['python', 'postPrep.py', folder, target_directory], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while processing {subfolder_path}: {e}")
            except Exception as e:
                print(f"Unexpected error for {subfolder_path}: {e}")
        else:
            print(f"Subfolder 'Post processing images' not found in: {folder}")

if __name__ == "__main__":
    # Set your target directory and the target main folder for outputs
    source_directory = "path_to_source_directory"  # Replace with your directory path
    target_main_folder = "path_to_target_main_folder"  # Replace with your target main folder path

    # Ensure the target main folder exists
    os.makedirs(target_main_folder, exist_ok=True)

    # Continuous loop for monitoring and processing
    while True:
        try:
            # Get the 100 most recent folders
            recent_folders = get_recent_folders(source_directory)
            
            # Process the folders
            process_folders(recent_folders, target_main_folder)
            
            # Cleanup the target main folder to ensure only 100 folders are retained
            cleanup_old_folders(target_main_folder)

            # Wait for a while before checking again
            time.sleep(10)  # Adjust the sleep interval as needed
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
