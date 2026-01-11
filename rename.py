import os

directory = 'Images/testing'  # Specify the directory where your images are located

# Get a list of all files in the directory
file_list = os.listdir(directory)

# Sort the file list alphabetically
file_list.sort()

# Initialize a counter variable
counter = 1375

# Iterate over each file in the directory
for filename in file_list:
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Modify the extensions if needed
        # Create the new filename using the counter
        new_filename = f"{counter:04d}.jpg"  # You can modify the format as desired

        # Build the full path for the original and new filenames
        original_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(original_path, new_path)

        # Increment the counter for the next iteration
        counter += 1