import os
import shutil
from collections import Counter


roboflow_dataset_path = "C:\\Users\\Gabriel\\Documents\\ai-football-chartmaker\\yolov8\\datasets\\Football Player Detection Just Number NA.v9i.yolov8"

output_dataset_path = "C:\\Users\\Gabriel\\Documents\\ai-football-chartmaker\\yolov8\\datasets\\FootballFieldNumbers.yolo"


# Creating the 'obj_train_data' directory in the root
obj_train_data_path = os.path.join(output_dataset_path, 'obj_train_data')
os.makedirs(obj_train_data_path, exist_ok=True)

# Function to move and rename files from a source directory to 'obj_train_data'
def move_and_copy_files(source_folder):
    for folder_name in ['images', 'labels']:
        source_dir = os.path.join(source_folder, folder_name)
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(obj_train_data_path, file_name)
            #os.rename(source_file, destination_file)
            shutil.copy2(source_file, destination_file)

# Moving files from 'train', 'test', and 'valid' directories
for folder_name in ['train', 'test', 'valid']:
    folder_path = os.path.join(roboflow_dataset_path, folder_name)
    move_and_copy_files(folder_path)

# File path for train.txt
train_txt_path = os.path.join(output_dataset_path, 'train.txt')

# Writing the paths of all image files to train.txt
with open(train_txt_path, 'w') as file:
    for file_name in os.listdir(obj_train_data_path):
        if file_name.endswith('.jpg'):  # Include only image files
            file_path = f"data/obj_train_data/{file_name}"
            file.write(file_path + '\n')


# File paths for obj.data and obj.names
obj_data_path = os.path.join(output_dataset_path, 'obj.data')
obj_names_path = os.path.join(output_dataset_path, 'obj.names')

# Function to extract class IDs from a label file
def extract_class_ids(label_file_path):
    with open(label_file_path, 'r') as file:
        return [int(line.split()[0]) for line in file]

# Collecting all class IDs from the label files
class_ids = []
for file_name in os.listdir(obj_train_data_path):
    if file_name.endswith('.txt'):  # Only include label files
        label_file_path = os.path.join(obj_train_data_path, file_name)
        class_ids.extend(extract_class_ids(label_file_path))

# Counting the unique class IDs
class_id_counter = Counter(class_ids)

# Number of classes and their IDs
num_classes = len(class_id_counter)
unique_class_ids = sorted(class_id_counter)

# Creating obj.data with the specified content
obj_data_content = (
    f"classes = {num_classes}\n"
    "train = data/train.txt\n"
    "names = data/obj.names\n"
    "backup = backup/\n"
)

with open(obj_data_path, 'w') as file:
    file.write(obj_data_content)

# Creating obj.names with labels from label_0 to label_9
with open(obj_names_path, 'w') as file:
    for i in range(num_classes):
        file.write(f"label_{i}\n")

# Verifying the contents of obj.data and obj.names
with open(obj_data_path, 'r') as file:
    obj_data_content = file.read()

with open(obj_names_path, 'r') as file:
    obj_names_content = file.read()

