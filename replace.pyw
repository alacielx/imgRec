import os
import chardet



count = 0

def is_utf8_encoded(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        global count
        count += 1
        print(file_path, result['encoding'] == 'utf-8', count)
        return result['encoding'] == 'utf-8'

def clean_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            print(f"Cleaned: {file_path}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {str(e)}")

def clean_directory(root_dir):
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if is_utf8_encoded(file_path):
                clean_file(file_path)

if __name__ == "__main__":

    target_directory = r"C:\Users\agarza\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorPanel\train"  # Replace with your target directory
    clean_directory(target_directory)
    print("Non-UTF-8 characters removed from files.")
