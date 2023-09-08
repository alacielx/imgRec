import os
import re

def rename_files_alphanumeric(root_folder):
    non_alphanumeric = re.compile(r'[^a-zA-Z0-9\-_. ]')
    
    for folderpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            old_path = os.path.join(folderpath, filename)
            new_filename = non_alphanumeric.sub('_', filename)
            new_path = os.path.join(folderpath, new_filename)
            
            if old_path != new_path:
                print(old_path)
                os.rename(old_path, new_path)
                # print(f'Renamed: {old_path} -> {new_path}')

if __name__ == '__main__':
    target_folder = r'C:\Users\alaci\OneDrive - Arrow Glass Industries\Documents\Scripts\Test\imgRec\trainingData\doorType\train'
    rename_files_alphanumeric(target_folder)
