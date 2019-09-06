import os

def get_list_of_files(src_dir):
    # create a list of file and sub directories
    # names in the given directory
    list_of_files = os.listdir(src_dir)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_files:
        # Create full path
        full_path = os.path.join(src_dir, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
