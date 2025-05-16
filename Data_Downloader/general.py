import os
import datetime

# create an empty folder takes in two strings
def create_directory(path, name):
    dir_path = os.path.join(path, name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        pass

# create an empty folder directory in a date style in a path. Takes in a string
def create_date_directory(path):
    date = datetime.datetime.today()
    y = date.strftime("%Y")
    m = date.strftime("%B")
    d = date.strftime("%d")
    dir_path = os.path.join(path,'collected_data', y, m, d)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        print ('Today\'s data folder has already been created')

# the path to today's date directory, takes in where the "collected_data"
# folder has been saved. Will also create the folder just in case
def save_path(path):
    date = datetime.datetime.today()
    y = date.strftime("%Y")
    m = date.strftime("%B")
    d = date.strftime("%d")
    dir_path = os.path.join(path,'collected_data', y, m, d)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path



# create a file holding all of the previously downloaded xml links
# and the most recent downloaded links
def create_data_files():
    finished = 'Previously_Downloaded.txt'
    recent = 'Most_Recently_Downloaded.txt'
    if not os.path.isfile(finished):
        write_file(finished)
    if not os.path.isfile(recent):
        write_file(recent)

# create an empty file
def write_file(path):
    f = open(path, 'w')
    f.write('')
    f.close()

# store xml data to a path
def store_data(path, data):
    f = open(path,'wb')
    f.write(data)
    f.close()

# add data onto a file
def append_to_file(path,data):
    with open(path, 'a') as file:
        file.write(data + '\n')
        file.close()

# Delete data from a file
def clear_file(path):
    with open(path, 'w') as file:
        file.close()

# Read a file and turn it into a set
def file_to_set(path):
    data = set()
    with open(path, 'rt') as f:
        for line in f:
            data.add(line.replace('\n',''))
        f.close()
    return data

# Take a set and put it into a pre-existing file
def set_to_file(data, file):
    for item in sorted(data):
        append_to_file(file, item)

