from os import listdir, path
from shutil import move

from config import Config
from utils import strip_accents

if __name__ == '__main__':
    conf = Config()

    # 2) Get video names
    names = listdir(conf.DATASET)

    for name in names:
        print(f"Processing {name}")
        stripped_name = strip_accents(name)
        old_dir_path = path.join(conf.DATASET, name)
        new_dir_path = path.join(conf.DATASET, stripped_name)
        move(old_dir_path, new_dir_path)

        for file_name in listdir(new_dir_path):
            stripped_file_name = strip_accents(file_name)
            old_file_path = path.join(new_dir_path, file_name)
            new_file_path = path.join(new_dir_path, stripped_file_name)
            move(old_file_path, new_file_path)
