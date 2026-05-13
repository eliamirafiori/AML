import os
import shutil

GLOBAL_PATH = os.getcwd()
TRAIN_CSV_PATH = os.path.join(GLOBAL_PATH,'data','release','train.csv')

SRC_PATH = os.path.join(GLOBAL_PATH, 'src','data','release','images')
DST_PATH_TRAIN = os.path.join(GLOBAL_PATH, 'src','data','release','train_images')
DST_PATH_TEST = os.path.join(GLOBAL_PATH, 'src','data','release','test_images')

os.makedirs(DST_PATH_TRAIN, exist_ok=True)
os.makedirs(DST_PATH_TEST, exist_ok=True)


def get_train_test_dirs():

    if os.path.exists(SRC_PATH):
        for file in os.listdir(SRC_PATH):
            if file.endswith('.jpg'):
                split = file.split('_')[1]
                split2 = split.split('.')[0]
                if '(1)' in split2:
                    continue
                elif int(split2) > int('5000'):
                    src_file_path = os.path.join(SRC_PATH, file)
                    dst_file_path = os.path.join(DST_PATH_TEST, file)
                    shutil.copy(src_file_path, dst_file_path)
                else:
                    src_file_path = os.path.join(SRC_PATH, file)
                    dst_file_path = os.path.join(DST_PATH_TRAIN, file)
                    shutil.copy(src_file_path, dst_file_path)
    else:
        print(f"Source directory '{SRC_PATH}' does not exist.")

    shutil.rmtree(SRC_PATH)