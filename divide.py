
import os
import os.path as osp
from tqdm import tqdm

import random
import shutil
import importlib

try:
    import logger
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "logger"])
    importlib.import_module("logger")


def mkdir(p, is_remove=False):
    """Create a folder
    Args:
        p: file path. 
        is_remove: whether delete the exist folder or not. Default [False].
    """
    paths = p if isinstance(p, list) else [p]
    for p in paths:
        if is_remove and os.path.exists(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)


def load_nxywh_info(p, sep=' '):
    '''extract a list object from the *.txt file
    Args:
        p: *.txt file path.
        sep: Separators. [Default: ' ']
    Returns:
        A list object. [[*], [*], [*], ...]
    '''
    with open(p) as f:
        info = f.readlines()
    res = [x.strip() for x in info]
    res = [x.split(sep=sep) for x in res]
    return res


def divide_yolo_dataset(path_dict, save_path, keep_neg=True, seed=10086, mode='obb'):
    """path_dict format:
    |- folderA
    |   |- images
    |   |- labels(hbb) / labelTxt(obb)
    |- folderB
    |   |- images
    |   |- labels
    |   ...
    Args:
        path_dict: 
            {
                #folder: tran/val/test
                folderA: [0.8, 0.2, 0.0], 
                folderB: [0.7, 0.2, 0.1], 
                ...,
            }
        keep_neg: whether to reserve the background sample or not.
    """

    assert mode in ['hbb', 'obb'], f"Invalid mode:{mode}!"
    label_name = ''
    if mode == 'hbb':
        label_name = 'labels'
    elif mode == 'obb':
        label_name = 'labelTxt'

    train_flag, val_flag, test_flag = False, False, False

    logger.info(f"Checking the dataset info >>>")
    for path, ratios in path_dict.items():
        tolerance = 1e-10
        assert abs(sum(ratios) - 1.0) < tolerance, f"'train + val + test' must be equal to 1."
        dirs_to_check = ['images', label_name]
        assert all([osp.exists(osp.join(path, subdir)) for subdir in dirs_to_check]), f"'images' and {label_name} folder must be existed in {path}"

        train_ratio, val_ratio, test_ratio = ratios
        if train_ratio > 0:
            train_flag = True
        if val_ratio > 0:
            val_flag = True
        if test_ratio > 0:
            test_flag = True
        if train_flag and val_flag and test_flag:
            break

    random.seed(seed)
    if train_flag:
        dst_img_train_path = osp.join(save_path, 'images', 'train')
        dst_lbl_train_path = osp.join(save_path, label_name, 'train')
        mkdir([dst_img_train_path, dst_lbl_train_path], is_remove=True)
    if val_flag:
        dst_img_val_path =  osp.join(save_path, 'images', 'val')
        dst_lbl_val_path =  osp.join(save_path, label_name, 'val')
        mkdir([dst_img_val_path, dst_lbl_val_path], is_remove=True)
    if test_flag:
        dst_img_test_path =  osp.join(save_path, 'images', 'test')
        dst_lbl_test_path =  osp.join(save_path, label_name, 'test')
        mkdir([dst_img_test_path, dst_lbl_test_path], is_remove=True)

    train_cnt, val_cnt, test_cnt, drop_cnt = 0, 0, 0, 0
    file_list = []
    for path, ratios in path_dict.items():
        train_ratio, val_ratio, test_ratio = ratios
        img_path = osp.join(path, 'images')
        lbl_path = osp.join(path, label_name)
        img_list = sorted(os.listdir(img_path))

        drop_index = []
        for i, img_name in enumerate(img_list):
            lbl_name = osp.splitext(img_name)[0] + '.txt'
            src_lbl_file = osp.join(lbl_path, lbl_name)
            try:
                lbl_info = load_nxywh_info(src_lbl_file)
                if not lbl_info and not keep_neg:
                    logger.warning(f"⚠️ empty label filterd! -> {src_lbl_file}")
                    drop_index.append(i)
            except FileNotFoundError:
                logger.error(f"❌ file not exist! -> {src_lbl_file}")
                drop_index.append(i)
            if img_name in file_list:
                drop_index.append(i)
            else:
                file_list.append(img_name)
        drop_cnt += len(drop_index)
        filtered_img_list = [element for index, element in enumerate(img_list) if index not in drop_index]
        filtered_img_nums = len(filtered_img_list)
        train_img_nums = int(filtered_img_nums * train_ratio)
        if train_ratio + val_ratio == 1.0:
            val_img_nums = filtered_img_nums - train_img_nums
            test_img_nums = 0
        else:
            val_img_nums   = int(filtered_img_nums * val_ratio)
            test_img_nums  = filtered_img_nums - train_img_nums - val_img_nums
        train_cnt += train_img_nums
        val_cnt   += val_img_nums
        test_cnt  += test_img_nums
        train_img_list = filtered_img_list[:train_img_nums]
        if train_ratio + val_ratio == 1.0:
            val_img_list = filtered_img_list[train_img_nums:]
            test_img_list = []
        else:
            val_img_list   = filtered_img_list[train_img_nums: train_img_nums+val_img_nums]
            test_img_list  = filtered_img_list[train_img_nums+val_img_nums:]

        if train_img_list:
            for img_name in tqdm(train_img_list):
                lbl_name = osp.splitext(img_name)[0] + '.txt'
                src_img_file = osp.join(img_path, img_name)
                src_lbl_file = osp.join(lbl_path, lbl_name)
                lbl_info = load_nxywh_info(src_lbl_file)
                dst_img_file = osp.join(dst_img_train_path, img_name)
                dst_lbl_file = osp.join(dst_lbl_train_path, lbl_name)              
                os.symlink(src_img_file, dst_img_file)
                os.symlink(src_lbl_file, dst_lbl_file)

        if val_img_list:
            for img_name in tqdm(val_img_list):
                lbl_name = osp.splitext(img_name)[0] + '.txt'
                src_img_file = osp.join(img_path, img_name)
                src_lbl_file = osp.join(lbl_path, lbl_name)
                lbl_info = load_nxywh_info(src_lbl_file)
                dst_img_file = osp.join(dst_img_val_path, img_name)
                dst_lbl_file = osp.join(dst_lbl_val_path, lbl_name)              
                os.symlink(src_img_file, dst_img_file)
                os.symlink(src_lbl_file, dst_lbl_file)

        if test_img_list:
            print(f"test_img_list = {test_img_list}")
            for img_name in tqdm(test_img_list):
                lbl_name = osp.splitext(img_name)[0] + '.txt'
                src_img_file = osp.join(img_path, img_name)
                src_lbl_file = osp.join(lbl_path, lbl_name)
                lbl_info = load_nxywh_info(src_lbl_file)
                dst_img_file = osp.join(dst_img_test_path, img_name)
                dst_lbl_file = osp.join(dst_lbl_test_path, lbl_name)              
                os.symlink(src_img_file, dst_img_file)
                os.symlink(src_lbl_file, dst_lbl_file)
        
        logger.info(f"{path} done!")

    logger.info("🚀 Results:")
    logger.info(f"  train_cnt: {train_cnt}")
    logger.info(f"  val_cnt  : {val_cnt}")
    logger.info(f"  test_cnt : {test_cnt}")
    logger.info(f"  drop_cnt : {drop_cnt}")
    logger.info("✅ Processing completed!")


if __name__ == '__main__':
    '''
    |- folderA
    |   |- images
    |   |   |- xxx.jpg
    |   |   |- yyy.png
    |   |   |- ...
    |   |- labelTxt
    |   |   |- xxx.txt
    |   |   |- yyy.txt
    |   |   |- ...
    |- folderB
    |   |- images
    |   |- labelTxt
    |   |- ...
    [a, b, c]: train/val/test
    '''

    path_dict = {
        "/path/to/folderA": [0.7, 0.2, 0.1],
        "/path/to/folderB": [0.8, 0.2, 0.0],
        "...": []
    }
    save_path = '/dataset/task_name'
    divide_yolo_dataset(path_dict, save_path, keep_neg=True, seed=10086, mode='obb')