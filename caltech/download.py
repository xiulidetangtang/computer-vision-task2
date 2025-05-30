

import os
import zipfile
import tarfile
import shutil
from pathlib import Path

def extract_nested_archives():

    print("=== 处理嵌套压缩文件 ===\n")
    

    base_path = "data/caltech-101"
    tar_gz_file = os.path.join(base_path, "101_ObjectCategories.tar.gz")
    annotations_tar = os.path.join(base_path, "Annotations.tar")
    
    print(f"检查文件:")
    print(f"  TAR.GZ文件: {tar_gz_file} - {'存在' if os.path.exists(tar_gz_file) else '不存在'}")
    print(f"  标注文件: {annotations_tar} - {'存在' if os.path.exists(annotations_tar) else '不存在'}")
    

    if os.path.exists(tar_gz_file):
        print(f"\n开始解压: {tar_gz_file}")
        try:
            with tarfile.open(tar_gz_file, 'r:gz') as tar:

                members = tar.getnames()

                for member in members[:10]:
                    print(f"  {member}")
                if len(members) > 10:
                    print(f"  ... 还有 {len(members) - 10} 个")

                tar.extractall("data/")

            
        except Exception as e:
            print(f"解压失败: {e}")
            return False
    else:
        print("找不到文件")
        return False

    if os.path.exists(annotations_tar):
        print(f"\n解压标注文件: {annotations_tar}")
        try:
            with tarfile.open(annotations_tar, 'r') as tar:
                tar.extractall("data/")
            print("标注文件解压完成!")
        except Exception as e:
            print(f"标注文件解压失败: {e}")
    
    return True

def find_and_organize_dataset():


    possible_paths = [
        "data/101_ObjectCategories"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"找到数据集: {dataset_path}")
            break
    
    if not dataset_path:
        for root, dirs, files in os.walk("data/"):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
            
            if len(dirs) > 50:
                dataset_path = root
                break
    
    if dataset_path:
        standard_path = "data/caltech-101/101_ObjectCategories"
        
        if dataset_path != standard_path:


            os.makedirs(os.path.dirname(standard_path), exist_ok=True)

            if os.path.exists(standard_path):
                shutil.rmtree(standard_path)
            

            try:
                shutil.move(dataset_path, standard_path)
                print("目录重新组织完成")
                dataset_path = standard_path
            except Exception as e:
                print(f"移动失败，尝试复制: {e}")
                try:
                    shutil.copytree(dataset_path, standard_path)
                    dataset_path = standard_path
                except Exception as e2:
                    print(f"复制失败: {e2}")
    
    return dataset_path



def main():

    if not extract_nested_archives():
        print("解压过程失败")
        return

    dataset_path = find_and_organize_dataset()
    
    if not dataset_path:
        print("无法找到有效的数据集目录")
        return



if __name__ == "__main__":
    main()