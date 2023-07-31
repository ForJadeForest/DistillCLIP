import os
import shutil
from PIL import Image
from tqdm import tqdm

def move_images(directory_list, target_directory, dataset_name):
    for directory, d_n in zip(directory_list, dataset_name):
        for root, dirs, files in os.walk(directory):
            for file in tqdm(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    # Create the new filename based on the directory structure.
                    file_path = os.path.join(root, file)
                    img = Image.open(file_path).convert('RGB')
                    img_resized = img.resize((224, 224))
                    new_filename = root.replace("/", "-") + "-" + file

                    new_filename = new_filename[new_filename.index(d_n):]

                    # Create the full path for the source and target files.
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_directory, new_filename)

                    # Copy the file to the target directory.
                    # shutil.copy(source_file, target_file)
                    img_resized.save(target_file)

# Example usage:
move_images(['/data/share/pyz/data/mscoco/train2014', '/data/share/pyz/data/imagenet1k/train'], '/home/pyz32/data_tmp/combine_dataset2', ['mscoco', 'imagenet1k'])