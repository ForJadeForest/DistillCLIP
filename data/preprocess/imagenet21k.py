import os
import tarfile
from PIL import Image
from tqdm import tqdm


def resize_images(root_path, save_path):
    error_num = 0
    total_num = 0
    with tqdm(os.listdir(root_path), desc=f"Processing num: {total_num}") as progress:
        for filename in progress:
            if filename.endswith('.tar'):
                with tarfile.open(os.path.join(root_path, filename)) as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            with tar.extractfile(member) as file:
                                # Open image and resize
                                try:
                                    img = Image.open(file)
                                    img = img.resize((224, 224))

                                    # Save resized image to new file location
                                    new_filename = os.path.join(save_path, os.path.basename(member.name))
                                    img.save(new_filename)
                                except:
                                    error_num += 1
                                    continue
                            total_num += 1
            progress.desc = f"Processing num: {total_num}"
    print(f'total: {total_num}\n'
          f'error: {error_num}\n'
          f'correct: {total_num - error_num}\n'
          f'ratio: {error_num / total_num}')


root_path = '/data/share/pyz/data/imagenet_total/winter21_whole'
save_path = os.path.join('/data/share/pyz/data/imagenet_total', 'resize')

if not os.path.exists(save_path):
    os.mkdir(save_path)
resize_images(root_path, save_path)
