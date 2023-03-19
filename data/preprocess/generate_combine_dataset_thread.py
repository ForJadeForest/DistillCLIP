from PIL import Image
import os
import concurrent.futures
from tqdm import tqdm


def process_image(args):
    dataset_name, image_path, save_path = args
    with Image.open(image_path) as img:
        img = img.convert('RGB')  # convert to RGB mode
        img = img.resize((224, 224))
        filename = os.path.splitext(os.path.basename(image_path))[0]
        new_filename = dataset_name + '_' + filename + '.jpg'
        new_path = os.path.join(save_path, new_filename)
        img.save(new_path)


def process_dataset(root_path, dataset_name, save_path):
    image_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('JPEG'):
                image_paths.append(os.path.join(root, file))
    with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=len(image_paths),
                                                                   desc=f"Processing {dataset_name}") as progress:
        for _ in executor.map(process_image, [(dataset_name, path, save_path) for path in image_paths]):
            progress.update()


def preprocess_images(root_paths, dataset_names, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(root_paths)):
        root_path = root_paths[i]
        dataset_name = dataset_names[i]
        process_dataset(root_path, dataset_name, save_path)


root_paths = ['/data/pyz/data/VOC/', '/data/share/pyz/data/imagenet1k/train', '/data/pyz/data/mscoco/train2017']
dataset_names = ['VOC2010', 'imagenet1k', 'mscoco2017']
save_path = '/data/share/pyz/data/combine_data'

preprocess_images(root_paths, dataset_names, save_path)
