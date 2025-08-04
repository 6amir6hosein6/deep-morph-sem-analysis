import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import importlib.util

class SEMDataset(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, img_size=(256, 256), shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        assert len(self.image_filenames) == len(self.mask_filenames), "Number of images and masks should be equal"
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_images = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_masks = self.mask_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        images = []
        masks = []

        for img_name, mask_name in zip(batch_images, batch_masks):
            # Load image
            img_path = os.path.join(self.image_dir, img_name)
            img = load_img(img_path, color_mode='grayscale', target_size=self.img_size)
            img = img_to_array(img) / 255.0

            # Load mask
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = load_img(mask_path, color_mode='grayscale', target_size=self.img_size)
            mask = img_to_array(mask) / 255.0
            mask = (mask > 0.5).astype(np.float32)  # Threshold mask to binary

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_filenames, self.mask_filenames))
            np.random.shuffle(temp)
            self.image_filenames, self.mask_filenames = zip(*temp)

def load_model_from_file(arch_file, input_shape):
    spec = importlib.util.spec_from_file_location("model_module", arch_file)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.build_model(input_shape)
    return model

def main(args):
    dataset_dir = args.directory
    arch_file = args.architecture
    batch_size = args.batchsize
    epochs = 30
    img_size = (256, 256)
    input_shape = (img_size[0], img_size[1], 1)  # grayscale image

    image_dir = os.path.join(dataset_dir, 'SEMimage')
    mask_dir = os.path.join(dataset_dir, 'SegmentedImage')

    train_gen = SEMDataset(image_dir, mask_dir, batch_size, img_size)

    model = load_model_from_file(arch_file, input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    n = len(train_gen.image_filenames)
    val_split = int(n * 0.8)

    train_indices = list(range(val_split))
    val_indices = list(range(val_split, n))


    train_gen_on = SEMDataset(image_dir, mask_dir, batch_size, img_size)
    val_gen_on = SEMDataset(image_dir, mask_dir, batch_size, img_size)


    train_gen_on.image_filenames = [train_gen.image_filenames[i] for i in train_indices]
    train_gen_on.mask_filenames = [train_gen.mask_filenames[i] for i in train_indices]
    val_gen_on.image_filenames = [train_gen.image_filenames[i] for i in val_indices]
    val_gen_on.mask_filenames = [train_gen.mask_filenames[i] for i in val_indices]

    history = model.fit(
        train_gen_on,
        validation_data=val_gen_on,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model on SEM dataset")
    parser.add_argument('-d', '--directory', type=str, default='Dataset', help="Dataset root directory")
    parser.add_argument('-a', '--architecture', type=str, required=True, help="Python file defining build_model function")
    parser.add_argument('-b', '--batchsize', type=int, default=8, help="Batch size for training")
    args = parser.parse_args()
    main(args)

