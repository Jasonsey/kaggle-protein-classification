from typing import Tuple, List
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img
from sklearn.preprocessing import MultiLabelBinarizer



def resize(im: Image.Image, size):
    im = im.resize(size, Image.ANTIALIAS)
    return im


def normalized_img(im: Image.Image):
    # # normalize tensor: center on 0., ensure std is 0.1
    # x = np.asarray(im, dtype='float32')
    # x -= x.mean()
    # x /= (x.std() + K.epsilon())
    # x *= 0.1    # 曲线内缩比率为10

    # # clip to [0, 1],
    # x += 0.5    # 曲线移动到0.5为中心
    # x = np.clip(, 0, 1)    # 小于0的都为0，大于0的都为1
    x = np.asarray(im, dtype=K.floatx())
    x /= 255
    return x


def deprocess_img(im: Image.Image, size=(256, 256)):
    im = resize(im, size=size)
    im_array = normalized_img(im)
    return im_array


def init_path(paths: List):
    '''
    创建路径，不存在则新建
    '''
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


def get_array_label(path_row_colors):
    '''根据颜色选择读取的矩阵组合后的维度

    需要输入带有三个参数组合成的tuple
    '''
    path, row, colors = path_row_colors
    img_id, label = row['Id'], tuple([int(i) for i in row['Target'].split()])

    img_paths = [path / (img_id + '_' + color + '.png') for color in colors]
    
    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path)
        img_array = deprocess_img(img, size=(224, 224))
        imgs.append(img_array)
    imgs = np.asarray(imgs)
    imgs = np.transpose(imgs, axes=[1, 2, 0])

    return imgs, label


class MyImageDataGenerator(ImageDataGenerator):
    '''添加从生成器中创建新的flow_from_generator方法
    '''
    def flow_flow_mydataframe(self, **kwargs):
        '''Iterator capable of reading images from a directory on disk
            through a dataframe.
        '''
        return MyDataFrameIterator(**kwargs, image_data_generator=self)


class MyDataFrameIterator(Iterator):
    """Iterator capable of reading images from a directory on disk
        through a dataframe.

    # Arguments
        dataframe: Pandas dataframe containing the filenames of the
                   images in a column and classes in another or column/s
                   that can be fed as raw target data.
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
            if used with dataframe,this will be the directory to under which
            all the images are present.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        x_col: Column in dataframe that contains all the filenames.
        y_col: Column/s in dataframe that has the target data.
        target_size: tuple of integers, dimensions to resize input images to.
        classes: Optional list of strings, names of
            each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
    """

    def __init__(self, dataframe: pd.DataFrame, directory, image_data_generator,
                 x_col="filenames", y_col="class", y_col_split=None,
                 target_size=(256, 256), color_mode='rgb', color_layer='rgby',
                 classes=None,
                 batch_size=32, shuffle=True, seed=None,
                 data_format='channels_last',
                 save_to_dir=None, save_prefix='', save_format='png',
                 subset=None,
                 interpolation='nearest',
                 dtype='float32'):
        super().common_init(image_data_generator,
                            target_size,
                            color_mode,
                            data_format,
                            save_to_dir,
                            save_prefix,
                            save_format,
                            subset,
                            interpolation)
        flag = False
        for layer in list(color_layer):
            if layer in set('rbgy'):
                flag = True
                break
        if not flag:
            raise ValueError('color_layer should be in [rgby]')

        self.image_shape = target_size + (len(color_layer), )
        self.color_layer = color_layer

        if type(x_col) != str:
            raise ValueError("x_col must be a string.")
        self.df = dataframe.drop_duplicates(x_col)
        self.df[x_col] = self.df[x_col].astype(str)
        self.directory = directory
        self.casses = classes

        self.dtype = dtype
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = set()
            for label in self.df[y_col]:
                for la in label.split():
                    classes.add(np.int64(la))
            classes = list(classes)
        self.num_classes = len(classes)

        # Second, build an index of the images.
        self.filenames = []

        file_ids = _list_valid_filenames_id_in_directory(
            directory,
            white_list_formats,
            self.split)

        temp_df = pd.DataFrame({x_col: file_ids}, dtype=str)
        temp_df = self.df.merge(temp_df, how='right', on=x_col)
        temp_df = temp_df.set_index(x_col)
        temp_df = temp_df.dropna()
        self.filenames = temp_df.index.tolist()
        self.df = temp_df.copy()

        labels, label_num = [], []
        labels_append = labels.append
        label_num_extend = label_num.extend
        y_col_split = None if not y_col_split else y_col_split
        for label in self.df[y_col]:
            label_num_extend([int(la) for la in label.split(y_col_split)])
            labels_append([int(la) for la in label.split(y_col_split)])
        mlb = MultiLabelBinarizer()
        self.classes = mlb.fit_transform(labels)

        label_num = np.asarray(label_num)
        label_num_dict = {}
        for cla in classes:
            mask = label_num == cla
            label_num_dict[cla] = np.sum(mask)
        self.label_num_dict = label_num_dict

        self.samples = len(self.filenames)
        if self.num_classes > 0:
            print('Found %d images belonging to %d classes.' %
                  (self.samples, self.num_classes))
        else:
            print('Found %d images.' % self.samples)

        super().__init__(self.samples,
                         batch_size,
                         shuffle,
                         seed)
        print('Batch Size: %s' % self.batch_size)

    def _get_batches_of_transformed_samples(self, index_array, get_valitation=False):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname_id = self.filenames[j]
            x = load_img(self.directory, fname_id,
                         color_layer=self.color_layer,
                         target_size=self.target_size,
                         data_format=self.data_format)
            if not get_valitation:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        batch_y = self.classes[index_array]
        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    
    def get_validation_data(self):
        if self.subset == 'validation':
            split = self.split
        elif self.subset == 'training':
            split = (0, self.split[0])
        else:
            raise ValueError('Subset is None')
        index_array = list(range(int(self.samples * split[0]), int(self.samples * split[1])))
        return self._get_batches_of_transformed_samples(index_array, get_valitation=True)


def _list_valid_filenames_id_in_directory(directory, white_list_formats, split, **kwargs):
    '''列出已有文件的id号
    '''
    directory = Path(directory)

    file_ids = []
    file_ids_append = file_ids.append
    for white_format in white_list_formats:
        pattern = '**/*.' + white_format
        for filename in directory.glob(pattern):
            file_id = filename.stem.split('_')[0]
            file_ids_append(file_id)
    file_ids = list(set(file_ids))

    if split:
        num_ids = len(file_ids)
        start, stop = int(num_ids * split[0]), int(num_ids * split[1])
        file_ids = file_ids[start : stop]
    
    return file_ids


def load_img(directory, fname_id, grayscale=False, color_layer='rgb', target_size=(256, 256), data_format='channels_last'):
    '''Loads an image into PIL format.
    '''
    directory = Path(directory)

    color_dict = dict(r='red', g='green', b='blue', y='yellow')

    arrays = []
    for layer in color_layer:
        path = directory / (fname_id + '_' + color_dict[layer] + '.png')
        img = Image.open(path)
        if target_size:
            img = img.resize(target_size)
        arrays.append(np.asarray(img))
    arrays = np.asarray(arrays, dtype=K.floatx())
    if data_format == 'channels_last':
        arrays = np.transpose(arrays, axes=[1, 2, 0])
    return arrays

    



    

