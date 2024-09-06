from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import random
import matplotlib.image as mpimg
from PIL import Image
import tensorflow

url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write('%s%%' % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
        last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
    if force or not os.path.exists(filename):
        print('Попытка загрузить:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print ('\nЗагрузка завершена!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print ('Найдено и проверено', filename)
    else:
        raise Exception(
            'Не удалось проверить' + filename + '. Можете ли вы получить доступ к нему с помощью браузера?'
        )
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10
np.random.seed(133)

def maybe_extract(filename,force=True):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        print('Извлечение данных для %s. Это может занять некоторое время. Пожалуйста, подождите.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    if len(data_folders) != num_classes:
        raise Exception(
            'Ожидаемые папки %d, по одной на класс. Вместо этого я нашел %d.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

def plot_samples(data_folders, sample_size, title=None):
    fig = plt.figure()
    if title: fig.suptitle(title, fontsize=16, fontweight='bold')
    for folder in data_folders:
        image_files = os.listdir(folder)
        image_sample = random.sample(image_files, sample_size)
        for image in image_sample:
            image_file = os.path.join(folder, image)
            ax = fig.add_subplot(len(data_folders), sample_size, sample_size * data_folders.index(folder) +
                                 image_sample.index(image) + 1)
            image = mpimg.imread(image_file)
            ax.imshow(image)
            ax.set_axis_off()
    #plt.show()

plot_samples(train_folders, 10, 'Train Folders')
plot_samples(test_folders, 10, 'Test Folders')
image_size = 28
pixel_depth = 255.0
sys.setrecursionlimit(3000)

def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size), dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (np.asarray(Image.open(image_file)).astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Неожиданная форма изображения: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Не мог читать:', image_file, ':', e, '- все в порядке, пропускаю.')
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Намного меньше изображений, чем ожидалось: %d < %d' %
                        (num_images, min_num_images))
    print('Полный тензор набора данных:', dataset.shape)
    print('Означать:', np.mean(dataset))
    print('Стандартное отклонение:', np.std(dataset))
    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s уже присутствует - маринование пропускаем.' % set_filename)
        else:
            print('Маринование, %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Не удается сохранить данные в', set_filename, ':', e)
    return dataset_names

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

def generate_fake_label(sizes):
    labels = np.ndarray(sum(sizes), dtype=np.int32)
    start = 0
    end = 0
    for label, size in enumerate(sizes):
        start = end
        end += size
        for j in range(start, end):
            labels[j] = label
    return labels

def plot_balance():
    fig, ax = plt.subplots(1, 2)
    bins = np.arange(train_labels.min(), train_labels.max() + 2)
    ax[0].hist(train_labels, bins=bins)
    ax[0].set_xticks((bins[:1] + bins[1:]) / 2, [chr(k) for k in range(ord('A'), ord('J') + 1)])
    ax[1].set_title('Test data')
    #plt.show()

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def balance_check(sizes):
    mean_val = mean(sizes)
    print('среднее значение # изображений :', mean_val)
    for i in sizes:
        if abs(i - mean_val) > 0.1 * mean_val:
            print('Слишком много или меньше изображений')
        else:
            print('Хорошо сбалансированный', i)

def load_and_display_pickle(datasets, sample_size, title=None):
    fig = plt.figure()
    if title: fig.suptitle(title, fontsize=16, fontweight='bold')
    num_of_images = []
    for pickle_file in datasets:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            print('Общее количество изображений в', pickle_file, ':', len(data))
            for index, image in enumerate(data):
                if index == sample_size: break
                ax = fig.add_subplot(len(datasets), sample_size, sample_size * datasets.index(pickle_file) + index + 1)
                ax.imshow(image)
                ax.set_axis_off()
                ax.imshow(image)
            num_of_images.append(len(data))
    balance_check(num_of_images)
    #plt.show()
    return num_of_images

test_labels = generate_fake_label(load_and_display_pickle(test_datasets, 10, 'Test Datasets'))
train_labels = generate_fake_label(load_and_display_pickle(train_datasets, 10, 'Train Datasets'))
plot_balance()


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, trainlabels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes
    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v +=vsize_per_class
                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :,:] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Не удается обработать данные из', pickle_file, ':', e)
            raise
    return  valid_dataset, valid_labels, train_dataset, train_labels

train_size = 200000
valid_size = 10000
test_size = 10000
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
print('Обучение:', train_dataset.shape, train_labels.shape)
print('Проверка:', valid_dataset.shape, valid_labels.shape)
print('Тестирование:', test_dataset.shape, test_labels.shape)

'''
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

randomize(test_datasets, test_labels)

#train_dataset, train_labels = randomize(train_datasets, train_labels)
#test_dataset, test_labels = randomize(test_datasets, test_labels)
#valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
'''

pickle_file = 'notMNIST.pickle'
try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_datasets,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Не удается сохранить данные в', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Размер спрессованного маринованного огурца:', statinfo.st_size)



train_subset = 10000
graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    weights = tf.Variable(tf.truncated_normal([image_size + image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_maan(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)

num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        _, loss, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))






