from GeneralTools.input_func import binary_image_to_tfrecords


def load_cifar10():
    file_path = 'cifar-10-binary/cifar-10-batches-bin/'
    filename = [file_path + 'data_batch_{}'.format(i) for i in range(1, 5)] + [file_path + 'test_batch']
    binary_image_to_tfrecords(
        filename, 'cifar_NCHW/cifar', 50000, [3, 32, 32], num_labels=1,
        image_format_in_file='NCHW', target_image_format='NCHW', save_label=False)


if __name__ == '__main__':
    load_cifar10()
