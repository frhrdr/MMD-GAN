import tensorflow as tf
import tensorflow_datasets as tfd
from GeneralTools.input_func import binary_image_to_tfrecords


def load_cifar10():
    file_path = 'cifar-10-binary/cifar-10-batches-bin/'
    filename = [file_path + 'data_batch_{}'.format(i) for i in range(1, 5)] + [file_path + 'test_batch']
    binary_image_to_tfrecords(
        filename, 'cifar_NCHW/cifar', 50000, [3, 32, 32], num_labels=1,
        image_format_in_file='NCHW', target_image_format='NCHW', save_label=False)


def load_mnist():
    mnist_ds = tfd.image.mnist.MNIST.as_dataset
    ds2tfrecord(mnist_ds, 'Data/mnist/')


def ds2tfrecord(ds, filepath):
    with tf.python_io.TFRecordWriter(filepath) as writer:
        feat_dict = ds.make_one_shot_iterator().get_next()
        serialized_dict = {name: tf.serialize_tensor(fea) for name, fea in feat_dict.items()}
        with tf.Session() as sess:
            try:
                while True:
                    features = {}
                    for name, serialized_tensor in serialized_dict.items():
                        bytes_string = sess.run(serialized_tensor)
                        bytes_list = tf.train.BytesList(value=[bytes_string])
                        features[name] = tf.train.Feature(bytes_list=bytes_list)
                    # Create a Features message using tf.train.Example.
                    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                    example_string = example_proto.SerializeToString()
                    # Write to TFRecord
                    writer.write(example_string)
            except tf.errors.OutOfRangeError:
                pass



if __name__ == '__main__':
    load_cifar10()
