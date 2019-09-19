import numpy as np
import tensorflow as tf


class MeshCode(object):
    def __init__(self, code_length, mesh_num=None):
        """ This function creates meshed code for generative models

        :param code_length:
        :param mesh_num:
        :return:
        """
        self.D = code_length
        if mesh_num is None:
            self.mesh_num = (10, 10)
        else:
            self.mesh_num = mesh_num

    def get_batch(self, mesh_mode, name=None):
        if name is None:
            name = 'Z'
        if mesh_mode == 0 or mesh_mode == 'random':
            z_batch = self.by_random(name)
        elif mesh_mode == 1 or mesh_mode == 'sine':
            z_batch = self.by_sine(name)
        elif mesh_mode == 2 or mesh_mode == 'feature':
            z_batch = self.by_feature(name)
        else:
            raise AttributeError('mesh_mode is not supported.')
        return z_batch

    def by_random(self, name=None):
        """ This function generates mesh code randomly

        :param name:
        :return:
        """
        return tf.random_normal(
            [self.mesh_num[0] * self.mesh_num[1], self.D],
            mean=0.0,
            stddev=1.0,
            name=name)

    def by_sine(self, z_support=None, name=None):
        """ This function creates mesh code by interpolating between four supporting codes

        :param z_support:
        :param name: list or tuple of two elements
        :return:
        """
        if z_support is None:
            z_support = tf.random_normal(
                [4, self.D],
                mean=0.0,
                stddev=1.0)
        elif isinstance(z_support, np.ndarray):
            z_support = tf.constant(z_support, dtype=tf.float32)
        z0 = tf.expand_dims(z_support[0], axis=0)  # create 1-by-D vector
        z1 = tf.expand_dims(z_support[1], axis=0)
        z2 = tf.expand_dims(z_support[2], axis=0)
        z3 = tf.expand_dims(z_support[3], axis=0)
        # generate phi and psi from 0 to 90 degrees
        mesh_phi = np.float32(  # mesh_num[0]-by-1 vector
            np.expand_dims(np.pi / 4.0 * np.linspace(0.0, 1.0, self.mesh_num[0]), axis=1))
        mesh_psi = np.float32(
            np.expand_dims(np.pi / 4.0 * np.linspace(0.0, 1.0, self.mesh_num[1]), axis=1))
        # sample instances on the manifold
        z_batch = tf.identity(  # mesh_num[0]*mesh_num[1]-by-1 vector
            kron_by_reshape(  # do kronecker product
                tf.matmul(tf.cos(mesh_psi), z0) + tf.matmul(tf.sin(mesh_psi), z1),
                tf.cos(mesh_phi),
                mat_shape=[self.mesh_num[1], self.D, self.mesh_num[0], 1])
            + kron_by_reshape(
                tf.matmul(tf.cos(mesh_psi), z2) + tf.matmul(tf.sin(mesh_psi), z3),
                tf.sin(mesh_phi),
                mat_shape=[self.mesh_num[1], self.D, self.mesh_num[0], 1]),
            name=name)

        return z_batch

    def by_feature(self, grid=2.0, name=None):
        """ This function creates mesh code by varying a single feature. In this case,
        mesh_num[0] refers to the number of features to mesh, mesh[1] refers to the number
        of variations in one feature

        :param grid:
        :param name: string
        :return:
        """
        mesh = np.float32(  # mesh_num[0]-by-1 vector
            np.expand_dims(np.linspace(-grid, grid, self.mesh_num[1]), axis=1))
        # sample instances on the manifold
        z_batch = kron_by_reshape(  # mesh_num[0]*mesh_num[1]-by-1 vector
            tf.eye(num_rows=self.mesh_num[0], num_columns=self.D),
            tf.constant(mesh),
            mat_shape=[self.mesh_num[0], self.D, self.mesh_num[1], 1])
        # shuffle the columns of z_batch
        z_batch = tf.identity(
            tf.transpose(tf.random_shuffle(tf.transpose(z_batch, perm=[1, 0])), perm=[1, 0]),
            name=name)

        return z_batch

    def simple_grid(self, grid=None):
        """ This function creates simple grid meshes

        Note: this function returns np.ndarray

        :param grid:
        :return:
        """
        if self.D != 2:
            raise AttributeError('Code length has to be two')
        if grid is None:
            grid = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        x = np.linspace(grid[0][0], grid[0][1], self.mesh_num[0])
        y = np.linspace(grid[1][0], grid[1][1], self.mesh_num[1])
        z0 = np.reshape(np.transpose(np.tile(x, (self.mesh_num[1], 1))), [-1, 1])
        z1 = np.reshape(np.tile(y, (1, self.mesh_num[0])), [-1, 1])
        z = np.concatenate((z0, z1), axis=1)

        return z, x, y

    def j_diagram(self, name=None):
        """ This function creates a j diagram using slerp

        This function is not finished as there is a problem with the slerp idea.

        :param name:
        :return:
        """
        raise NotImplementedError('This function has not been implemented.')
        # z_support = np.random.randn(4, self.D)
        # z0 = tf.expand_dims(z_support[0], axis=0)  # create 1-by-D vector
        # z1 = tf.expand_dims(z_support[1], axis=0)
        # z2 = tf.expand_dims(z_support[2], axis=0)
        # pass


def kron_by_reshape(mat1, mat2, mat_shape=None):
    """ This function does kronecker product through reshape and perm

    :param mat1: 2-D tensor
    :param mat2: 2-D tensor
    :param mat_shape: shape of mat1 and mat2
    :return mat3: mat3 = kronecker(mat1, mat2)
    """
    if mat_shape is None:
        a, b = mat1.shape
        c, d = mat2.shape
    else:  # in case of tensorflow, mat_shape must be provided
        a, b, c, d = mat_shape

    if isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray):
        mat3 = np.matmul(np.reshape(mat1, [-1, 1]), np.reshape(mat2, [1, -1]))  # (axb)-by-(cxd)
        mat3 = np.reshape(mat3, [a, b, c, d])  # a-by-b-by-c-by-d
        mat3 = np.transpose(mat3, axes=[0, 2, 1, 3])  # a-by-c-by-b-by-d
        mat3 = np.reshape(mat3, [a * c, b * d])  # (axc)-by-(bxd)
    elif isinstance(mat1, tf.Tensor) and isinstance(mat2, tf.Tensor):
        mat3 = tf.matmul(tf.reshape(mat1, [-1, 1]), tf.reshape(mat2, [1, -1]))  # (axb)-by-(cxd)
        mat3 = tf.reshape(mat3, [a, b, c, d])  # a-by-b-by-c-by-d
        mat3 = tf.transpose(mat3, perm=[0, 2, 1, 3])  # a-by-c-by-b-by-d
        mat3 = tf.reshape(mat3, [a * c, b * d])  # (axc)-by-(bxd)
    else:
        raise AttributeError('Input should be numpy array or tensor')

    return mat3
