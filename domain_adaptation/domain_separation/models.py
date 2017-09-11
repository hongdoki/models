# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains different architectures for the different DSN parts.

We define here the modules that can be used in the different parts of the DSN
model.
- shared encoder (dsn_cropped_linemod, dann_xxxx)
- private encoder (default_encoder)
- decoder (large_decoder, gtsrb_decoder, small_decoder)
"""
import tensorflow as tf

#from models.domain_adaptation.domain_separation
import utils

slim = tf.contrib.slim


def default_batch_norm_params(is_training=False):
  """Returns default batch normalization parameters for DSNs.

  Args:
    is_training: whether or not the model is training.

  Returns:
    a dictionary that maps batch norm parameter names (strings) to values.
  """
  return {
      # Decay for the moving averages.
      'decay': 0.5,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'is_training': is_training
  }


################################################################################
# PRIVATE ENCODERS
################################################################################
def default_encoder(images, code_size, batch_norm_params=None,
                    weight_decay=0.0):
  """Encodes the given images to codes of the given size.

  Args:
    images: a tensor of size [batch_size, height, width, 1].
    code_size: the number of hidden units in the code layer of the classifier.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    end_points: the code of the input.
  """
  end_points = {}
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], padding='SAME'):
      net = slim.conv2d(images, 32, scope='conv1')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
      net = slim.conv2d(net, 64, scope='conv2')
      net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')

      net = slim.flatten(net)
      end_points['flatten'] = net
      net = slim.fully_connected(net, code_size, scope='fc1')
      end_points['fc3'] = net
  return end_points


def svhn_model_encoder(images, code_size, batch_norm_params=None, weight_decay=1e-4):
    """Encodes the given images to codes of the given size using svhn model.

      Args:
        images: a tensor of size [batch_size, height, width, 1].
        code_size: the number of hidden units in the code layer of the classifier.
        batch_norm_params: a dictionary that maps batch norm parameter names to
          values.
        weight_decay: the value for the weight decay coefficient.

      Returns:
        end_points: the code of the input.
    """
    _, code = svhn_model(images=images, code_size=code_size, weight_decay=weight_decay, encoder=True)
    return code


def suanet_encoder(images, code_size, batch_norm_params=None, weight_decay=1e-4):
    """Encodes the given images to codes of the given size using SuaNet.

          Args:
            images: a tensor of size [batch_size, height, width, 1].
            code_size: the number of hidden units in the code layer of the classifier.
            batch_norm_params: a dictionary that maps batch norm parameter names to
              values.
            weight_decay: the value for the weight decay coefficient.

          Returns:
            end_points: the code of the input.
        """
    _, code = suanet(images=images, code_size=code_size, weight_decay=weight_decay, encoder=True)
    return code


def resnet_v2_18_encoder(images, code_size, batch_norm_params=None, weight_decay=1e-4):
    """Encodes the ginve images to codes using ResNet-18"""
    _, code = resnet_v2_18(images, encoder=True, code_size=code_size, weight_decay=weight_decay)
    return code

################################################################################
# DECODERS
################################################################################
def large_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = slim.fully_connected(codes, 600, scope='fc1')
    batch_size = net.get_shape().as_list()[0]
    net = tf.reshape(net, [batch_size, 10, 10, 6])

    net = slim.conv2d(net, 32, [5, 5], scope='conv1_1')

    net = tf.image.resize_nearest_neighbor(net, (16, 16))

    net = slim.conv2d(net, 32, [5, 5], scope='conv2_1')

    net = tf.image.resize_nearest_neighbor(net, (32, 32))

    net = slim.conv2d(net, 32, [5, 5], scope='conv3_2')

    output_size = [height, width]
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv4_1')

  return net


def larger_256_decoder(codes,
                       height,
                       width,
                       channels,
                       batch_norm_params=None,
                       weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = slim.fully_connected(codes, 600, scope='fc1')
    batch_size = net.get_shape().as_list()[0]
    net = tf.reshape(net, [batch_size, 10, 10, 6])

    net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')

    net = tf.image.resize_nearest_neighbor(net, (16, 16))

    net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv2_2')

    net = tf.image.resize_nearest_neighbor(net, (32, 32))
    net = slim.conv2d(net, 32, [3, 3], scope='conv3_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv3_2')

    net = tf.image.resize_nearest_neighbor(net, (64, 64))

    net = slim.conv2d(net, 32, [3, 3], scope='conv4_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv4_2')

    net = tf.image.resize_nearest_neighbor(net, (128, 128))

    net = slim.conv2d(net, 32, [3, 3], scope='conv5_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv5_2')

    output_size = [height, width]
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv6_1')

  return net

def gtsrb_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size. This decoder is specific to GTSRB

  Args:
    codes: a tensor of size [batch_size, 100].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].

  Raises:
    ValueError: When the input code size is not 100.
  """
  batch_size, code_size = codes.get_shape().as_list()
  if code_size != 100:
    raise ValueError('The code size used as an input to the GTSRB decoder is '
                     'expected to be 100.')

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = codes
    net = tf.reshape(net, [batch_size, 10, 10, 1])
    net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')

    # First upsampling 20x20
    net = tf.image.resize_nearest_neighbor(net, [20, 20])

    net = slim.conv2d(net, 32, [3, 3], scope='conv2_1')

    output_size = [height, width]
    # Final upsampling 40 x 40
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, 16, scope='conv3_1')
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv3_2')

  return net


def small_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    net = slim.fully_connected(codes, 300, scope='fc1')
    batch_size = net.get_shape().as_list()[0]
    net = tf.reshape(net, [batch_size, 10, 10, 3])

    net = slim.conv2d(net, 16, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 16, [3, 3], scope='conv1_2')

    output_size = [height, width]
    net = tf.image.resize_nearest_neighbor(net, output_size)

    with slim.arg_scope([slim.conv2d], kernel_size=[3, 3]):
      net = slim.conv2d(net, 16, scope='conv2_1')
      net = slim.conv2d(net, channels, activation_fn=None, scope='conv2_2')

  return net


def svhn_model_decoder(codes,
                  height,
                  width,
                  channels,
                  batch_norm_params=None,
                  weight_decay=0.0):
  """Decodes the codes to a fixed output size using svhn_model.

  Args:
    codes: a tensor of size [batch_size, code_size].
    height: the height of the output images.
    width: the width of the output images.
    channels: the number of the output channels.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    weight_decay: the value for the weight decay coefficient.

  Returns:
    recons: the reconstruction tensor of shape [batch_size, height, width, 3].
  """
  with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(weight_decay)):
      net = slim.fully_connected(codes, 48, scope='fc1')
      batch_size = net.get_shape().as_list()[0]
      net = tf.reshape(net, [batch_size, 4, 4, 3])
      net = tf.image.resize_nearest_neighbor(net, [8, 8])
      net = slim.conv2d(net, 128, [3, 3], scope='conv1_1')
      net = slim.conv2d(net, 128, [3, 3], scope='conv1_2')
      net = slim.conv2d(net, 128, [3, 3], scope='conv1_3')
      net = tf.image.resize_nearest_neighbor(net, [16, 16])
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
      net = tf.image.resize_nearest_neighbor(net, [32, 32])
      net = slim.conv2d(net, 32, [3, 3], scope='conv3_1')
      net = slim.conv2d(net, 32, [3, 3], scope='conv3_2')
      net = slim.conv2d(net, 3, [3, 3], scope='conv3_3')

  return net


################################################################################
# SHARED ENCODERS
################################################################################
def svhn_model(images,
               weight_decay=1e-4,
               prefix='model',
               num_classes=10,
               code_size=128,
               standardization=True,
               encoder=False,
               **kwargs):
  """Creates a convolution SVHN model.

  Note that this model implements the architecture for SVHN proposed in:
   P. Haeusser et al, Learning by Association A versatile semi-supervised traning method for neural networks, CVPR, 2017

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 32, 32, 3].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """
  end_points = {}
  if standardization:
      mean = tf.reduce_mean(images, [1, 2], True)
      std = tf.reduce_mean(tf.square(images - mean), [1, 2], True)
      images = (images - mean) / (std + 1e-5)
  with slim.arg_scope(
          [slim.conv2d, slim.fully_connected],
          activation_fn=tf.nn.elu,
          weights_regularizer=slim.l2_regularizer(weight_decay)):
      end_points['conv1_1'] = slim.conv2d(images, 32, [3, 3], scope='conv1_1')
      end_points['conv1_2'] = slim.conv2d(end_points['conv1_1'], 32, [3, 3], scope='conv1_2')
      end_points['conv1_3'] = slim.conv2d(end_points['conv1_2'], 32, [3, 3], scope='conv1_3')
      end_points['pool1'] = slim.max_pool2d(end_points['conv1_3'], [2, 2], scope='pool1')  # 14
      end_points['conv2_1'] = slim.conv2d(end_points['pool1'], 64, [3, 3], scope='conv2_1')
      end_points['conv2_2'] = slim.conv2d(end_points['conv2_1'], 64, [3, 3], scope='conv2_2')
      end_points['conv2_3'] = slim.conv2d(end_points['conv2_2'], 64, [3, 3], scope='conv2_3')
      end_points['pool2'] = slim.max_pool2d(end_points['conv2_3'], [2, 2], scope='pool2')  # 7
      end_points['conv3_1'] = slim.conv2d(end_points['pool2'], 128, [3, 3], scope='conv3_1')
      end_points['conv3_2'] = slim.conv2d(end_points['conv3_1'], 128, [3, 3], scope='conv3_2')
      end_points['conv3_3'] = slim.conv2d(end_points['conv3_2'], 128, [3, 3], scope='conv3_3')
      end_points['pool3'] = slim.max_pool2d(end_points['conv3_3'], [2, 2], scope='pool3')  # 3
      end_points['pool3_flatten'] = slim.flatten(end_points['pool3'], scope='flatten')


      with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
          end_points['fc1'] = slim.fully_connected(end_points['pool3_flatten'], code_size, scope='fc1')
          if not encoder:
              logits = slim.fully_connected(
                  end_points['fc1'], num_classes, activation_fn=None, scope='fc2',
                  weights_regularizer=slim.l2_regularizer(weight_decay))
              return logits, end_points
          else:
              return None, end_points


def dann_mnist(images,
               weight_decay=0.0,
               prefix='model',
               num_classes=10,
               **kwargs):
  """Creates a convolution MNIST model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the MNIST digits, a tensor of size [batch_size, 28, 28, 1].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """
  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 48, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['pool2']), 100, scope='fc3')
      end_points['fc4'] = slim.fully_connected(
          slim.flatten(end_points['fc3']), 100, scope='fc4')

  logits = slim.fully_connected(
      end_points['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, end_points


def dann_svhn(images,
              weight_decay=0.0,
              prefix='model',
              num_classes=10,
              **kwargs):
  """Creates the convolutional SVHN model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 32, 32, 3].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):

      end_points['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [3, 3], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 64, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [3, 3], 2, scope='pool2')
      end_points['conv3'] = slim.conv2d(
          end_points['pool2'], 128, [5, 5], scope='conv3')

      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['conv3']), 3072, scope='fc3')
      end_points['fc4'] = slim.fully_connected(
          slim.flatten(end_points['fc3']), 2048, scope='fc4')

  logits = slim.fully_connected(
      end_points['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, end_points


def dann_gtsrb(images,
               weight_decay=0.0,
               prefix='model',
               num_classes=43,
               **kwargs):
  """Creates the convolutional GTSRB model.

  Note that this model implements the architecture for MNIST proposed in:
   Y. Ganin et al., Domain-Adversarial Training of Neural Networks (DANN),
   JMLR 2015

  Args:
    images: the GTSRB images, a tensor of size [batch_size, 40, 40, 3].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    **kwargs: Placeholder for keyword arguments used by other shared encoders.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,):
    with slim.arg_scope([slim.conv2d], padding='SAME'):

      end_points['conv1'] = slim.conv2d(images, 96, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 144, [3, 3], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      end_points['conv3'] = slim.conv2d(
          end_points['pool2'], 256, [5, 5], scope='conv3')
      end_points['pool3'] = slim.max_pool2d(
          end_points['conv3'], [2, 2], 2, scope='pool3')

      end_points['fc3'] = slim.fully_connected(
          slim.flatten(end_points['pool3']), 512, scope='fc3')

  logits = slim.fully_connected(
      end_points['fc3'], num_classes, activation_fn=None, scope='fc4')

  return logits, end_points


def dsn_cropped_linemod(images,
                        weight_decay=0.0,
                        prefix='model',
                        num_classes=11,
                        batch_norm_params=None,
                        is_training=False):
  """Creates the convolutional pose estimation model for Cropped Linemod.

  Args:
    images: the Cropped Linemod samples, a tensor of size
      [batch_size, 64, 64, 4].
    weight_decay: the value for the weight decay coefficient.
    prefix: name of the model to use when prefixing tags.
    num_classes: the number of output classes to use.
    batch_norm_params: a dictionary that maps batch norm parameter names to
      values.
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  end_points = {}

  tf.summary.image('{}/input_images'.format(prefix), images)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm if batch_norm_params else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
      end_points['pool1'] = slim.max_pool2d(
          end_points['conv1'], [2, 2], 2, scope='pool1')
      end_points['conv2'] = slim.conv2d(
          end_points['pool1'], 64, [5, 5], scope='conv2')
      end_points['pool2'] = slim.max_pool2d(
          end_points['conv2'], [2, 2], 2, scope='pool2')
      net = slim.flatten(end_points['pool2'])
      end_points['fc3'] = slim.fully_connected(net, 128, scope='fc3')
      net = slim.dropout(
          end_points['fc3'], 0.5, is_training=is_training, scope='dropout')

      with tf.variable_scope('quaternion_prediction'):
        predicted_quaternion = slim.fully_connected(
            net, 4, activation_fn=tf.nn.tanh)
        predicted_quaternion = tf.nn.l2_normalize(predicted_quaternion, 1)
      logits = slim.fully_connected(
          net, num_classes, activation_fn=None, scope='fc4')
  end_points['quaternion_pred'] = predicted_quaternion

  return logits, end_points


def suanet(images,
           num_classes=10,
           encoder=False,
           weight_decay=1e-4,
           prefix='model',
           code_size=256,
           is_training=True,
           batch_norm_params=None):
    """model based on AlexNet with fewer parameter (designed by kilho kim)"""

    from tensorflow.contrib import layers
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib
    from tensorflow.contrib.layers.python.layers import regularizers
    from tensorflow.python.ops import init_ops
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.ops import variable_scope

    def suanet_v2_arg_scope(_weight_decay=weight_decay):
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected],
                activation_fn=nn_ops.relu,
                biases_initializer=init_ops.glorot_normal_initializer(),
                weights_regularizer=regularizers.l2_regularizer(_weight_decay)):
            with arg_scope([layers.conv2d], padding='SAME'):
                with arg_scope([layers_lib.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def suanet_v2(inputs,
                   is_training=True,
                   emb_size=256,
                   scope='suanet_v2'):

        inputs = tf.cast(inputs, tf.float32)

        net = inputs
        mean = tf.reduce_mean(net, [1, 2], True)
        std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
        net = (net - mean) / (std + 1e-5)
        inputs = net

        with variable_scope.variable_scope(scope, 'suanet_v2', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            end_points = {}
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope(
                    [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                    outputs_collections=[end_points_collection]):
                end_points['conv1'] = layers.conv2d(inputs, 96, [11, 11], 4, scope='conv1')
                end_points['pool1'] = layers_lib.max_pool2d(end_points['conv1'], [3, 3], 2, scope='pool1')
                end_points['conv2'] = layers.conv2d(end_points['pool1'], 256, [5, 5], scope='conv2')
                end_points['pool2'] = layers_lib.max_pool2d(end_points['conv2'], [3, 3], 2, scope='pool2')
                end_points['conv3'] = layers.conv2d(end_points['pool2'], emb_size, [3, 3], scope='conv3')
                filter_n_stride_height = end_points['conv3'].get_shape()[1]
                filter_n_stride_width = end_points['conv3'].get_shape()[2]
                end_points['pool3'] = layers_lib.max_pool2d(end_points['conv3'],
                                                            [filter_n_stride_height, filter_n_stride_width],
                                                            [filter_n_stride_height, filter_n_stride_width],
                                                            scope='pool3')
                end_points['flatten'] = slim.flatten(end_points['pool3'], scope='flatten')
        return end_points

    with slim.arg_scope(suanet_v2_arg_scope()):
        if encoder:
            return None, suanet_v2(images, is_training, code_size)
        else:
            end_points = suanet_v2(images, is_training, code_size)
            logits = slim.fully_connected(
                  end_points['flatten'], num_classes, weights_regularizer=slim.l2_regularizer(weight_decay)
                ,activation_fn=None, scope='fc1')
            return logits, end_points


def resnet_v2_18(images,
                 num_classes=2,
                 encoder=False,
                 is_training=True,
                 weight_decay=1e-4,
                 prefix='model',
                 code_size=256):
    """ResNet-18 model"""
    from tensorflow.contrib.slim.nets import resnet_v2 as rv2
    resnet_v2 = rv2.resnet_v2
    resnet_v2_block = rv2.resnet_v2_block

    inputs = tf.cast(images, tf.float32)
    blocks = [
        resnet_v2_block('block1', base_depth=64, num_units=2, stride=2),
        resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
        resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
        resnet_v2_block('block4', base_depth=512, num_units=2, stride=2),
    ]

    net, end_points = resnet_v2(inputs, blocks, num_classes=None,
                                is_training=is_training, global_pool=True,
                                output_stride=None, include_root_block=True,
                                scope='resnet_v2')

    end_points['flatten'] = slim.flatten(net, scope='flatten')
    if encoder:
        return None, end_points
    else:
        logits = slim.fully_connected(
            end_points['flatten'], num_classes, weights_regularizer=slim.l2_regularizer(weight_decay)
            , activation_fn=None, scope='fc1')
        return logits, end_points