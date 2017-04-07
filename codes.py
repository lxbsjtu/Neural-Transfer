import numpy as np
import scipy.misc
import tensorflow as tf
import scipy.io


# use 'imagenet-vgg-verydepp-19.mat' to help compute the CNN
# file link: http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
OUTPUT_PATH = 'out'
LEARNING_RATE = 2

PHOTO_WEIGHT = 8
STYLE_WEIGHT = 150
TV_FLAG = True
TOTAL_VARIATION_WEIGHT = 50
ITERATIONS = 300
CHECKPOINT_BAR = 50

PHOTO_OUTPUT = 'relu4_2'
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')
CNN_layers_dict = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4'
)

def main():

    # input path for the photo image and style image
    PHOTO_PATH = 'image/photo'
    STYLE_PATH = 'image/style'

#remeber to change image_path below
    for ii in range(2):
        #load input content image into photo
        photo = scipy.misc.imread(PHOTO_PATH + str(ii) + '.jpg', False, 'RGB').astype(np.float)
        #load input style image into style
        style = scipy.misc.imread(STYLE_PATH + str(ii) + '.jpg', False, 'RGB').astype(np.float)
        style = scipy.misc.imresize(style, photo.shape[1] / style.shape[1])
        photo_shape = (1,) + photo.shape
        style_shape = (1,) + style.shape

        # Pass the photo to the CNN and get the graph at certain layer
        photo_at_layer = train_CNN(photo, photo_shape, PHOTO_OUTPUT, False)
        # Pass the style image to the CNN and get the Gram matrix at certain layers
        style_Gram_mat = train_CNN(style, style_shape, STYLE_LAYERS, True)
        # Initialize a graph and pass it into CNN for training
        with tf.Graph().as_default():
            initial_graph = tf.random_normal(photo_shape) * 0.256
            image = tf.Variable(initial_graph)

            # Use vgg_net to obtain each layer's output stored in net
            net, mean_image = vgg_net(VGG_PATH, image, CNN_layers_dict)

            # content loss
            diff = net[PHOTO_OUTPUT] - photo_at_layer[PHOTO_OUTPUT]
            photo_loss = PHOTO_WEIGHT * 2 * (tf.nn.l2_loss(diff) / photo_at_layer[PHOTO_OUTPUT].size)
            # style loss
            style_loss = 0
            for layer in STYLE_LAYERS:
                layer_rst = net[layer]
                shape_list = layer_rst.get_shape()
                h = shape_list[1].value
                w = shape_list[2].value
                number_input = shape_list[3].value

                # Flatten the tensor so every row is an image
                layer_rst_transformed = tf.reshape(layer_rst, (-1, number_input))
                # Compute the Gram matrix
                Gram_mat = tf.matmul(tf.transpose(layer_rst_transformed), layer_rst_transformed) / (h * w * number_input)

                diff1 = Gram_mat - style_Gram_mat[layer]
                partial_loss = 2 * tf.nn.l2_loss(diff1) / style_Gram_mat[layer].size
                style_loss += STYLE_WEIGHT * partial_loss

            total_variation_loss = 0
            # If use total variation loss, then compute the loss
            if TV_FLAG:

                image_h_num = sizeOfTensor(image[:, 1:, :, :])
                image_w_num = sizeOfTensor(image[:, :, 1:, :])

                image_h_loss = tf.nn.l2_loss(image[:, 1:,:,:] - image[:, :photo_shape[1] - 1, :, :]) / image_h_num
                image_w_loss = tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :photo_shape[2] - 1, :]) / image_w_num

                total_variation_loss = 2 * TOTAL_VARIATION_WEIGHT * (image_h_loss + image_w_loss)

            # compute the total loss by adding photo loss and style loss
            loss = photo_loss + style_loss + total_variation_loss

            # Use Adam Optimizer to start training process
            TRAIN = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

            best_loss = float('inf')
            best = None
            with tf.Session() as s:
                s.run(tf.initialize_all_variables())
                for i in range(ITERATIONS):
                    if i == ITERATIONS - 1:
                        last_step = True
                    else:
                        last_step = False

                    # Print loss info during processing
                    print('Iteration %d/%d' % (i + 1, ITERATIONS))
                    print('total loss: %f' % loss.eval())

                    TRAIN.run()

                    if last_step or (CHECKPOINT_BAR and i % CHECKPOINT_BAR == 0):
                        current_loss = loss.eval()
                        if current_loss < best_loss:
                            best_loss = current_loss
                            best = image.eval()
                        # Save the output image into output directory
                        image_path = PHOTO_PATH[6:] + str(ii)+ '_' + OUTPUT_PATH + '_' + str(i) + '.jpg'
                        # image_path = '/final/out' + str(CHECKPOINT_BAR) + '.jpg'
                        target = best.reshape(photo_shape[1:]) + mean_image
                        scipy.misc.imsave(image_path, np.clip(target, 0, 255).astype(np.uint8))


def sizeOfTensor(tensor):
    rst = 1
    for dim in tensor.get_shape():
        rst *= dim.value
    return rst
def vgg_net(data_path, input_image, CNN_layers_dict):

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_image = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(CNN_layers_dict):
        kind = name[:4]
        # Convolution layer
        if kind == 'conv':
            temp_weights, bias = weights[i][0][0][0][0]
            temp_weights = np.transpose(temp_weights, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            temp = tf.nn.conv2d(current, tf.constant(temp_weights), strides=(1, 1, 1, 1), padding='SAME')
            current = tf.nn.bias_add(temp, bias)
        # Pooling layer
        elif kind == 'pool':
            # Use average pooling can get better results
            current = tf.nn.avg_pool(current, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        # Relu layer
        elif kind == 'relu':
            current = tf.nn.relu(current)

        net[name] = current
    return net, mean_image

def train_CNN(image, shape, layers, Gram):
    graph = tf.Graph()
    with graph.as_default(), graph.device('/cpu:0'), tf.Session() as s:
        cnn_input = tf.placeholder('float', shape=shape)
        net, mean_image = vgg_net(VGG_PATH, cnn_input, CNN_layers_dict)
        image_processed = np.array([image - mean_image])
        rst = {}
        if not Gram:
            rst[layers] = net[layers].eval(feed_dict={cnn_input: image_processed})
        else:
            for layer in layers:
                layer_rst = net[layer].eval(feed_dict={cnn_input: image_processed})

                # Transform 4D tensor into 2D tensor
                layer_rst_transformed = np.reshape(layer_rst, (-1, layer_rst.shape[3]))
                rst[layer] = np.matmul(layer_rst_transformed.T, layer_rst_transformed) / layer_rst_transformed.size
        return rst

if __name__ == '__main__':
    main()
