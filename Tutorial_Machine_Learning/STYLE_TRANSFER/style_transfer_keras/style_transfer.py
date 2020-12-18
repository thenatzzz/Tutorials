from keras import backend as keras_backend
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, save_img
import os
import numpy as np
import time
from PIL import Image

# MAIN_IMG_PATH = "/d/project/TUTORIAL/MACHINE_LEARNING_TUTORIAL/STYLE_TRANSFER/style_transfer_keras/INPUT_IMAGE/"
# STYLE_IMG_PATH = "/d/project/TUTORIAL/MACHINE_LEARNING_TUTORIAL/STYLE_TRANSFER/style_transfer_keras/STYLE_IMAGE/"
# OUTPUT_IMG_PATH = "/d/project/TUTORIAL/MACHINE_LEARNING_TUTORIAL/STYLE_TRANSFER/style_transfer_keras/OUTPUT_IMAGE/"

MAIN_IMG_FOLDER = "INPUT_IMAGE"
STYLE_IMG_FOLDER = "STYLE_IMAGE"
OUTPUT_IMG_FOLDER = "OUTPUT_IMG"

MAIN_IMG_FILE_NAME = "dog.png"
STYLE_IMG_FILE_NAME = "wave.jpg"
OUTPUT_IMG_FILE_NAME = "output.jpg"
OUTPUT_IMG_FILE_NAME = ""


# MAIN_IMG = MAIN_IMG_PATH + MAIN_IMG_FILE_NAME
# STYLE_IMG = STYLE_IMG_PATH + STYLE_IMG_FILE_NAME
# OUTPUT_IMG = OUTPUT_IMG_PATH + OUTPUT_IMG_FILE_NAME

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

MAIN_IMG = os.path.join(CURRENT_WORKING_DIR,MAIN_IMG_FOLDER,MAIN_IMG_FILE_NAME)
STYLE_IMG = os.path.join(CURRENT_WORKING_DIR,STYLE_IMG_FOLDER,STYLE_IMG_FILE_NAME)
OUTPUT_IMG = os.path.join(CURRENT_WORKING_DIR,OUTPUT_IMG_FOLDER)

HEIGHT_IMG = 512
WIDTH_IMG = 512
SIZE_IMG = (HEIGHT_IMG, WIDTH_IMG)

# MAIN_IMAGE = Image.open(MAIN_IMG)
# cImageSizeOrig = MAIN_IMAGE.size

width, height = load_img(MAIN_IMG).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

main_img_unprocessed = load_img(path= MAIN_IMG, target_size =SIZE_IMG)
# print(main_img_unprocessed)
# print(keras_backend.shape(main_img_unprocessed))
main_img_tensor = img_to_array(main_img_unprocessed)
# print(main_img_array)
# print(keras_backend.shape(main_img_array))
main_img_tensor = np.expand_dims(main_img_tensor,axis=0)
# print(main_img_array)
main_img_var = keras_backend.variable(preprocess_input(main_img_tensor),dtype='float32')
# print(main_img_array)

style_img_unprocessed = load_img(path= STYLE_IMG, target_size=SIZE_IMG)
style_img_tensor = img_to_array(style_img_unprocessed)
style_img_tensor = np.expand_dims(style_img_tensor,axis=0)
style_img_var = keras_backend.variable(preprocess_input(style_img_tensor),dtype='float32')

output_img_init = np.random.randint(256,size =(HEIGHT_IMG,WIDTH_IMG,3)).astype('float64')
# print(output_img_init)
# print(keras_backend.shape(output_img_init))
output_img_init = preprocess_input(np.expand_dims(output_img_init,axis=0))
output_img_placeholder = keras_backend.placeholder(shape=(1,HEIGHT_IMG,WIDTH_IMG,3))


def get_feature_reps(x, layer_names, model):
    """
    Get feature representations of input x for one or more layers in a given model.
    """
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = keras_backend.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = keras_backend.reshape(featRaw, (M_l, N_l))
        featMatrix = keras_backend.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def get_content_loss(F, P):
    cLoss = 0.5*keras_backend.sum(keras_backend.square(F - P))
    return cLoss
def get_Gram_matrix(F):
    G = keras_backend.dot(F, keras_backend.transpose(F))
    return G
def get_style_loss(ws, Gs, As):
    sLoss = keras_backend.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = keras_backend.int_shape(G)[1]
        N_l = keras_backend.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss+= w*0.25*keras_backend.sum(keras_backend.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
    return sLoss
def get_total_loss(gImPlaceholder, alpha=1.0, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    contentLoss = get_content_loss(F, P)
    styleLoss = get_style_loss(ws, Gs, As)
    totalLoss = alpha*contentLoss + beta*styleLoss
    return totalLoss
def calculate_loss(gImArr):
    """
    Calculate total loss using K.function
    """
    if gImArr.shape != (1, WIDTH_IMG, WIDTH_IMG, 3):
        gImArr = gImArr.reshape((1, WIDTH_IMG, HEIGHT_IMG, 3))
        loss_fcn = keras_backend.function([gModel.input], [get_total_loss(gModel.input)])
        return loss_fcn([gImArr])[0].astype('float64')

def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, WIDTH_IMG, HEIGHT_IMG, 3):
        gImArr = gImArr.reshape((1, WIDTH_IMG, HEIGHT_IMG, 3))
    grad_fcn = keras_backend.function([gModel.input],
                          keras_backend.gradients(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad
def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (WIDTH_IMG, HEIGHT_IMG, 3):
        x = x.reshape((WIDTH_IMG, HEIGHT_IMG, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def reprocess_array(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x

# def save_original_size(x, target_size=cImageSizeOrig):
#     xIm = Image.fromarray(x)
#     xIm = xIm.resize(target_size)
#     xIm.save(OUTPUT_IMG)
#     return xIm
def deprocess_image(x):
    if keras_backend.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
from keras.applications import VGG16
from scipy.optimize import fmin_l_bfgs_b

tf_session = keras_backend.get_session()
cModel = VGG16(include_top=False, weights='imagenet', input_tensor=main_img_var)
sModel = VGG16(include_top=False, weights='imagenet', input_tensor=style_img_var)
gModel = VGG16(include_top=False, weights='imagenet', input_tensor=output_img_placeholder)
cLayerName = 'block4_conv2'
sLayerNames = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                ]

P = get_feature_reps(x=main_img_var, layer_names=[cLayerName], model=cModel)[0]
As = get_feature_reps(x=style_img_var, layer_names=sLayerNames, model=sModel)
ws = np.ones(len(sLayerNames))/float(len(sLayerNames))

iterations = 1
x_val = output_img_init.flatten()
start = time.time()

xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                            maxiter=iterations, disp=True)
# xOut = postprocess_array(xopt)
# xIm = save_original_size(xOut)
# print('Image saved')
# end = time.time()
# print ('Time taken: {}'.format(end-start))
img = deprocess_image(xopt.copy())
fname = result_prefix + '_at_iteration_%d.png' % i
save_img(fname, img)
end_time = time.time()
print('Image saved as', fname)
print('Iteration %d completed in %ds' % (i, end_time - start_time))
