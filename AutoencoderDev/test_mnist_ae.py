import argparse
from scipy import misc
from keras import load_model


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--modelpath', help='path for saving autoencoder model', type=str, default="/hmewald/Autoencoders/ModelCF/mnist_ae_model.h5")
args = parser.parse_args()

input_x = 243
input_y = 243


def loadMark():
    mark_big = misc.imread("mark_raw.jpeg" ,mode='L')
    mark_res = misc.imresize(mark_big, (input_x, input_y))

    mark_im = mark_res.reshape([1, input_x, input_y, 1])

    return mark_im

model = load_model(args.modelpath)

mark_im = loadMark()

decoded_mark = model.predict(mark_im)
misc.imsave(model_path + "ae_mark.png", np.concatenate([mark_im.reshape([input_x,input_y]), decoded_mark.reshape([input_x,input_y]), mark_im.reshape([input_x,input_y]) - decoded_mark.reshape([input_x,input_y])], axis=0))
