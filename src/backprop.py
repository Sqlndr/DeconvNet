import os
from tqdm import tqdm
import numpy as np 
import keras
import pandas as pd
from glob import glob
from optparse import OptionParser


parser = OptionParser()


parser.add_option("", "--iter",
                  dest="iter",
                  type=int,
                  help="number of epochs in one learning",
                  default=50)


parser.add_option("", "--epoch",
                  dest="epoch",
                  type=int,
                  help="number of epochs in one learning",
                  default=1)



parser.add_option("", "--batch-size",
                  dest="batch_size",
                  type=int,
                  help="size of the batch size",
                  default=32)



parser.add_option("", "--pixel",
                  dest="pixel",
                  type=int,
                  help="number of pixels in one pixcture",
                  default=32)



parser.add_option("", "--channels",
                  dest="channels",
                  type=int,
                  help="number of channels in one picture",
                  default=3)



parser.add_option("", "--restore-path",
                  dest="restore_path",
                  type=str,
                  help="path to model to restore",
                  default="")


parser.add_option("", "--restore",
                  dest="restore",
                  type=str,
                  help="path to model to restore",
                  default="")



parser.add_option("", "--use-cifar",
                  dest="use_cifar",
                  type=int,
                  help="Use CIFAR-10bool",
                  default=True)


parser.add_option("", "--data-path",
                  dest="data_path",
                  type=str,
                  help="path to model to data",
                  default="./data")


parser.add_option("", "--model",
                  dest="model",
                  type=str,
                  help="model's name",
                  default="model1")


parser.add_option("", "--validation_split",
                  dest="validation_split",
                  type=float,
                  help="optimizer",
                  default=0.1)


parser.add_option("", "--loss",
                  dest="loss",
                  type=str,
                  help="loss funtion",
                  default="categorical_crossentropy")


parser.add_option("", "--method",
                  dest="method",
                  type=str,
                  help="optimizer",
                  default="Adam")


parser.add_option("", "--num-classes",
                  dest="num_classes",
                  type=int,
                  help="optimizer",
                  default=10)


parser.add_option("", "--metric",
                  dest="metric",
                  type=str,
                  help="optimizer",
                  default="accuracy,acc")


parser.add_option("", "--lr",
                  dest="lr",
                  type=float,
                  help="model's Learning Rate",
                  default=0.03)



parser.add_option("", "--decay",
                  dest="decay",
                  type=float,
                  help="model's Decay Rate for Adam Optimizer",
                  default=0.03)


parser.add_option("", "--momentum",
                  dest="momentum",
                  type=float,
                  help="model's Decay Rate for Adam Optimizer",
                  default=0.03)


def generate_dataset(path, sampling, use_cifar):
    if use_cifar:
        return keras.datasets.cifar10.load_data()
    else:
        # Load dataset

        # Divide between train and test set
        pass


def backward_model(method="model1_valid_no_fc", name='block1_conv1'):
    from deconv2D import Deconv2D
    from pool_unpool import MaxPooling2D, UndoMaxPooling2D
    from keras.models import Model
    from keras.layers import Input   

    if method=="model1_valid_no_fc":
        
        if name == 'block2_conv2':
            
            inp = Input(batch_shape = (1, 10, 10, 128))
            x = inp

            x = Deconv2D(64, 3, padding = 'valid', activation = 'relu', name = 'block2_conv2')(x)
            x = Deconv2D(32, 3, padding = 'valid', activation = 'relu', name = 'block2_conv1')(x)

            pos = Input(batch_shape = (1, 14, 14, 32))
            print("x shape:{}".format(x))
            x = UndoMaxPooling2D((1, 28, 28, 32), name = 'block1_pool1')([x, pos])

            x = Deconv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
            x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)

            return Model(inputs = [inp, pos], outputs = x)
         
        elif name == 'block1_pool1':
             
            inp = Input(batch_shape = (1, 14, 14, 32) )
            pos = Input(batch_shape = (1, 14, 14, 32))
            x = UndoMaxPooling2D((1, 28, 28, 32), name = 'block1_pool1')([inp, pos])

            x = Deconv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
            x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)
            return Model(inputs = [inp, pos], outputs = x)
        
        elif name == 'block2_conv1':
             
            inp = Input(batch_shape = (1, 12, 12, 64) )
            x = inp
            x = Deconv2D(32, 3, padding = 'valid', activation = 'relu', name = 'block2_conv1')(x)
            pos = Input(batch_shape = (1, 14, 14, 32))
            print("x shape:{}".format(x))
            x = UndoMaxPooling2D((1, 28, 28, 32), name = 'block1_pool1')([x, pos])
            x = Deconv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
            x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)

            return Model(inputs = [inp, pos], outputs = x)
        
        elif name == 'block1_conv2':
             
            inp = Input(batch_shape = (1, 28, 28, 32) )
            x = inp
            print("x shape:{}".format(x))
            x = Deconv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
            x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)

            return Model(inputs = inp, outputs = x)
        else:
                  
            inp = Input(batch_shape = (1, 30, 30, 16))
            x = inp
            print("x shape:{}".format(x))
            x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)

            return Model(inputs = inp, outputs = x)
 

def forward_model(name='block1_conv1'):  

    from pool_unpool import MaxPooling2D, UndoMaxPooling2D
    from keras.models import Model

    #Let's build a CNN
    from keras.layers import Flatten, BatchNormalization
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import Conv2D, Input   
    from keras.layers import Flatten, Dropout


    inp = Input(shape = (32, 32, 3))
    x = inp
         
    x = Conv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)
    if name == 'block1_conv1':
        return Model(inputs = inp, outputs = x)
    x = Conv2D(32, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
    if name == 'block1_conv2':
        return Model(inputs = inp, outputs = x)
    x, pos1 = MaxPooling2D(name = 'block1_pool')(x)
    if name == 'block1_pool1':
        return Model(inputs = inp, outputs = [x, pos1])
    x = Conv2D(64, 3, padding = 'valid', activation = 'relu', name = 'block2_conv1')(x)
    if name == 'block2_conv1':
        return Model(inputs = inp, outputs = [x, pos1])
    x = Conv2D(128, 3, padding = 'valid', activation = 'relu', name = 'block2_conv2')(x)
    if name == 'block2_conv2':
        return Model(inputs = inp, outputs = [x, pos1])

    return Model(inputs = inp, outputs = [x, pos1])


if __name__=="__main__":

    (args, _) = parser.parse_args()

    input_shape = (args.pixel, args.pixel, args.channels)

    print("input_shape: {}".format(input_shape ))

    (x_train, y_train), (x_test, y_test) = generate_dataset("", "", args.use_cifar)
    
    print("X_shape: {}".format(x_train.shape))

    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    y_test = keras.utils.to_categorical(y_test, args.num_classes)

    shapes = {
            "block1_conv1":(x_test.shape[0], 30, 30, 3),
            "block1_conv2":(x_test.shape[0], 28, 28, 16),
            "block1_pool1":(x_test.shape[0], 28, 28, 32),
            "block2_conv1":(x_test.shape[0], 14, 14, 32),
            "block2_conv2":(x_test.shape[0], 12, 12, 64),
             }
 
    shape_back = {
                "block1_conv1":(1, 28, 28, 16),
                "block1_conv2":(1, 14, 14, 32),
                "block1_pool1":(1, 28, 28, 32),
                "block2_conv1":(1, 12, 12, 64),
                "block2_conv2":(1, 10, 10, 128),
                }
    
    for filename in glob("../restore/{}_{}_{}_{}_{}_{}_{}_iter_49*".format(args.model, args.method, args.lr, args.loss, args.metric.replace(",","_"), args.validation_split, args.batch_size)):

        print("filename: {}".format(filename))
        for name in ['block2_conv2','block2_conv1','block1_conv2','block1_conv1','block1_pool1'] :
            print("#############name: {}".format(name))

            model = forward_model(name=name)
            model.load_weights(filename, by_name = True )
            print("#"*30)
            print("model {}".format(model.weights))
            print(model.summary())
            print("#"*30)

            print((1,)+shapes[name])
            backward = backward_model(name=name)
            backward.load_weights(filename, by_name = True)
            print("#"*30)
            print("backward: {}".format(backward.weights))
            print(backward.summary())
            print("#"*30)

            
            x_back = np.zeros((x_test.shape[0],)+(32, 32, 3))
            for i, x in enumerate(tqdm(x_test)):
                tmp = model.predict(x.reshape((1,)+x.shape))
                print(tmp[0].shape)
                x_back[i] = backward.predict(tmp)

            np.save("../backprop/{}_{}_{}_{}_{}_{}_{}_iter_{}_name_{}".format(args.model, args.method, args.lr, args.loss,
                                                                              args.metric.replace(",","_"), args.validation_split,
                                                                              args.batch_size, filename.split("_")[-1],name),
                    x_back)
