import os
from sklearn.preprocessing import Normalizer
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



def optimize(method="Adam", lr=0.003, decay=0.01, beta_1=0.9, beta_2=0.99, momentum=0.9):

    if method == "Adam":
        #Let's choose the optimizer
        from keras.optimizers import Adam
        optimize = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=None, decay=decay, amsgrad=False)

    elif method == "SGD":
        from keras.optimizers import SGD
        optimize = SGD(lr=lr, decay=decay, momentum=momentum)

    return optimize


def metric(metriques):
    return [metrique for metrique in metriques.split(",")]


def models(model, input_shape, num_classes, restore_path=""):
    if model=="restore":
        assert (len(restore_path) > 0)
        from keras.models import load_model
        model = load_model(restore_path)


    #Let's build a CNN
    from keras.layers import Flatten, BatchNormalization
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers import Conv2D, MaxPooling2D, Input   
    from keras.layers import Flatten, Dropout


    if model=="model1_valid":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', 
                   input_shape=input_shape, name="block1_conv1"),
            Conv2D(32, (3,3), activation='relu', name="block1_conv2"),
            MaxPooling2D(pool_size=(2,2), name="block1_pool1"),
            Conv2D(64, (3,3), activation='relu', name="block2_pool1"),
            Conv2D(128, (3,3), activation='relu', name="block2_pool2"),
            MaxPooling2D(pool_size=(2,2), name="block2_pool1"),
            Flatten(),
            Dense(256,activation='relu', name="fc_1"),
            Dense(num_classes, activation='softmax', name="fc_2")
            ])


    elif model=="model1_valid_no_fc_batch_norm_21":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Dropout(0.3),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.3),
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_no_fc_batch_norm":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_no_fc_0":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape, name="block1_conv1"),
            Dropout(0.15, name="block1_drop1"),
            Conv2D(32, (3,3), activation='relu', name="block1_conv2"),
            MaxPooling2D(pool_size=(2,2), name="block1_pool1"),
            Dropout(0.25, name="block2_drop1"),
            Conv2D(64, (3,3), activation='relu', name="block2_conv1"),
            MaxPooling2D(pool_size=(2,2), name="block2_pool3"),
            Dropout(0.25, name="block2_drop3"),
            Flatten(),
            Dense(num_classes, activation='softmax', name="fc")
            ])


    elif model=="model1_valid_no_fc":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape, name="block1_conv1"),
            Dropout(0.15, name="block1_drop1"),
            Conv2D(32, (3,3), activation='relu', name="block1_conv2"),
            MaxPooling2D(pool_size=(2,2), name="block1_pool1"),
            Dropout(0.25, name="block2_drop1"),
            Conv2D(64, (3,3), activation='relu', name="block2_conv1"),
            Dropout(0.15, name="block2_drop2"),
            Conv2D(128, (3,3), activation='relu', name="block2_conv2"),
            MaxPooling2D(pool_size=(2,2), name="block2_pool3"),
            Dropout(0.25, name="block2_drop3"),
            Flatten(),
            Dense(num_classes, activation='softmax', name="fc")
            ])



    elif model=="model1_valid_no_fc_2":
        model = Sequential([
            Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(256, (3,3), activation='relu'),
            Conv2D(512, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_fc":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(512,activation='relu'),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])



    elif model=="model1_valid_dropout__1":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Dropout(0.2),
            Conv2D(32, (3,3), activation='relu'),
            Dropout(0.3),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Dropout(0.05),
            Conv2D(128, (3,3), activation='relu'),
            Dropout(0.05),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.01),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout__2":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Dropout(0.1),
            Conv2D(32, (3,3), activation='relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Dropout(0.1),
            Conv2D(128, (3,3), activation='relu'),
            Dropout(0.1),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_2":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Dropout(0.5),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_3":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            Dropout(0.5),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_4":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.5),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_5":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Dropout(0.5),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_6":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Dropout(0.5),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_7":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            Dropout(0.5),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid_dropout_8":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.5),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])



    elif model=="model1_valid_dropout_9":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
            ])



    elif model=="model1_valid_5":
        model = Sequential([
            Conv2D(16, (5,5), activation='relu', input_shape=input_shape),
            Conv2D(32, (5,5), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (5,5), activation='relu'),
            Conv2D(128, (5,5), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])



    elif model=="model1_valid_7":
        model = Sequential([
            Conv2D(16, (7,7), activation='relu', input_shape=input_shape),
            Conv2D(32, (7,7), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (7,7), activation='relu'),
            Conv2D(128, (7,7), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1_valid":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu'),
            Conv2D(64, (3,3), activation='relu'),
            Conv2D(128, (3,3), activation='relu'),
            Conv2D(256, (3,3), activation='relu'),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model1":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model2":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])


    elif model=="model3":
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])



    elif model=="model4":
        model = Sequential([
            Conv2D(8, (3,3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(16, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(68, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(256,activation='relu'),
            Dense(num_classes, activation='softmax')
            ])

    return model


def train(X_train, Y_train, X_test, Y_test, model, optimizer, loss, metrics, validation_split, epoch=300, batch_size=128, shuffle=True):

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history = model.fit(X_train, Y_train, batch_size, epoch, validation_data=(X_test, Y_test), shuffle=shuffle)

    return history 



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
        inp = Input(batch_shape = (1, 10, 10, 128))
        x = inp

        x = Deconv2D(64, 3, padding = 'valid', activation = 'relu', name = 'block2_conv2')(x)
        if name == 'block2_conv2':
            return Model(inputs = inp, outputs = x)
        
        x = Deconv2D(32, 3, padding = 'valid', activation = 'relu', name = 'block2_conv1')(x)
        if name == 'block2_conv1':
            return Model(inputs = inp, outputs = x)

        pos2 = Input(batch_shape = (1, 14, 14, 32))
        print("x shape:{}".format(x))
        x = UndoMaxPooling2D((1, 28, 28, 32), name = 'block1_pool1')([x, pos2])
        
        x = Deconv2D(16, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
        if name == 'block2_conv2':
            return Model(inputs = [inp, pos2], outputs = x)
        
        x = Deconv2D(3, 3, padding = 'valid', activation = 'relu', name = 'block1_conv1')(x)
        if name == 'block2_conv2':
            return Model(inputs = [inp, pos2], outputs = x)

        return Model(inputs = [inp, pos2], outputs = x)
 

def forward_model():  

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
    x = Conv2D(32, 3, padding = 'valid', activation = 'relu', name = 'block1_conv2')(x)
    x, pos1 = MaxPooling2D(name = 'block1_pool')(x)
    x = Conv2D(64, 3, padding = 'valid', activation = 'relu', name = 'block2_conv1')(x)
    x = Conv2D(128, 3, padding = 'valid', activation = 'relu', name = 'block2_conv2')(x)

    return Model(inputs = inp, outputs = [x, pos1])


if __name__=="__main__":

    (args, _) = parser.parse_args()

    input_shape = (args.pixel, args.pixel, args.channels)

    print("input_shape: {}".format(input_shape ))

    (x_train, y_train), (x_test, y_test) = generate_dataset("", "", args.use_cifar)
    
    print("X_shape: {}".format(x_train.shape))

    x_train = x_train/255.
    x_test = x_test/255.

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, args.num_classes)
    y_test = keras.utils.to_categorical(y_test, args.num_classes)

    def main():
        model = models(model=args.model, input_shape=input_shape, num_classes=args.num_classes, restore_path=args.restore_path)

        metrics = metric(args.metric)

        print(model.summary())

        for iter in range(args.iter):
            print("epoch:{}".format(args.epoch))

            optimizer =  optimize(method=args.method, lr=args.lr, decay=args.decay)

            history =  train(x_train, y_train, x_test, y_test, model, optimizer, args.loss, metrics , epoch=args.epoch, batch_size=args.batch_size, validation_split=args.validation_split)
            history.history["iter"] = iter

            filename = "../models/{}_{}_{}_{}_{}_{}_{}.h5".format(args.model, args.method, args.lr, args.loss, args.metric.replace(",","_"), args.validation_split, args.batch_size)

            if os.path.isfile(filename):
                data = pd.read_csv(filename)
                data = data.append(pd.DataFrame(history.history), ignore_index=True)
                data.to_csv(filename)
            else:
                pd.DataFrame(history.history).to_csv(filename)
            model.save("../restore/{}_{}_{}_{}_{}_{}_{}_iter_{}.h5".format(args.model, args.method, args.lr, args.loss, args.metric.replace(",","_"), args.validation_split, args.batch_size, iter))
    main()
