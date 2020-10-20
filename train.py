from keras import optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from sklearn.model_selection import StratifiedShuffleSplit
from numpy.random import seed
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = InteractiveSession(config=config)

from CNNmodel import *
from train_callbacks import *
from utils import *
from preprocessing import *

seed(1)
tf.random.set_seed(2)


if __name__ == '__main__':

    if not os.path.exists(save_fldr):
        os.mkdir(save_fldr)

    if os.path.exists(store_train_df):
        train_df = pd.read_csv(store_train_df)
    else:
        train_df = get_data_df('traffic_signs_train')
        train_df = train_df.sort_values(by=['class', 'sample', 'sample_k'], ignore_index=True)
        train_df.to_csv(store_train_df)

    if os.path.exists(store_test_df):
        test_df = pd.read_csv(store_test_df)
    else:
        test_df = get_data_df('traffic_signs_test')
        test_df = test_df.sort_values(by=['class', 'sample', 'sample_k'], ignore_index=True)
        test_df.to_csv(store_test_df)

    # occurenceHistogram(train_df[train_df['sample_k'] == 0]['class'].to_numpy(), num_classes)
    # occurenceHistogram(test_df[test_df['sample_k'] == 0]['class'].to_numpy(), num_classes)

    # get median sample dimensions using median(max(h,w))
    dataset = pd.concat([train_df, test_df], ignore_index=True)
    dataset = dataset.sort_values(by=['class', 'sample', 'sample_k'], ignore_index=True)
    median_res = get_median_sample_resolution(dataset)
    input_shape = (median_res, median_res, 1)

    x = train_df[train_df['sample_k'] == 0]['ID'].to_numpy()
    y = train_df[train_df['sample_k'] == 0]['class'].to_numpy()
    x_test = test_df['ID'].to_numpy()
    y_test = test_df['class'].to_numpy()

    # stratified k-fold train/validate split generator
    k_fold = 1
    train_idx, val_idx = [], []
    stratgen = StratifiedShuffleSplit(n_splits=k_fold, test_size=0.2, random_state=0)
    for idx1, idx2 in stratgen.split(np.zeros(len(y)), y):
        train_idx.append(idx1)
        val_idx.append(idx2)

    # train related variables: stratified k-fold
    fold_epochs = 28
    batch_size = 8
    LR = 1e-3
    norm = max_norm(4.0)
    linearValLR = 0.5
    minValLR = 1e-8
    patienceLR = 2

    # apply weights for classes
    class_weight = {0: 1,
                    1: 1,
                    2: 1,
                    3: 1,
                    4: 1,
                    5: 1,
                    6: 1,
                    7: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    12: 1,
                    13: 1,
                    14: 1,
                    15: 1,
                    16: 1,
                    17: 1,
                    18: 1,
                    19: 1,
                    20: 1}


    augmentations = [brightness_dec, brightness_inc, shift, rotation, zoom_in, zoom_out]
    # augmentations = []
    train_generator = ImageDataGenerator(list_IDs=x[train_idx],
                                         sample_k=3,
                                         mode='training',
                                         batch_size=batch_size,
                                         dim=(median_res, median_res),
                                         n_channels=num_channels,
                                         n_classes=num_classes,
                                         aug=augmentations,
                                         shuffle=True)

    val_generator = ImageDataGenerator(list_IDs=x[val_idx],
                                       sample_k=3,
                                       mode='validation',
                                       batch_size=batch_size,
                                       dim=(median_res, median_res),
                                       n_channels=num_channels,
                                       n_classes=num_classes,
                                       shuffle=True)


    # checkpoint_path = './weightsInception{:d}'.format(i) + '-{epoch:02d}-{val_acc:.2f}.hdf5'
    checkpoint_path = os.path.join(save_fldr, f"weights_CNN.hdf5")
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
                                          mode='max')

    # enlist callbacks
    printLRval_callback = LRval()
    reduceLR_callback = ReduceLROnPlateau(monitor='val_loss', factor=linearValLR, patience=patienceLR, min_lr=minValLR)
    callbacks_list = [printLRval_callback, reduceLR_callback, checkpoint_callback]
    # for plotting learning curves
    histories_callback = []

    # model = conv2Dnet(input_shape, num_classes, norm)
    model = conv2Dsimple2(input_shape, num_classes, norm)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=LR), metrics=['accuracy'])
    # model.summary()
    histories_callback.append(model.fit(train_generator,
                                        batch_size=batch_size,
                                        steps_per_epoch=len(x[train_idx]) // batch_size,
                                        epochs=fold_epochs,
                                        validation_data=val_generator,
                                        callbacks=callbacks_list,
                                        verbose=1,
                                        class_weight=class_weight))

    # apply ALL test data
    test_generator = ImageDataGenerator(list_IDs=x_test,
                                        sample_k=3,
                                        mode='testing',
                                        batch_size=len(x_test),
                                        dim=(median_res, median_res),
                                        n_channels=num_channels,
                                        n_classes=num_classes,
                                        shuffle=False)

    # plots train/val loss and accuracy
    plotLearningCurves(histories_callback, save_fldr)

    # confusion matrix, precision, recall and f1 score
    testCNN(model, test_generator, y_test, labelNames)

    # save model arch
    if isinstance(model, Sequential):
        model = seq2funcModel(model)
    plot_model(model, to_file=os.path.join(f"{save_fldr}", 'CNN_arch.png'), show_shapes=True, show_layer_names=False)

