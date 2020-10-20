import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# dataset related variables
num_channels = 1
labelNames = ['speed_limit_20',
              'speed_limit_50',
              'speed_limit_70',
              'no_overtaking',
              'roundabout',
              'priority_road',
              'give_way',
              'stop',
              'road_closed',
              'no_heavy_goods_vehicles',
              'no_entry',
              'obstacles',
              'left_hand_curve',
              'right_hand_curve',
              'keep_straight_ahead',
              'slippery_road',
              'keep_straight_or_turn_right',
              'construction_ahead',
              'rough_road',
              'traffic_lights',
              'school_ahead']
num_classes = len(labelNames)

# save checkpoints, model arch, learning curves
save_fldr = os.path.join(os.getcwd(), 'results')
store_train_df = os.path.join(os.getcwd(), 'train_df.csv')
store_test_df = os.path.join(os.getcwd(), 'test_df.csv')


# verify number of samples with classes binning
def occurenceHistogram(trainY, num_classes):
    #occurencies, classes = np.histogram(trainY,bins=np.arange(11))
    # or plot
    _ = plt.hist(trainY, bins=num_classes, align='mid', rwidth=0.9)
    plt.title('trainY histogram')
    plt.xlabel('classes num')
    plt.ylabel('num occurencies')
    plt.show()

# plot the training loss and accuracy
def plotLearningCurves(histories_callback, save_fldr):

    fold_epochs = len(histories_callback[0].history["accuracy"])
    # plot losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(0, fold_epochs), histories_callback[0].history["loss"], label="train_loss")
    plt.plot(range(0, fold_epochs), histories_callback[0].history["val_loss"], label="val_loss")
    plt.title(f"Train/Val Losses")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_fldr, f"LossPlotFold_CNN.png"))
    # plot accuracies
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(0, fold_epochs), histories_callback[0].history["accuracy"], label="train_acc")
    plt.plot(range(0, fold_epochs), histories_callback[0].history["val_accuracy"], label="val_acc")
    plt.title(f"Train/Val Accuracies")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(save_fldr, f"AccPlotFold_CNN.png"))


# see model performance and actual repartition of misses
def testCNN(model, X, Y, labelNames):

    predicted_classes = model.predict(X)
    predicted_classes = np.argmax(predicted_classes, axis=1)

    print(classification_report(Y, predicted_classes, target_names=labelNames))
    print(confusion_matrix(Y, predicted_classes))


# convert model from sequential to functional
def seq2funcModel(seqModel):

    input_layer = layers.Input(batch_shape=seqModel.layers[0].input_shape)
    prev_layer = input_layer
    for layer in seqModel.layers:
        prev_layer = layer(prev_layer)

    funcmodel = models.Model([input_layer], [prev_layer])

    return funcmodel


def get_data_df(subset_path):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(os.path.join(os.getcwd(), subset_path)):
        for name in files:
            label = int(root.split(os.sep)[-1])
            tmp = name.split(os.sep)[-1]
            tmp = tmp.split('.')[0]
            tmp = tmp.split('_')
            (h, w) = Image.open(os.path.join(root, name)).size
            df = df.append({'country': root.split(os.sep)[-2], 'class': label, 'sample': int(tmp[0]),
                            'sample_k': int(tmp[1]), 'max_dim': max(h, w), 'ID': os.path.join(root, name)},
                           ignore_index=True)
    return df


def get_median_sample_resolution(dataset):
    dim_array = []
    tmp_array = []
    sample = dataset['sample'][0]
    for i, row in dataset.iterrows():
        if sample == row['sample']:
            tmp_array.append(row['max_dim'])
        else:
            dim_array.append(np.median(np.array(tmp_array)))
            tmp_array = [row['max_dim']]
            sample = row['sample']
    dim_array.append(np.median(np.array(tmp_array)))

    print(f"num samples in dataset: {len(dim_array)}")
    print(f"median dim: {np.median(np.array(dim_array))}")

    return int(np.median(np.array(dim_array)))


if __name__ == '__main__':

    # train and test data miss a couple of samples from belgium dataset!
    train_df = pd.read_csv(store_test_df)
    train_df['clean'] = False
    for i, row in train_df.iterrows():
        sample_cond = train_df['sample'] == row['sample']
        class_cond = train_df['class'] == row['class']
        country_cond = train_df['country'] == 'germany'
        if len(train_df[sample_cond & class_cond]) < 3:
            row['clean'] = True
            print(row['ID'])



