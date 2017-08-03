# -*- coding: utf-8 -*-
import os
import cPickle

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
import matplotlib.pyplot as plt
import pickle
import keras
import sys
sys.path.append('../confusion')
import plotcm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score

name = 'CNN2'

# %%
with open("../RML2016.10a_dict.dat", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
    Xd = pickle.load(xd1)  # , encoding='latin1'
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
# %%
np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
n_examples = X.shape[0]
n_train = n_examples * 0.5  # 对半
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
# %%
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
# %%
dr = 0.5  # dropout rate (%) 卷积层部分  https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d
model = models.Sequential()  # 这里使用keras的序贯模型  https://keras-cn.readthedocs.io/en/latest/models/sequential/
# Reshape [N,2,128] to [N,1,2,128] on input
model.add(Reshape(([1] + in_shp), input_shape=in_shp))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
# 卷积核维度(输出尺度),卷积核的宽度和长度,“valid”代表只进行有效的卷积，即对边界数据不处理,
# 层权重weights的初始化函数
# channels_first corresponds to inputs with shape (batch, channels, height, width).
model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1", init='glorot_uniform',
                 data_format="channels_first"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2), data_format="channels_first"))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", init='glorot_uniform',
                 data_format="channels_first"))
model.add(Dropout(dr))
# 多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), init='he_normal', name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# %%
# Set up some params
epochs = 100  # number of epochs to train on
batch_size = 1024  # training batch size default1024
# %%
filepath = "convmodrecnets_%s_0.5.wts.h5" % (name)  # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
# Show loss curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('%s Training performance' %(name))
# plt.show()

model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('evaluate_score:', score)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)
    # plt.show()


# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)

# %%调用库产生混淆矩阵
pre_labels = []
for x in test_Y_hat:
    tmp = np.argmax(x, 0)
    pre_labels.append(tmp)
true_labels = []
for x in Y_test:
    tmp = np.argmax(x, 0)
    true_labels.append(tmp)

kappa = cohen_kappa_score(pre_labels, true_labels)
oa = accuracy_score(true_labels, pre_labels)
kappa_oa = {}
print('oa_all:', oa)
print('kappa_all:', kappa)
kappa_oa['oa_all'] = oa
kappa_oa['kappa_all'] = kappa
fd = open('results_all_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, kappa_oa), fd)
fd.close()
cnf_matrix = confusion_matrix(true_labels, pre_labels)
# np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
# plt.figure()
plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                             normalize=False,
                             title='%s Confusion matrix, without normalization' % (name), showtext=True)
plt.savefig('%s Confusion matrix, without normalization' % (name))
# Plot normalized confusion matrix
# plt.figure()
plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                             normalize=True,
                             title='%s Normalized confusion matrix' % (name), showtext=True)
plt.savefig('%s Normalized confusion matrix' % (name))
# plt.show()

# %%自定义产生混淆矩阵
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] += 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, labels=classes, title='%s Confusion matrix' % (name))

# %%Plot confusion matrix 画图
acc = {}
kappa_dict = {}
oa_dict = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)

    # %%调用库产生混淆矩阵
    pre_labels_i = []
    for x in test_Y_i_hat:
        tmp = np.argmax(x, 0)
        pre_labels_i.append(tmp)
    true_labels_i = []
    for x in test_Y_i:
        tmp = np.argmax(x, 0)
        true_labels_i.append(tmp)
    kappa = cohen_kappa_score(pre_labels_i, true_labels_i)
    oa = accuracy_score(true_labels_i, pre_labels_i)
    oa_dict[snr] = oa
    kappa_dict[snr] = kappa
    cnf_matrix = confusion_matrix(true_labels_i, pre_labels_i)
    # np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    # plt.figure()
    plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                                 normalize=False,
                                 title='%s Confusion matrix, without normalization (SNR=%d)' % (name, snr), showtext=True)
    plt.savefig('%s Confusion matrix, without normalization (SNR=%d)' % (name, snr))
    # Plot normalized confusion matrix
    # plt.figure()
    plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                                 normalize=True,
                                 title='%s Normalized confusion matrix (SNR=%d)' % (name, snr), showtext=True)
    plt.savefig('%s Normalized confusion matrix (SNR=%d)' % (name, snr))
    # plt.show()

    # %%自定义产生混淆矩阵
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] += 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    # plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="%s Confusion Matrix (SNR=%d)" % (name, snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

# %%Save results to a pickle file for plotting later
print 'acc:', acc
fd = open('results_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, acc), fd)
fd.close()
print('oa:', oa_dict)
fd = open('results_oa_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, oa_dict), fd)
fd.close()
print('kappa:', kappa_dict)
fd = open('results_kappa_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, kappa_dict), fd)
fd.close()

# %%Plot accuracy curve
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("%s Classification Accuracy on RadioML 2016.10 Alpha" % (name))
plt.savefig("%s Classification Accuracy" % (name))
# plt.show()
