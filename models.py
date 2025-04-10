from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras import initializers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, Activation, LayerNormalization, Reshape, Attention, BatchNormalization
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import lightgbm as lgb

seed_value = 42

def create_ann(input_shape, task):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    model.add(Dense(32, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    if task == "cradle-3":
        model.add(Dense(3, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif task == "all-7":
        model.add(Dense(7, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn(input_shape, task):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(32, kernel_size=2, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(64, kernel_size=2, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    if task == "cradle-3":
        model.add(Dense(3, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif task == "all-7":
        model.add(Dense(7, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_lstm(input_shape, task):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(48, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value),return_sequences=True))
    model.add(Dropout(0.2))  
    model.add(LSTM(32, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    model.add(Dropout(0.2))
    if task == "cradle-3":
        model.add(Dense(3, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif task == "all-7":
        model.add(Dense(7, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_lstmcnn(input_shape, task):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(48, activation='relu', kernel_initializer=initializers.GlorotUniform(seed=seed_value),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Reshape((48, 1), input_shape=(1, 48)))
    model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(48, 1), kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    if task == "cradle-3":
        model.add(Dense(3, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif task == "all-7":
        model.add(Dense(7, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform(seed=seed_value)))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(x)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)
    o = add([r, shortcut])
    o = Activation('relu')(o)
    return o

def create_tcn(input_shape, task):
    inputs = Input(shape=input_shape)
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=16, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    if task == "cradle-3":
        x = Dense(3, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value))(x)
        model = Model(inputs, x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif task == "all-7":
        x = Dense(7, activation='softmax', kernel_initializer=initializers.GlorotUniform(seed=seed_value))(x)
        model = Model(inputs, x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        x = Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform(seed=seed_value))(x)
        model = Model(inputs, x)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_lr(task):
    max_iter = 1000      
    tol = 1e-4 
    multi_class = "ovr"     
    if task != "slide-2":      
        multi_class = "multinomial"
    model = LogisticRegression(max_iter=max_iter, multi_class=multi_class, solver='lbfgs', tol=tol, random_state=seed_value)
    return model

def create_lda(task):
    model = LDA()
    return model

def create_svm(task):
    model = SVC(kernel='rbf', random_state=seed_value)
    return model

def create_knn(task):
    model = KNeighborsClassifier(n_neighbors=5)
    return model

def create_dt(task):
    model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=seed_value)
    return model











