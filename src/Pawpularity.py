
sssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sssplit.split(data, data['Pawpularity']):
    training_set = data.iloc[train_index]
    eval_set = data.iloc[test_index]
    
training_set['Pawpularity'].hist(label='Training set')
eval_set['Pawpularity'].hist(label='Eval set')
plt.title('Pawpularity score distribution in training and test set')
plt.xlabel('Pawpularity score')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.show()

training_set['Id'] = training_set['Id'].apply(lambda x: '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'+x+'.jpg')
training_set[['Id', 'Pawpularity']].to_csv('training_set.csv', header=False, index=False)
eval_set['Id'] = eval_set['Id'].apply(lambda x: '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'+x+'.jpg')
eval_set[['Id', 'Pawpularity']].to_csv('eval_set.csv', header=False, index=False)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

path = '/Users/wmeikle/Downloads/petfinder-pawpularity-score/train/'
training_img = os.listdir(path)
rand_idx = np.random.randint(0, len(training_img)-1)
rand_img = training_img[rand_idx]

show_image(path+rand_img)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation=None)
])

model.summary()

tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

%%time

history = model.fit(train_dataset, validation_data=eval_dataset, epochs=25, batch_size=BATCH_SIZE)

def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls='dashed')
        ax[idx].set_xlabel('Epochs')
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history['val_'+metric]);
        ax[idx].legend(['train_'+metric, 'val_'+metric])

training_plot(['loss', 'root_mean_squared_error'], history)

sample_submission = pd.read_csv('/Users/wmeikle/Downloads/petfinder-pawpularity-score/sample_submission.csv')
sample_submission['Id'] = sample_submission['Id'].apply(lambda x: '../input/petfinder-pawpularity-score/test/'+x+'.jpg')
sample_submission.to_csv('sample_submission.csv', index=False, header=False)
sample_submission = tf.data.TextLineDataset(
    'sample_submission.csv'
).map(decode_csv).batch(BATCH_SIZE)

sample_prediction = model.predict(sample_submission)

submission_output = pd.concat(
    [pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv').drop('Pawpularity', axis=1),
    pd.DataFrame(sample_prediction)],
    axis=1
)
submission_output.columns = [['Id', 'Pawpularity']]

submission_output.to_csv('submission.csv', index=False)
