import numpy as np
import tensorflow as tf
import spiegelib as spgl
import os

# Load training and testing data
trainFeatures = np.load("mfcc_features_scaled.npy")
trainParams = np.load("trainParams.npy").astype(np.float32)  # Convert to float32
testFeatures = np.load("mfcc_test_features_scaled.npy")
testParams = np.load("testParams.npy").astype(np.float32)    # Convert to float32

# Define checkpointing
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Setup other callbacks for training
logger = spgl.estimator.TFEpochLogger()
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

# Instantiate LSTM++ Model with the input shape and callbacks
bi_lstm = spgl.estimator.HwyBLSTM(
    input_shape=trainFeatures.shape[-2:],  # Shape of each training feature sample
    num_outputs=trainParams.shape[-1],     # Number of parameters to predict
    callbacks=[logger, earlyStopping, checkpoint_callback],  # Include checkpoint callback here
    highway_layers=6                       # Number of highway layers
)

# Define batch size
batch_size = 32

# Add training and testing data to the model with specified batch size
bi_lstm.add_training_data(trainFeatures, trainParams, batch_size=batch_size)
bi_lstm.add_testing_data(testFeatures, testParams, batch_size=batch_size)

# Check for the latest checkpoint and load it
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f"Resuming training from checkpoint: {latest_checkpoint}")
    bi_lstm.model.load_weights(latest_checkpoint)
else:
    print("No checkpoint found. Starting training from scratch.")

# Display model summary
bi_lstm.model.summary()

# Train the model
bi_lstm.fit(epochs=300)

# Save the final trained model (optional, since checkpoints are saved during training)
bi_lstm.save_model('./saved_models/simple_fm_bi_lstm_final.h5')

# Plot the training progress
#logger.plot()
