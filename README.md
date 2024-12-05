# trichechus
Sound Matching in the Fred's Lab Manatee Spectral Synthesizer

The aim of this project was to explore an interactive machine learning (IML) approach to parameter prediction and sound matching, using the [Fred's Lab Manatee](https://fredslab.net/en/manatee-module.php) spectral synthesizer, released in Summer 2024. 

## Script functionality overview
### calculate_combinations.py
Reads a CSV file of parameters and possible values, calculating the total number of possible combinations for distinct synthesizer settings.

### generate_dataset.py
Cycles through selected MIDI parameters, sending MIDI commands to the Manatee, recording corresponding audio outputs with filenames encoded with pitch and parameter values to create a dataset for training​.

### training_feature_analysis.py
Extracts and scales Mel-frequency cepstral coefficient (MFCC) and Short-time Fourier transform (STFT) features from training audio files, scales the MFCC features, and saves both the raw and scaled features, along with the scaler used, to ensure consistency between training and evaluation.

### testing_feature_analysis.py
Extracts and scales Mel-frequency cepstral coefficient (MFCC) and Short-time Fourier transform (STFT) features from evaluation audio files, scales the MFCC features, and saves both the raw and scaled features, along with the scaler used, to ensure consistency between training and evaluation.

### train_and_test.py
Splits scaled MFCC features and corresponding MIDI parameters into training and testing datasets, scales the MIDI parameters, and saves the resulting datasets and parameter scaler for use during model training and evaluation.

### generate_bi_lstm_model.py
oads training and testing datasets to train a bi-LSTM model for synthesizer parameter prediction. Includes checkpointing and early stopping to optimize training, ultimately saving the model for use in the sound matching process​.

### soundmatch_NSGA_III.py
Uses a genetic algorithm (NSGA III) for synthesizer parameter estimation. It processes target audio files, predicts synthesizer settings, and evaluates the accuracy of the predictions against the actual settings.

### soundmatch_bi-LSTM.py
Uses a trained bi-LSTM model to predict synthesizer parameters for a set of target audio files, compares the predictions to ground truth values, and calculates the accuracy of parameter estimation for each audio file.

### audiofile_to_MIDI.py
Uses the trained bi-LSTM model to predict synthesizer parameters based on audio input from a .wav file, sending the results as MIDI Note On, pitch and CC commands to the Manatee for sound matching​.

### audiofile_to_MIDI_singlenote.py
Uses the trained bi-LSTM model to predict synthesizer parameters based on audio input from a .wav file, sending the results as MIDI CC commands to the Manatee for sound matching​ without note retriggering.

### live_audio_to_midi.py
Processes live audio input to predict synthesizer parameters using the trained bi-LSTM model and sending MIDI CC messages to the Manatee in real time, enabling live sound matching. Interactive prompts allow for selection and gain multiplication of input device, scaling of parameter sensitivity, fixed delay or  input-threshold based MIDI parameter updating, and single-line or waterall visual output.

## Acknowledgements
Thank you to Jordie Shier, George Tzanetakis, and Kirk McNally for developing the [_Spiegelib_](https://github.com/spiegelib/spiegelib) Automatic Synthesizer Programming (ASP) Library for Python, as well as the documentation and code for their [_FM Sound Match Experiment_](https://spiegelib.github.io/spiegelib/examples/fm_sound_match.html).

Many thanks to [Gabriel Vigliensoni](https://github.com/vigliensoni) for his generosity, expertise, and encouragement throughout every step of the process.
