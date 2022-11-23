"""
Module used to train or load existing recurrent neural network

The Asian American poems are fed into the recurrent neural network and trained
on 120 EPOCHS with .2 temperature. 

Credits goes to https://www.tensorflow.org/text/tutorials/text_generation. 

Author: Dejie Zhen
CSCI 3725
Date: November 22, 2022
"""

import tensorflow as tf
import numpy as np
import os
import time
from django.conf import settings
import random


class Neural_Network:
    def __init__(self) -> None:
        """
        A class representing the Recurrent Neural Network that will be used to
        generate our model

        Args:
            none
        """
        self.vocab = ""
        self.chars = ""
        self.ids_from_chars = None
        self.ids = None
        self.chars_from_ids = ""
        self.chars = ""
        self.all_ids = ""
        self.ids_dataset = ""
        self.seq_length = 100
        self.sequences = ""
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 10000
        self.dataset = ""
        self.vocab_size = 0
        self.embedding_dim = 256
        self.rnn_units = 1024
        self.EPOCHS = 120


    def read_text(self):
        """
        Reads and returns the Asian American text
        
        Args:
            none
        """
        static_folder = settings.STATICFILES_DIRS[0]
        path_to_file = static_folder + "/merged/asian_poems.txt"
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        return text

    def get_vocab(self, text):
        """
        Get all the vocab in the text and put it into a set

        Args:
            text(str): text from input file
        """
        vocab = sorted(set(text))
        return vocab

    def set_initial_variables(self):
        """
        Set all the initial variables to prepare for training model

        Args:
            none
        """
        text = self.read_text()
        self.vocab = self.get_vocab(text)

        example_texts = ['abcdefg', 'xyz']
        chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
        self.chars = chars

        ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(self.vocab), mask_token=None)
        self.ids_from_chars = ids_from_chars

        ids = ids_from_chars(chars)
        self.ids = ids

        chars_from_ids = tf.keras.layers.StringLookup(
                            vocabulary=ids_from_chars.get_vocabulary(), 
                            invert=True, 
                            mask_token=None)
        self.chars_from_ids = chars_from_ids

        chars = chars_from_ids(ids)
        self.chars = chars
        tf.strings.reduce_join(chars, axis=-1).numpy()

        all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        self.all_ids = all_ids
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        self.ids_dataset = ids_dataset
        self.sequences = \
            self.ids_dataset.batch(self.seq_length+1, drop_remainder=True)
        dataset = self.sequences.map(self.split_input_target)
        self.dataset = (
            dataset
            .shuffle(self.BUFFER_SIZE)
            .batch(self.BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))
        self.vocab_size = len(self.ids_from_chars.get_vocabulary())

    def text_from_ids(self,ids):
        """
        Get the corresponding text by their id's

        Args:
            ids(int): id of text
        """
        return tf.strings.reduce_join(self.chars_from_ids(self.ids), axis=-1)
         
    def split_input_target(self, sequence):
        """
        Get the input sequence text and target sequence text

        Args:
            sequence(list): current sequence in model
        """
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def train(self, model):
        """
        Train new model in the training checkpoints directory

        Args:
            model(Model): model of the neural network
        """
        # Directory where the checkpoints will be saved
        static_folder = settings.STATICFILES_DIRS[0]
        checkpoint_dir = static_folder + '/training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = \
            model.fit(self.dataset, 
                    epochs=self.EPOCHS, 
                    callbacks=[checkpoint_callback])

    def load_model(self):
        """
        Load the trained model from the latest training checkpoint

        Args:
            none
        """
        static_folder = settings.STATICFILES_DIRS[0]
        checkpoint_dir = static_folder + '/training_checkpoints'

        model = MyModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            rnn_units=self.rnn_units)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest).expect_partial()
        return model

    def generate_poem(self, model, speech_data):
        """
        Generates a poem based on character prediction. Uses speech recognition
        data as a prompt to begin generating the poem. If no speech recognition
        is avaiable, the prompt will be randomly defaulted to a quote.

        Credits to default quotes from:
        https://teambuilding.com/blog/asian-heritage-month-quotes
        https://tinyurl.com/58rk2vs6

        Args:
            model(Model): model of the neural network
            speech_data(str): data from speech recognition
        """
        one_step_model = \
            OneStep(model, self.chars_from_ids, self.ids_from_chars)
        start = time.time()
        states = None

        default_prompts = \
            ["Asian Empowerment ", 
            "The American Dream belongs to all of us ", 
            "The power of visibility can never be underestimated ",
            "In a gentle way, you can shake the world ",
            "Don't deny the past. Remember everything. \nIf you're bitter, "
            + "be bitter.\n Cry it out! Scream! Denial is gangrene ",
            "Don't ever think that just because you do things differently, "
            + "you're wrong ",
            "I intend to live life, not just exist ",
            "The Asian culture has to be a part of what we see on TV and "
            + "in movies ",
            "You change the world by being yourself ",
            "I dream. Sometimes I think that's the only right thing to do "
            ]
        next_char = ""
        if speech_data == "":
            default = random.choice(default_prompts)
            print('Defaulted Prompt: ' + default)
            next_char = tf.constant([default])
        else:
            print('Speech Data: ' + speech_data)
            next_char = tf.constant([speech_data])
        result = [next_char]

        for n in range(800):
            next_char, states = one_step_model.generate_one_step(next_char, 
                                                                states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        end = time.time()
        print('\nRun time:', end - start)
        return result[0].numpy().decode('utf-8')

    def execute_model(self, speech_data):
        """
        Execute model by setting all initial variables and defining model. 
        Can either train a new model or load existing trained model

        Args:
            speech_data(str): data from speech recognition
        """
        self.set_initial_variables()
        model = MyModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            rnn_units=self.rnn_units)

        for input_example_batch, target_example_batch in self.dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape, 
                    "# (batch_size, sequence_length, vocab_size)")

        model.summary()
        # sampled_indices
        sampled_indices = \
            tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        example_batch_mean_loss = \
            loss(target_example_batch, example_batch_predictions)

        print("Prediction shape: ", 
            example_batch_predictions.shape, 
            " # (batch_size, sequence_length, vocab_size)")
        print("Mean loss:        ", example_batch_mean_loss)

        tf.exp(example_batch_mean_loss).numpy()
        model.compile(optimizer='adam', loss=loss)

        # Training Model
        # self.train(model)
        # curr_model = model

        # Loading Model
        loaded_model = self.load_model()
        curr_model = loaded_model
        poem = self.generate_poem(curr_model, speech_data)
        return poem

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        """
        Initialization of model that we will use to train on our data

        Args:
            vocab_size(int): size of vocab
            embedding_dim(int): embedding dimension
            rnn_units(int): number of rnn units
        """
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=.2):
        """
        Generating text with this model by running it in a loop and keeping
        track of the model's internal state

        Args:
            model(Model): model of our trained data
            chars_from_ids(char): chars that correspond to ids
            ids_from_chars(int): ints that correspond to chars
            temperature(float): how greedy our model is
        """
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        """
        Get input chars and predict the next character

        Args:
            inputs(string): current string input
            states(states): data saved between different calls
        """
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states
    