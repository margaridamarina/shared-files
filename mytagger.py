from pathlib import Path
import json
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

class MyTagger:
    # Implement your models using the functional API
    # https://www.tensorflow.org/guide/keras/functional_api
    def build_model(self, vectorizer, word_embedding_size=64, units=64, layers=1, dropout=0.3,
                    *, max_sentence_length, seed, tags, corpus, version):
        self.config = dict(word_embedding_size=word_embedding_size, units=units, layers=layers,
                           dropout=dropout, max_sentence_length=max_sentence_length, seed=seed, tags=list(tags),
                           corpus=corpus, version=version)
        self.vectorizer = vectorizer
        keras.utils.set_random_seed(seed)
        # Each sentence is a fixed-length sequence of word IDs
        # https://www.tensorflow.org/api_docs/python/tf/keras/Input
        shape = (self.config["max_sentence_length"],)
        dtype = "int32"
        name = "word_ids"
        inputs = keras.Input(shape=shape, dtype=dtype, name=name)
        # Learn word vectors and mark zero-valued padding positions
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        input_dim = self.vectorizer.vocabulary_size()
        output_dim = self.config["word_embedding_size"]
        mask_zero = True
        # Output shape: (batch_size, max_sentence_length, word_embedding_size)
        hidden = keras.layers.Embedding(
            input_dim=input_dim, output_dim=output_dim, mask_zero=mask_zero,
        )(inputs)
        # Return every time step so each word receives a prediction
        # https://www.tensorflow.org/guide/keras/working_with_rnns
        # https://keras.io/api/layers/recurrent_layers/lstm/
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        units = self.config["units"]
        return_sequences = True
        for _ in range(self.config["layers"]):
            hidden = keras.layers.LSTM(units=units, return_sequences=return_sequences)(hidden)
        # Drop activations during training to reduce overfitting
        rate = self.config["dropout"]
        hidden = keras.layers.Dropout(rate=rate)(hidden)
        # Apply the same 17-class Softmax classifier to every word
        # https://keras.io/api/layers/recurrent_layers/time_distributed/
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        units = len(self.config["tags"])
        activation = "softmax"
        layer = keras.layers.Dense(units=units, activation=activation)
        name = "upos"
        outputs = keras.layers.TimeDistributed(layer=layer, name=name)(hidden)
        name = "lstm_pos_tagger"
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        # Train with tag IDs and compute accuracy using token weights
        # Sparse crossentropy accepts integer tag IDs
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
        from_logits = False  # The output already uses Softmax
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=loss,
            weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        return self.model

    # Train and save the model and its training history
    def train(self, train_batches, val_batches, output_dir, epochs=25, batch_size=64):
        output_dir = Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError("Use a new output directory for a new experiment")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.update(epochs=epochs, batch_size=batch_size, patience=4,
                           tensorflow_version=tf.__version__, keras_version=keras.__version__)
        # Save everything needed to reconstruct preprocessing later
        for filename, data in {
            "config.json": self.config,
            "vocabulary.json": self.vectorizer.get_vocabulary()[2:],
        }.items():
            (output_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        # Keep the lowest val loss, stop early and log each epoch
        callbacks = [
            keras.callbacks.ModelCheckpoint(output_dir / "best.keras", monitor="val_loss", save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            keras.callbacks.CSVLogger(output_dir / "history.csv"),
            keras.callbacks.TerminateOnNaN(),
        ]
        # Measure training time; validation data never update weights
        started = time.perf_counter()
        history = self.model.fit(
            train_batches,
            validation_data=val_batches,
            epochs=epochs, callbacks=callbacks, verbose=2,
        ).history
        # Restore the checkpoint selected by validation loss
        self.model = keras.models.load_model(output_dir / "best.keras")
        # Save learning curves, stopping status and the selected epoch
        training = {
            "seconds": time.perf_counter() - started,
            "epochs": len(history["loss"]),
            "best_epoch": int(np.argmin(history["val_loss"])) + 1,
            "early_stopped": len(history["loss"]) < epochs,
            "initial_train_loss": float(history["loss"][0]),
            "final_train_loss": float(history["loss"][-1]),
        }
        for filename, data in {"history.json": history, "training.json": training}.items():
            (output_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return history

    # Evaluate real tokens while ignoring padding
    def evaluate(self, encoded, batch_size=64):
        x, y, mask = encoded
        # Predict in batches with dropout disabled
        predictions = []
        for start in range(0, len(x), batch_size):
            probabilities = self.model(x[start:start + batch_size], training=False).numpy()
            predictions.extend(probabilities.argmax(axis=-1))
        predictions = np.asarray(predictions)
        real = mask.astype(bool)
        # Unknown-word metrics include only real tokens with the UNK ID
        # ID 1 is [UNK]; exclude padding from OOV counts
        oov = (x == 1) & real
        # Rows are gold tags; columns are predicted tags
        confusion = np.zeros((len(self.config["tags"]), len(self.config["tags"])), dtype="int64")
        np.add.at(confusion, (y[real], predictions[real]), 1)
        # Compute precision, recall and F1 separately for each tag
        per_tag = {}
        for i, tag in enumerate(self.config["tags"]):
            true_positive = int(confusion[i, i])
            support = int(confusion[i].sum())
            proposed = int(confusion[:, i].sum())
            precision = true_positive / proposed if proposed else 0.0
            recall = true_positive / support if support else 0.0
            per_tag[tag] = dict(precision=precision, recall=recall, support=support,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        # Summarize token, OOV and complete-sentence performance
        return dict(
            accuracy=float((predictions[real] == y[real]).mean()),
            tokens=int(real.sum()), sentences=len(x),
            oov_accuracy=float((predictions[oov] == y[oov]).mean()) if oov.any() else None,
            oov_tokens=int(oov.sum()), oov_rate=float(oov.sum() / real.sum()),
            sentence_accuracy=float(np.all((predictions == y) | ~real, axis=1).mean()),
            per_tag=per_tag, confusion_labels=self.config["tags"], confusion_matrix=confusion.tolist(),
        )

    # Generate labels for an unseen sentence
    def predict(self, sentence):
        words = re.findall(r"\w+(?:['’\-]\w+)*|[^\w\s]", sentence) if isinstance(sentence, str) else list(sentence)
        if not words:
            return []
        max_sentence_length = self.config["max_sentence_length"]
        if len(words) > max_sentence_length or any(not isinstance(word, str) or not word for word in words):
            raise ValueError(f"Provide 1–{max_sentence_length} nonempty tokens")
        # Reuse the training vocabulary and remove padding from the output
        tokens = tf.constant([words + [""] * (max_sentence_length - len(words))])
        x = self.vectorizer(tokens)
        probabilities = self.model(x, training=False).numpy()[0, :len(words)]
        # Return the most likely tag and its Softmax confidence per word
        return [dict(token=word, upos=self.config["tags"][int(p.argmax())], confidence=float(p.max()))
                for word, p in zip(words, probabilities)]
