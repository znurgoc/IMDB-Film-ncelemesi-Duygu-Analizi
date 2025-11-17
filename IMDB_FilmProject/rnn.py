import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# AUC fonksiyonunu ve roc_curve'ü ayrı ayrı içe aktarıyoruz
from sklearn.metrics import classification_report, roc_curve, auc

from kerastuner.tuners import RandomSearch

import warnings
warnings.filterwarnings("ignore")

# Veri yükleme
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

def build_model(hp): # hp: hyperparametre
    model = Sequential()
    model.add(Embedding(input_dim=10000,
              output_dim=hp.Int("Embedding_output", min_value=32, max_value=128, step=32),
              input_length=maxlen))
    model.add(SimpleRNN(units=hp.Int("rnn_units", min_value=32, max_value=128, step=32)))
    model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy", "AUC"])
    return model

# KerasTuner kurulumu
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=2,
    executions_per_trial=1,
    directory="rnn_tunner_directory"
)

# Geri çağrı (Callback)
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Arama (Search)
tuner.search(x_train, y_train,
             epochs=2,
             validation_split=0.2,
             callbacks=[early_stopping])


best_model = tuner.get_best_models(num_models=1)[0]
# AUC isminin çakışmasını önlemek için 'auc' yerine 'auc_model' kullanıldı
loss, accuracy, auc_model = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss:{loss:.4f}, test accuracy: {accuracy:.3f}, test auc: {auc_model:.3f}")

y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print(classification_report(y_test, y_pred))

# roc_curve'de olasılıkları kullanmak daha iyidir
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

# Burada, sklearn.metrics'ten gelen auc fonksiyonu çağrılır
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area=%0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()