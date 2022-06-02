from skin_cancer_detection.data import get_data_from_gcp
from skin_cancer_detection.model import initialize_model, compile_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":
    # Get and clean data
    N = 100
    skin_df = get_data_from_gcp(nrows=N)

    # MISSING: resize!!
    skin_df['image_resized']

    y = skin_df['dx']
    dict_target = {'bkl':0, 'nv':1, 'df':2, 'mel':3, 'vasc':4, 'bcc':5, 'akiec':6}
    y_num = y.map(dict_target.get)
    skin_df['target'] = y_num
    y_cat = to_categorical(y_num, num_classes = 7)
    X = skin_df['image_resized']
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3, random_state=42)

    # stack
    X_train_stack = np.stack(X_train)
    X_test_stack = np.stack(X_test)
    model = initialize_model()
    model = compile_model(model)

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
