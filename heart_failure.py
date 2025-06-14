import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix

from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Input

# Load dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(current_dir, 'input', 'heart.csv'))

# Split features and target
x_sample = dataset.drop("HeartDisease", axis=1)
y_sample = dataset["HeartDisease"]

# Label Encoding
label_encoders = {}
for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
    le = LabelEncoder()
    x_sample[col] = le.fit_transform(x_sample[col])
    label_encoders[col] = le

# Scaling
scaler = MinMaxScaler()
x_sample[["Age", "RestingBP", "Cholesterol", "MaxHR"]] = scaler.fit_transform(
    x_sample[["Age", "RestingBP", "Cholesterol", "MaxHR"]]
)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2, random_state=32)

# Logistic Regression (optional)
logreg = linear_model.LogisticRegression(random_state=32, max_iter=200)
logreg.fit(x_train, y_train)
y_test_pred = logreg.predict(x_test)
#print(classification_report(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))

# Neural Network Model
def build_classifier():
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(units=8, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=4, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# KerasClassifier with cross-validation
classifier = KerasClassifier(model=build_classifier, epochs=100, batch_size=32, verbose=1)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=3)
print("Accuracy mean:", accuracies.mean())
print("Accuracy std:", accuracies.std())

# Final model training
final_model = build_classifier()
final_model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

# Interactive prediction from user
def predict_user_input(model, scaler, label_encoders):
    print("\nPlease answer the following questions:")

    age = float(input("Age: "))
    sex = input("Sex (Male/Female): ")
    chest_pain = input("Chest Pain Type (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic): ")
    resting_bp = float(input("Resting Blood Pressure: "))
    cholesterol = float(input("Cholesterol: "))
    fasting_bs = int(input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No): "))
    rest_ecg = input("Resting ECG (Normal, ST, LVH): ")
    max_hr = float(input("Maximum Heart Rate Achieved: "))
    exercise_angina = input("Exercise-induced angina (Yes/No): ")
    oldpeak = float(input("Oldpeak: "))
    st_slope = input("ST Slope (Up, Flat, Down): ")

    # Build input DataFrame
    input_df = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": rest_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    # Encode categorical features
    for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale numeric features
    input_df[["Age", "RestingBP", "Cholesterol", "MaxHR"]] = scaler.transform(input_df[["Age", "RestingBP", "Cholesterol", "MaxHR"]])

    # Predict
    prediction = model.predict(input_df)
    if isinstance(prediction, (np.ndarray, list)):
        prediction = prediction[0]

    print("\nPrediction result:")
    if prediction >= 0.5:
        print("⚠️ The model predicts that you **may have heart disease**.")
    else:
        print("✅ The model predicts that you **likely do not have heart disease**.")

# Call the prediction function
predict_user_input(final_model, scaler, label_encoders)
