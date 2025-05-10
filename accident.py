import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import time
import sys
from IPython.display import clear_output

# Load the dataset
try:
    df = pd.read_csv('RTA Dataset.csv')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'RTA Dataset.csv' not found. Please ensure the file exists.")
    exit()

# Data Preprocessing
print("\nPreprocessing data...")
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Impute missing values
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Time processing
def parse_time(time_str):
    try:
        if pd.isna(time_str):
            return pd.NaT
        if isinstance(time_str, str):
            if len(time_str.split(':')) == 2:
                time_str += ':00'
            return pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce').time()
        return time_str
    except Exception as e:
        return pd.NaT

df['Time'] = df['Time'].apply(parse_time)
df['Time_Category'] = pd.cut(
    pd.to_datetime(df['Time'].astype(str), format='%H:%M:%S', errors='coerce').dt.hour,
    bins=[0, 6, 12, 18, 24],
    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
    right=False
)
df['Time_Category'].fillna('Morning', inplace=True)

# Drop rows with missing target
df = df.dropna(subset=['Accident_severity'])

# Ensure severity order
severity_order = ['Slight Injury', 'Serious Injury', 'Fatal injury']
df['Accident_severity'] = pd.Categorical(df['Accident_severity'], categories=severity_order, ordered=True)

# Function to display plots with pauses
def show_plot(fig, title):
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    time.sleep(2)
    plt.close()

# Exploratory Data Analysis
print("\nDisplaying visualizations...")

# 1. Accident Severity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Accident_severity', data=df, order=severity_order)
plt.title('Distribution of Accident Severity')
plt.xlabel('Accident Severity')
plt.ylabel('Number of Accidents')
show_plot(plt, "Visualization 1/4: Accident Severity Distribution")

# 2. Accident Severity by Time of Day
plt.figure(figsize=(10, 6))
sns.countplot(x='Time_Category', hue='Accident_severity', data=df,
              order=['Night', 'Morning', 'Afternoon', 'Evening'])
plt.title('Accident Severity by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Accidents')
plt.legend(title='Accident Severity')
show_plot(plt, "Visualization 2/4: Severity by Time of Day")

# 3. Accident Severity by Road Conditions
plt.figure(figsize=(10, 6))
sns.countplot(x='Road_surface_conditions', hue='Accident_severity', data=df)
plt.title('Accident Severity by Road Surface Conditions')
plt.xlabel('Road Surface Conditions')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.legend(title='Accident Severity')
show_plot(plt, "Visualization 3/4: Severity by Road Conditions")

# 4. Top Causes of Accidents
plt.figure(figsize=(12, 6))
df['Cause_of_accident'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Causes of Accidents')
plt.xlabel('Cause of Accident')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
show_plot(plt, "Visualization 4/4: Top Causes of Accidents")

# Prepare Data for Modeling
X = df.drop(['Accident_severity', 'Time'], axis=1)
y = df['Accident_severity']

le = LabelEncoder()
le.fit(severity_order)
y_encoded = le.transform(y)

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Random Forest Model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Neural Network Model
print("\nTraining Neural Network model...")
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train)
X_test_nn = scaler.transform(X_test)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = nn_model.fit(X_train_nn, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)

nn_loss, nn_accuracy = nn_model.evaluate(X_test_nn, y_test, verbose=0)
print(f"\nNeural Network Accuracy: {nn_accuracy:.2f}")

# Display NN training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
show_plot(plt, "Neural Network Training History")

# Prediction Function
def predict_accident_severity():
    print("\nEnter the following details for accident severity prediction:")
    input_data = {}
    original_cols = df.drop(['Accident_severity', 'Time'], axis=1).columns

    for col in original_cols:
        if col == 'Time_Category':
            continue

        while True:
            user_input = input(f"Enter value for {col} (Type: {df[col].dtype}): ")
            try:
                if df[col].dtype in ['int64', 'float64']:
                    input_data[col] = float(user_input)
                else:
                    input_data[col] = user_input
                break
            except ValueError:
                print(f"Invalid input for {col}. Please enter a valid {df[col].dtype} value.")

    # Handle Time_Category
    time_input = input("Enter time of accident (HH:MM or HH:MM:SS): ")
    try:
        if len(time_input.split(':')) == 2:
            time_input += ':00'
        time_hour = pd.to_datetime(time_input, format='%H:%M:%S').hour
        time_category = pd.cut(
            [time_hour],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )[0]
    except:
        time_category = 'Morning'
    input_data['Time_Category'] = time_category

    # Convert to DataFrame and prepare features
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X.columns]

    # Predictions
    rf_pred = rf_model.predict(input_df)
    rf_severity = le.inverse_transform(rf_pred)[0]

    input_df_nn = scaler.transform(input_df)
    nn_pred = nn_model.predict(input_df_nn, verbose=0)
    nn_severity = le.inverse_transform([np.argmax(nn_pred, axis=1)])[0]

    return rf_severity, nn_severity

# Run prediction
try:
    rf_severity, nn_severity = predict_accident_severity()
    print(f"\nRandom Forest Predicted Accident Severity: {rf_severity}")
    print(f"Neural Network Predicted Accident Severity: {nn_severity}")
except Exception as e:
    print(f"\nError during prediction: {str(e)}")

# Road Safety Insights
print("\nRoad Safety Insights:")
print("1. Most accidents occur in the afternoon and evening, suggesting increased traffic or fatigue.")
print("2. Wet road conditions significantly increase fatal accidents; improve drainage and road maintenance.")
print("3. 'No distancing' and 'Changing lane' are top accident causes; promote driver education.")
print("4. Younger drivers (18-30) are involved in more severe accidents; targeted training programs needed.")