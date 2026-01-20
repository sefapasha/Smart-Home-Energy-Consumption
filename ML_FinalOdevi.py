import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time



df = pd.read_csv('Smart Home Dataset.csv', low_memory=False)
df = df.dropna()

df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')

df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear


# eksik hava durumu verileri
weather_columns = ['temperature', 'humidity', 'apparentTemperature', 'pressure', 'windSpeed', 'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']

for col in weather_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())


le = LabelEncoder()
df['summary_encoded'] = le.fit_transform(df['summary'].astype(str))

# pivot 1 saat bazlı normal tüketim
pivot_hour = df.pivot_table(
    index='hour',
    values='use [kW]',
    aggfunc='mean'
)
pivot_hour.columns = ['hourAvg']


# pivot 2 sicaklik
pivot_temp = df.pivot_table(
    index='temperature',
    values='use [kW]',
    aggfunc='mean'
)
pivot_temp.columns = ['temperatureAvg']


# pivot 3 nem 
pivot_humidity = df.pivot_table(
    index='humidity',
    values='use [kW]',
    aggfunc='mean'
)
pivot_humidity.columns = ['humidityAvg']

df = df.merge(pivot_hour, on='hour', how='left')
df = df.merge(pivot_temp, on='temperature', how='left')
df = df.merge(pivot_humidity, on='humidity', how='left')

df['baseline_mean'] = (
    df['hourAvg'] +
    df['temperatureAvg'] +
    df['humidityAvg']
) / 3


df['is_anomaly'] = (df['use [kW]'] > df['baseline_mean']).astype(int) # kw degerimiz ortlama kw degerimizden fazla ise 1 degilse 0(normal) olarak işaretledimm


features = ['hour', 'day_of_week', 'month','temperature','temperatureAvg', 
            'humidity', 'apparentTemperature', 'pressure', 
            'windSpeed', 'windBearing', 'precipIntensity', 
            'dewPoint', 'precipProbability'
]
X = df[features]
X = X.fillna(0)
###
y = df['is_anomaly']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# logistic regresyon ve KNN için özellikleri olceklendirme 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("Modellerin eğitimleri Logistic,KNN,RF:")

models = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
    "K-En Yakın Komşu (KNN)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
model_predictions = {}
model_objects = {}

for name, model in models.items():
    start = time.time()
    print(f"\n-> {name} eğitiliyor...")
    
    # model eğitimi
    model.fit(X_train_scaled, y_train)
    model_objects[name] = model
    
    # tahminler 
    preds = model.predict(X_test_scaled)
    model_predictions[name] = preds
    
    # accurasy, recall, precision f1 gibi metrikleri hesaplama
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    elapsed = time.time() - start
    
    print(f"   Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"   Hassasiyet (Precision): %{precision*100:.2f}")
    print(f"   Duyarlılık (Recall): %{recall*100:.2f}")
    print(f"   F1-Skoru: %{f1*100:.2f}")
    print(f"   Süre: {elapsed:.2f} saniye")
    print("   Detaylı Rapor:")
    print(classification_report(y_test, preds))
    
    results.append({
        "Model": name, 
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Time": elapsed
    })

# SONUÇ
print("\n" + "="*50)
best_model = max(results, key=lambda x: x['Accuracy'])
print(f"En iyi model: {best_model['Model']} (Dogruluk: {best_model['Accuracy']:.4f})")# en iyi model RF olarak çıktı veriyor
print("="*50)
