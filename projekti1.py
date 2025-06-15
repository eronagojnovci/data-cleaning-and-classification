import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ================== LEXIMI DHE PËRPUNIMI I TË DHËNAVE ===================

file_path = 'vgsales.csv'
df = pd.read_csv(file_path)

df.fillna(df.median(numeric_only=True), inplace=True)

numerical_columns = df.select_dtypes(include=['number']).columns
df[numerical_columns] = df[numerical_columns].abs()

df[numerical_columns] = df[numerical_columns].round(1)

scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
price_column = 'Global_Sales'
if price_column in df.columns:
    scaled_price = df[price_column]
    intervals = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [1, 2, 3, 4]
    df[price_column] = pd.cut(scaled_price, bins=intervals, labels=labels, include_lowest=True).astype(int)

categorical_columns = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

df.to_csv('vgsales_processed.csv', index=False)

print(df.head())

# ================== ALGORITMAT KLASIFIKUES ===================

X = df.drop('Global_Sales', axis=1)
y = df['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================== 1. Decision Tree ===================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("\n--- Rezultatet për Decision Tree ---")
print(classification_report(y_test, y_pred_dt, zero_division=0))
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_report = classification_report(y_test, y_pred_dt, output_dict=True, zero_division=0)

# ================== 2. K-Nearest Neighbors ===================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("\n--- Rezultatet për KNN ---")
print(classification_report(y_test, y_pred_knn, zero_division=0))
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn, output_dict=True, zero_division=0)

# ================== 3. Random Forest ===================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("\n--- Rezultatet për Random Forest ---")
print(classification_report(y_test, y_pred_rf, zero_division=0))
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)

# ================== TABELA E KRAHASIMIT ===================
comparison_table = pd.DataFrame({
    'Precision': [
        dt_report['weighted avg']['precision'],
        knn_report['weighted avg']['precision'],
        rf_report['weighted avg']['precision']
    ],
    'Recall': [
        dt_report['weighted avg']['recall'],
        knn_report['weighted avg']['recall'],
        rf_report['weighted avg']['recall']
    ],
    'F1-Score': [
        dt_report['weighted avg']['f1-score'],
        knn_report['weighted avg']['f1-score'],
        rf_report['weighted avg']['f1-score']
    ],
    'Accuracy': [
        dt_accuracy,
        knn_accuracy,
        rf_accuracy
    ]
}, index=['Decision Tree', 'KNN', 'Random Forest'])

print("\n========== Tabela e Krahasimit të Algoritmeve ==========")
print(comparison_table.round(3))
