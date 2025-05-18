'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Cargar el dataset
# Suponemos que tienes un archivo "tiroides.csv" con las columnas que listaste
df = pd.read_csv('data/tiroides.csv')

# 2. Definir variables predictoras y objetivo
y = df['Recurred']  # objetivo sí/no
X = df.drop(columns=['Recurred'])

# 3. Preprocesamiento de variables categóricas
# Para simplicidad, usaremos OneHotEncoder para todas las categóricas excepto Age
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Combinar con la columna Age
df_num = X[['Age']].to_numpy()
X_preprocessed = np.hstack([df_num, X_encoded])
feature_names = ['Age'] + encoded_cols.tolist()

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluar desempeño
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)  # probabilidades de cada clase

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Razonamiento con incertidumbre
# Vamos a adjuntar un factor de certeza (CF) basado en la probabilidad de la clase predicha

def attach_certainty(proba_array, classes):
    # proba_array: array de probabilidades para un único ejemplo
    idx = np.argmax(proba_array)
    predicted_class = classes[idx]
    certainty = proba_array[idx]
    return predicted_class, certainty

results = []
for probs in y_prob:
    cls, cf = attach_certainty(probs, clf.classes_)
    results.append({'Predicted': cls, 'Certainty': cf})

results_df = pd.DataFrame(results)
print("\nEjemplos con factor de certeza:\n", results_df.head())

# 8. Visualizar la estructura del árbol (opcional)
text_tree = export_text(clf, feature_names=feature_names)
print("\nÁrbol de decisión:\n", text_tree)'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Cargar el dataset
df = pd.read_csv('data/tiroides.csv')

# 2. Definir variables predictoras y objetivo
y = df['Recurred']  # objetivo sí/no
X = df.drop(columns=['Recurred'])

# 3. Preprocesamiento de variables categóricas
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)

# Combinar con la columna Age
df_num = X[['Age']].to_numpy()
X_preprocessed = np.hstack([df_num, X_encoded])
feature_names = ['Age'] + encoded_cols.tolist()

# 4. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluar desempeño
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Razonamiento con incertidumbre
def attach_certainty(proba_array, classes):
    idx = np.argmax(proba_array)
    return classes[idx], proba_array[idx]

results = []
for probs in y_prob:
    cls, cf = attach_certainty(probs, clf.classes_)
    results.append({'Predicted': cls, 'Certainty': cf})

results_df = pd.DataFrame(results)
print("\nEjemplos con factor de certeza:\n", results_df.head())

# 8. Visualizar la estructura del árbol (opcional)
text_tree = export_text(clf, feature_names=feature_names)
print("\nÁrbol de decisión:\n", text_tree)

# 9. Sección interactiva para nuevas mediciones
print("\nIngresa nuevos valores para predecir recurrencia:")
# Recopilar inputs del usuario
input_age = int(input("Age: "))
input_values = [input_age]
for col in categorical_cols:
    val = input(f"{col} ({encoder.categories_[categorical_cols.index(col)]}): ")
    input_values.append(val)

# Convertir entrada a DataFrame
df_new = pd.DataFrame([input_values], columns=['Age'] + categorical_cols)
# Codificar categóricas
X_new_encoded = encoder.transform(df_new[categorical_cols])
X_new = np.hstack([np.array(df_new[['Age']]), X_new_encoded])

# Predecir
pred = clf.predict(X_new)[0]
prob = clf.predict_proba(X_new)[0]
# Obtener certeza
pred_class, certainty = attach_certainty(prob, clf.classes_)
print(f"\nPredicción: {pred_class} con certeza de {certainty:.2f}")




