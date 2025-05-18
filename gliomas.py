import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

class GliomaClassifierAgent:
    """
    Agente clasificador de gliomas basado en árbol de decisión.
    Permite obtener certeza de cada predicción.
    """
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.trained = False

    def fit(self, X, y):
        """Entrena el modelo con datos de entrenamiento."""
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X):
        """
        Realiza predicciones y devuelve etiquetas junto con factor de certeza.
        CF = 2 * P(predicho) - 1  # Rango [-1,1]
        """
        if not self.trained:
            raise RuntimeError("El modelo no ha sido entrenado.")
        probs = self.model.predict_proba(X)
        labels = self.model.predict(X)
        # Obtener probabilidad de la clase predicha
        cf = [2 * probs[i, lab] - 1 for i, lab in enumerate(labels)]
        return labels, cf

    def evaluate(self, X_test, y_test):
        """Mide desempeño usando métricas estándar."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['LGG', 'GBM'])
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        return {
            'accuracy': acc,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': auc
        }

    def export_tree(self, feature_names):
        """Exporta la representación textual del árbol."""
        return export_text(self.model, feature_names=feature_names)

if __name__ == '__main__':
    # Cargar dataset
    df = pd.read_csv('data/TCGA_InfoWithGrade.csv')

    # Mapeo de columnas categóricas si es necesario
    # Edad y mutaciones son numéricas; Race y Gender ya están codificados.

    # Separar características y etiqueta
    X = df.drop(columns=['Grade', 'ID']) if 'ID' in df.columns else df.drop(columns=['Grade'])
    y = df['Grade']

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Entrenar agente
    agent = GliomaClassifierAgent(max_depth=5)
    agent.fit(X_train, y_train)

    # Evaluar desempeño
    results = agent.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.2f}")
    print("Classification Report:\n", results['classification_report'])
    print("Confusion Matrix:\n", results['confusion_matrix'])
    print(f"ROC AUC: {results['roc_auc']:.2f}")

    # Ejemplo de predicción con certeza
    sample = X_test.iloc[:5]
    labels, certainties = agent.predict(sample)
    for i, (lab, cf) in enumerate(zip(labels, certainties)):
        print(f"Muestra {i}: Predicción = {'GBM' if lab==1 else 'LGG'}, Certeza = {cf:.2f}")

    # Mostrar estructura del árbol
    print("\nÁrbol de decisión:\n")
    print(agent.export_tree(feature_names=list(X.columns)))
