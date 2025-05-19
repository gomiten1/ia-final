import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

class GliomaClassifierAgent:
    """
    Agente clasificador de gliomas basado en árbol de decisión.
    Permite obtener certeza de cada predicción.
    """
    def __init__(self, max_depth=None, random_state=42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, class_weight='balanced')
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
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=['LGG', 'GBM']
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'roc_curve': (fpr, tpr)
        }

    def export_tree(self, feature_names):
        """Exporta la representación textual del árbol."""
        return export_text(self.model, feature_names=feature_names)


# Cargar dataset
df = pd.read_csv('data/TCGA_InfoWithGrade.csv')


def predecir(user_df):
    # Usuario
    labels, certainties = agent.predict(user_df)
    label = 'GBM' if labels[0] == 1 else 'LGG'
    cf = certainties[0]
    return f"## Predicción para el paciente ingresado: *{label}* (certeza=*{cf:.2f}*)"


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

feature_names = list(X.columns)
