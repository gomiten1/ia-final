class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index  # índice característica para dividir
        self.threshold = threshold          # valor de corte
        self.left = left                    # nodo izquierdo
        self.right = right                  # nodo derecho
        self.value = value                  # valor de clase si es hoja

# Calcular índice Gini para un split
def gini(y):
    m = len(y)
    if m == 0:
        return 0
    counts = np.bincount(y)
    probs = counts / m
    return 1 - np.sum(probs**2)

# Dividir dataset dado índice y valor de corte
def split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] < threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Encontrar el mejor split
def best_split(X, y):
    m, n_features = X.shape
    if m <= 1:
        return None, None
    best_gini = 1
    best_idx, best_thr = None, None
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = split(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / m
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_idx = feature_index
                best_thr = threshold
    return best_idx, best_thr

# Construir árbol recursivamente
def build_tree(X, y, depth=0, max_depth=5):
    num_samples_per_class = [np.sum(y == i) for i in range(2)]
    predicted_class = np.argmax(num_samples_per_class)
    node = Node(value=predicted_class)

    if depth < max_depth:
        feature_index, threshold = best_split(X, y)
        if feature_index is not None:
            X_left, y_left, X_right, y_right = split(X, y, feature_index, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                node.feature_index = feature_index
                node.threshold = threshold
                node.left = build_tree(X_left, y_left, depth + 1, max_depth)
                node.right = build_tree(X_right, y_right, depth + 1, max_depth)
                node.value = None  # no es hoja
    return node

# Predecir un solo ejemplo
def predict_sample(node, x):
    while node.value is None:
        if x[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

# Predecir conjunto
def predict(node, X):
    return np.array([predict_sample(node, x) for x in X])