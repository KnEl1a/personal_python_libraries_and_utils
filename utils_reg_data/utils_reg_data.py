from sklearn.model_selection import train_test_split

def obtener_datos_reg(X, y, train_size=0.8, random_state=1):
    """
    Divide el conjunto de datos en conjuntos de entrenamiento, validación y prueba para tareas de regresión.

    La función realiza dos divisiones sucesivas del conjunto de datos:
    1. Divide los datos en conjuntos de entrenamiento y prueba.
    2. Luego, divide el conjunto de entrenamiento en subconjuntos de entrenamiento y validación.

    La proporción del conjunto de datos utilizado para entrenamiento y validación está determinada por el parámetro `train_size`.
    El porcentaje de cada conjunto se calcula y se imprime para referencia.

    Args:
        X (array-like o DataFrame): Características (features) del conjunto de datos.
        y (array-like o Series): Etiquetas (targets) del conjunto de datos.
        train_size (float, opcional, por defecto=0.8): Fracción del conjunto de datos que se utiliza para entrenamiento y validación.
        random_state (int, opcional, por defecto=1): Semilla para la generación de números aleatorios, asegurando reproducibilidad.

    Returns:
        tuple: Un tuple con seis elementos:
            - X_train (array-like o DataFrame): Conjunto de características de entrenamiento.
            - X_valid (array-like o DataFrame): Conjunto de características de validación.
            - y_train (array-like o Series): Conjunto de etiquetas de entrenamiento.
            - y_valid (array-like o Series): Conjunto de etiquetas de validación.
            - X_test (array-like o DataFrame): Conjunto de características de prueba.
            - y_test (array-like o Series): Conjunto de etiquetas de prueba.

    Example:
        >>> X_train, X_valid, y_train, y_valid, X_test, y_test = obtener_datos_reg(X, y)
        >>> y_train.head()
        0    5.0
        1    7.2
        2    6.8
        3    9.1
        4    4.5
        Name: target, dtype: float64

    Notes:
        - La división de datos se realiza en dos etapas: primero se separa un conjunto de prueba, y luego el conjunto de entrenamiento se divide en entrenamiento y validación.
        - La proporción de datos para el conjunto de validación es la misma que la del conjunto de entrenamiento.
        - Asegúrate de que `X` e `y` tengan el mismo número de filas.
    """
    # Primera división: conjunto de prueba
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    # Segunda división: conjunto de validación
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, train_size=train_size, random_state=random_state)

    total = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]

    def porcent(num):
        result = num / total * 100 
        return f"{int(result)}%"

    print("El total de filas es: ", total)
    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} -> {porcent(X_train.shape[0])}")
    print(f"Tamaño del conjunto de validación: {X_valid.shape[0]} -> {porcent(X_valid.shape[0])}")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} -> {porcent(X_test.shape[0])}")

    return X_train, X_valid, y_train, y_valid, X_test, y_test
