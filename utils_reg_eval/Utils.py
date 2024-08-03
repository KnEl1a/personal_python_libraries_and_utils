import matplotlib.pyplot as plt
import seaborn as sns

#from visualization_utils import plot_error_distribution
def plot_error_distribution(y_true, y_pred):
    """
    Esta función toma los valores verdaderos y las predicciones, calcula el error,
    y muestra un gráfico con un histograma de errores y una curva KDE de la densidad.
    
    :param y_true: Array de valores verdaderos.
    :param y_pred: Array de valores predichos.
    """
    # Calcular errores
    errors = y_true - y_pred.flatten()

    # Crear la figura y el eje principal
    fig, ax1 = plt.subplots()

    # Crear el histograma en el eje principal
    sns.histplot(errors, bins=20, edgecolor='k', alpha=0.7, ax=ax1)
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Errores de Predicción')

    # Crear un segundo eje que comparte el mismo eje x
    ax2 = ax1.twinx()

    # Crear el KDE en el segundo eje
    sns.kdeplot(errors, ax=ax2, color='r')
    ax2.set_ylabel('Densidad')

    # Modificar la grilla del eje y de la derecha
    ax2.grid(True, which='both', axis='y', color='r', alpha=0.3)

    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso
# plot_error_distribution(y_test, y_pred)

from sklearn.metrics import mean_absolute_error

def reg_evaluation(model, X_test, y_test, verbose=False):
    """
    Evalúa el rendimiento de un modelo de regresión en términos de MAE y precisión relativa.

    Parámetros:
    model: El modelo de regresión entrenado.
    X_test: Características del conjunto de prueba.
    y_test: Valores verdaderos del conjunto de prueba.
    verbose: Si es True, retorna las predicciones y las métricas de evaluación.

    Retorna:
    Opcionalmente, las predicciones, el MAE en porcentaje y el MAE absoluto si verbose=True.
    """
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f'MAE (Error Absoluto Medio) en el conjunto de prueba: \n--> {mae:.4f}\n')

    # Calcular el rango de los valores objetivo
    y_range = y_test.max() - y_test.min()

    # Calcular el MAE en porcentaje
    mae_percentage = (mae / y_range) * 100
    print(f'Precisión relativa del modelo según el MAE y el rango de valores de la variable objetivo: \n-> {mae_percentage:.2f}%')

    if verbose:
        return y_pred, mae_percentage, mae


#------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def more_reg_eval(y_test, y_pred):
    """
    Calcula y muestra métricas de evaluación para un modelo de regresión.

    Parámetros:
    y_test (array-like): Valores reales del conjunto de prueba.
    y_pred (array-like): Predicciones realizadas por el modelo sobre el conjunto de prueba.

    Métricas Calculadas:
    - MAE (Error Absoluto Medio): Mide el promedio de los errores absolutos entre las predicciones y los valores reales.
    - MSE (Error Cuadrático Medio): Calcula el promedio de los cuadrados de los errores.
    - RMSE (Raíz del Error Cuadrático Medio): Proporciona una medida del error en las mismas unidades que los datos.
    - R^2 (Coeficiente de Determinación): Mide la proporción de la varianza en la variable dependiente que es explicada por el modelo.

    Resultados:
    Imprime las métricas MAE, MSE, RMSE y R^2 en la consola.
    """
    # Calcular MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calcular MSE
    mse = mean_squared_error(y_test, y_pred)
    
    # Calcular RMSE
    rmse = np.sqrt(mse)
    
    # Calcular R^2
    r2 = r2_score(y_test, y_pred)
    
    # Imprimir resultados
    # print(f'\nMAE (Error Absoluto Medio): {mae}')
    print(f'\nMSE (Error Cuadrático Medio): {mse}')
    print(f'\nRMSE (Raíz del Error Cuadrático Medio): {rmse}')
    print(f'\nR^2 (Coeficiente de Determinación): {r2}')


def complete_reg_eval(model, X_test, y_test, y_pred):
    """
    Evalúa y presenta el rendimiento del modelo de regresión mediante diferentes métricas y visualizaciones.

    Parámetros:
    model: El modelo de regresión que se está evaluando.
    X_test (array-like): Datos de entrada del conjunto de prueba.
    y_test (array-like): Valores reales del conjunto de prueba.
    y_pred (array-like): Predicciones realizadas por el modelo sobre el conjunto de prueba.

    Funcionalidad:
    - Visualiza la distribución de errores mediante un gráfico.
    - Evalúa el modelo con métricas de rendimiento como MAE, MSE, RMSE y R^2.
    - Calcula y muestra métricas adicionales de evaluación del modelo.

    Uso:
    Llama a esta función con el modelo, el conjunto de prueba y las predicciones para obtener un análisis completo del rendimiento.
    """
    # Visualizar la distribución de errores
    plot_error_distribution(y_test, y_pred)
    
    # Evaluar el modelo con métricas de rendimiento
    reg_evaluation(model, X_test, y_test)
    
    # Calcular y mostrar métricas adicionales
    more_reg_eval(y_test, y_pred)
    
#complete_reg_eval(model,X_test,y_test, y_pred)