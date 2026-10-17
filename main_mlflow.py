"""Taller 3 MLOps - Pipeline de scikit-learn con MLflow (versión script).
"""

import matplotlib

matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_PATH = "/Users/pin22880@uvg.edu.gt/taller3_mlops"
CATALOGO = "workspace"
ESQUEMA = "default"

MODEL_NAME = f"{CATALOGO}.{ESQUEMA}.taller3_mlops_model"
FEATURE_TABLE = f"{CATALOGO}.{ESQUEMA}.taller3_mlops_features"
METRICA_PRINCIPAL = "test_f1"
TAG_PROYECTO = "taller3_mlops"
USAR_FEATURE_STORE = False


def configurar_mlflow():
    """Define el experimento y el Model Registry (Unity Catalog)."""
    mlflow.set_experiment(EXPERIMENT_PATH)
    mlflow.set_registry_uri("databricks-uc")


def cargar_datos():
    """Mismo dataset sintético y split que `main.ipynb`."""
    X, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        flip_y=0.04,
        class_sep=1.2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    return X, y, X_train_df, X_test_df, y_train, y_test, feature_names


def calcular_metricas(modelo, X, y):
    """Métricas del notebook original + balanced_accuracy y log_loss, sobre test."""
    y_pred = modelo.predict(X)
    y_proba = modelo.predict_proba(X)[:, 1]
    return {
        "test_accuracy": accuracy_score(y, y_pred),
        "test_precision": precision_score(y, y_pred, zero_division=0),
        "test_recall": recall_score(y, y_pred, zero_division=0),
        "test_f1": f1_score(y, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, y_proba),
        "test_balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "test_log_loss": log_loss(y, y_proba),
    }


def figura_matriz_confusion(modelo, X, y, titulo):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(modelo, X, y, ax=ax, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def figura_importancia(modelo, feature_names, titulo):
    """Coeficientes (regresión logística) o feature_importances_ (árboles)."""
    estimador = modelo.named_steps["model"]
    if hasattr(estimador, "coef_"):
        valores, etiqueta = estimador.coef_[0], "Coeficiente"
    else:
        valores, etiqueta = estimador.feature_importances_, "Importancia"
    serie = pd.Series(valores, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    serie.plot.barh(ax=ax)
    ax.set_xlabel(etiqueta)
    ax.set_title(titulo)
    fig.tight_layout()
    return fig


def entrenar_run_manual(run_name, pipeline, datos):
    """Entrena el pipeline y registra todo a mano: params, métricas, gráficas y modelo."""
    _, _, X_train_df, X_test_df, y_train, y_test, feature_names = datos
    with mlflow.start_run(run_name=run_name) as run:
        pipeline.fit(X_train_df, y_train)
        metricas = calcular_metricas(pipeline, X_test_df, y_test)

        estimador = pipeline.named_steps["model"]
        params = {f"model__{k}": v for k, v in estimador.get_params().items()}
        params["modelo"] = type(estimador).__name__
        params["scaler"] = type(pipeline.named_steps["scaler"]).__name__
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)
        mlflow.set_tags({"proyecto": TAG_PROYECTO, "tipo_logging": "manual"})

        for fig, ruta in [
            (figura_matriz_confusion(pipeline, X_test_df, y_test, run_name), "graficas/matriz_confusion.png"),
            (figura_importancia(pipeline, feature_names, run_name), "graficas/importancia_variables.png"),
        ]:
            mlflow.log_figure(fig, ruta)
            plt.close(fig)

        signature = infer_signature(X_train_df, pipeline.predict(X_train_df))
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train_df.head(5),
        )
    print(f"[manual] {run_name}: {run.info.run_id} -> {METRICA_PRINCIPAL}={metricas[METRICA_PRINCIPAL]:.4f}")
    return run.info.run_id, metricas


def entrenar_run_autolog(run_name, pipeline, datos):
    """Entrena con autolog: MLflow guarda solo params, métricas de train y el modelo."""
    _, _, X_train_df, X_test_df, y_train, y_test, feature_names = datos
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)
    try:
        with mlflow.start_run(run_name=run_name) as run:
            pipeline.fit(X_train_df, y_train)
            metricas = calcular_metricas(pipeline, X_test_df, y_test)
            mlflow.log_metrics(metricas)
            mlflow.set_tags({"proyecto": TAG_PROYECTO, "tipo_logging": "autolog"})
            fig = figura_importancia(pipeline, feature_names, run_name)
            mlflow.log_figure(fig, "graficas/importancia_variables.png")
            plt.close(fig)
    finally:
        mlflow.sklearn.autolog(disable=True)
    print(f"[autolog] {run_name}: {run.info.run_id} -> {METRICA_PRINCIPAL}={metricas[METRICA_PRINCIPAL]:.4f}")
    return run.info.run_id, metricas


def seleccionar_mejor_run():
    """Busca los runs del taller y los ordena por la métrica principal."""
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_PATH],
        filter_string=f"tags.proyecto = '{TAG_PROYECTO}' and attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRICA_PRINCIPAL} DESC"],
    )
    columnas = [
        "run_id",
        "tags.mlflow.runName",
        "tags.tipo_logging",
        "metrics.test_f1",
        "metrics.test_recall",
        "metrics.test_roc_auc",
        "metrics.test_accuracy",
    ]
    print(runs[columnas].to_string(index=False))
    mejor = runs.iloc[0]
    print(f"\nMejor run: {mejor['tags.mlflow.runName']} ({mejor['run_id']})")
    return mejor["run_id"]


def registrar_y_validar_champion(run_id, X_test_df, y_test):
    """Registra el modelo, le asigna el alias `champion` y lo valida sobre test."""
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    print(f"Registrado {MODEL_NAME} versión {version.version} con alias 'champion'")

    modelo_champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    y_pred = modelo_champion.predict(X_test_df)
    print("Validación del champion sobre test:")
    print(f"  accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print(f"  f1       = {f1_score(y_test, y_pred):.4f}")
    return modelo_champion


def feature_store_opcional(X, feature_names):
    """Guarda las variables en una tabla de Feature Engineering en UC. Si falla, se omite."""
    if not USAR_FEATURE_STORE:
        print("Feature Store desactivado (USAR_FEATURE_STORE = False).")
        return
    try:
        from databricks.feature_engineering import FeatureEngineeringClient
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        fe = FeatureEngineeringClient()
        features_pdf = pd.DataFrame(X, columns=feature_names)
        features_pdf.insert(0, "id", range(len(features_pdf)))
        features_sdf = spark.createDataFrame(features_pdf)

        if spark.catalog.tableExists(FEATURE_TABLE):
            fe.write_table(name=FEATURE_TABLE, df=features_sdf, mode="merge")
        else:
            fe.create_table(
                name=FEATURE_TABLE,
                primary_keys=["id"],
                df=features_sdf,
                description="Variables sintéticas del taller 3 de MLOps",
            )
        print(fe.read_table(name=FEATURE_TABLE).limit(5).toPandas())
    except Exception as e:
        print("Feature Store no disponible, se omite esta sección:", e)


def main():
    configurar_mlflow()
    datos = cargar_datos()
    X, y, X_train_df, X_test_df, y_train, y_test, feature_names = datos
    print("Forma de los datos:", X.shape)
    print("Distribución objetivo:", pd.Series(y).value_counts().to_dict())

    entrenar_run_manual(
        "baseline_logreg",
        Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, C=0.5))]),
        datos,
    )
    entrenar_run_manual(
        "improved_logreg_balanced",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced")),
        ]),
        datos,
    )
    entrenar_run_autolog(
        "autolog_random_forest",
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
            )),
        ]),
        datos,
    )

    mejor_run_id = seleccionar_mejor_run()
    registrar_y_validar_champion(mejor_run_id, X_test_df, y_test)

    feature_store_opcional(X, feature_names)


if __name__ == "__main__":
    main()
