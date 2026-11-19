import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def generate_sample_data(num_samples=1000, random_state=42):
    """
    Synthetic dataset for resume inflation detection.
    Features:
      1. claimed_skill: Rating claimed on resume (1.0 - 10.0)
      2. measured_ability: Actual score in adaptive interview (1.0 - 10.0)
      3. difference: claimed_skill - measured_ability
    Label:
      1 if inflated (claimed skill significantly higher than measured ability)
      0 otherwise
    """
    # Note: Sample synthetic data used for exam demo. Replace with real data in production.
    np.random.seed(random_state)
    claimed = np.round(np.random.uniform(2.0, 10.0, num_samples), 2)
    measured = np.round(np.random.uniform(1.0, 9.5, num_samples), 2)
    diff = np.round(claimed - measured, 2)
    
    # Base deterministic rule: inflation if gap > 2.0
    labels = (diff > 2.0).astype(int)
    
    # Real-life evaluation data is noisy due to human judgment variation;
    # randomly flipping ~6% labels ensures realistic accuracy (around 0.90 - 0.95) rather than an artificial 100%.
    noise_indices = np.random.choice(num_samples, size=int(0.06 * num_samples), replace=False)
    labels[noise_indices] = 1 - labels[noise_indices]
    
    # Feature matrix X: [claimed_skill, measured_ability, difference]
    X = np.column_stack([claimed, measured, diff])
    y = labels
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train Resume Inflation Detector with MLflow Tracking")
    parser.add_argument("--n_estimators", type=int, default=50, help="Number of trees in RandomForest")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of trees")
    parser.add_argument("--tracking_uri", type=str, default="http://127.0.0.1:6001", help="MLflow Tracking Server URI")
    args = parser.parse_args()

    # 1. Dataset generation with label noise & train-test split
    X, y = generate_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=99)

    # MLflow tracking server ka local URL configure kar rahe hain
    mlflow.set_tracking_uri(args.tracking_uri)

    # Experiment ka naam set kar rahe hain jisme ye saari runs organize hongi
    mlflow.set_experiment("HireWise-AI")

    # Naya tracking run start kar rahe hain context manager ke saath
    with mlflow.start_run() as run:
        print(f"[MLflow] Active Run ID: {run.info.run_id}")

        # Model hyperparameters ko MLflow run parameters mein log kar rahe hain
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": 42
        })

        # RandomForest Classifier model initialize aur train kar rahe hain
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Test set par evaluation metrics calculate kar rahe hain
        y_pred = model.predict(X_test)
        test_accuracy = float(accuracy_score(y_test, y_pred))
        test_f1 = float(f1_score(y_test, y_pred))

        print(f"[Results] test_accuracy: {test_accuracy:.4f}, test_f1: {test_f1:.4f}")

        # Test evaluation metrics (accuracy aur F1-score) ko MLflow mein log kar rahe hain
        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_f1": test_f1
        })

        # Quality Gate Check: agar model ki accuracy 80% se kam hai toh error raise hoga
        if test_accuracy < 0.80:
            raise ValueError(f"Quality gate failed: test_accuracy {test_accuracy:.4f} < 0.80. Model will not be registered!")

        print("[Quality Gate] Passed! Proceeding to register model in Model Registry...")

        # Trained scikit-learn model ko MLflow model registry mein log aur register kar rahe hain
        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name="hirewise-inflation-detector",
                serialization_format="cloudpickle"
            )
        except TypeError:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="hirewise-inflation-detector",
                serialization_format="cloudpickle"
            )

        # Model Registry ko manage karne ke liye MlflowClient ka instance bana rahe hain
        client = MlflowClient()

        # Is run dwara create kiya gaya registered model version number nikal rahe hain
        model_version = getattr(model_info, "registered_model_version", None)
        if not model_version:
            latest_versions = client.get_latest_versions("hirewise-inflation-detector")
            model_version = latest_versions[-1].version if latest_versions else "1"

        # Current champion model ki details aur accuracy check kar rahe hain
        current_champ_version = None
        current_champ_accuracy = None
        try:
            # Model registry se current champion alias wala model version retrieve kar rahe hain
            champ_mv = client.get_model_version_by_alias("hirewise-inflation-detector", "champion")
            current_champ_version = champ_mv.version
            # Champion model ko train karne wali MLflow run se test_accuracy metric fetch kar rahe hain
            champ_run = client.get_run(champ_mv.run_id)
            current_champ_accuracy = float(champ_run.data.metrics.get("test_accuracy", 0.0))
        except Exception:
            # Agar pehle se koi champion nahi hai toh ye pehla model champion banega
            current_champ_version = None
            current_champ_accuracy = None

        # Smarter promotion rule: naya model tabhi champion banega jab accuracy >= current champion
        if current_champ_accuracy is None or test_accuracy >= current_champ_accuracy:
            # Naye registered model version ko 'champion' alias assign kar rahe hain
            client.set_registered_model_alias(
                name="hirewise-inflation-detector",
                alias="champion",
                version=str(model_version)
            )
            prev_acc_str = f"{current_champ_accuracy:.4f}" if current_champ_accuracy is not None else "none"
            print(f"New champion: v{model_version} (accuracy {test_accuracy:.4f} >= previous {prev_acc_str}).")
        else:
            # Naya version register toh ho gaya, lekin champion alias change nahi hoga
            print(f"Champion unchanged (still v{current_champ_version}, accuracy {current_champ_accuracy:.4f}) because new model accuracy {test_accuracy:.4f} is lower.")

if __name__ == "__main__":
    main()
