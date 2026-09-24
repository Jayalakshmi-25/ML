
from pathlib import Path
import joblib

model_files = [
    "kmeans_model_v1.pkl",
    "kmodes_model_v1.pkl",
    "Agglomerative_model_v1.pkl",
    "kprototypes_model_v1.pkl",
    "dbscan_model_v1.pkl",
    "meanshift_model_v1.pkl",
]

for file_name in model_files:
    model_path = Path(file_name)
    if not model_path.exists():
        print(f"SKIP: {file_name} not found")
        continue

    print(f"\n========================================")
    print(f"MODEL: {file_name}")
    print("========================================")

    model = joblib.load(model_path)
    print("TYPE:", type(model))
    print("MODEL:", model)

    try:
        print("PARAMETERS:", model.get_params())
    except Exception as e:
        print("PARAMETERS: unavailable ->", e)

    for attr in [
        "cluster_centers_",
        "labels_",
        "n_clusters",
        "components_",
        "inertia_",
        "epsilon",
        "core_sample_indices_",
    ]:
        if hasattr(model, attr):
            value = getattr(model, attr)
            print(f"{attr}: {value}")

    print("ATTRIBUTES:", list(getattr(model, "__dict__", {}).keys()))

print("\nALL MODELS DISPLAYED")