import joblib
import numpy as np
import os

print("===================================")
print("   CREATING LIGHTWEIGHT KNN MODEL")
print("===================================")

OLD_MODEL = "knn_model.pkl"
NEW_MODEL = "knn_model_light.pkl"

print("\nLoading existing model...")

model = joblib.load(
    OLD_MODEL,
    mmap_mode="r"
)

print("Original model loaded.")
print("Training data shape:", model._fit_X.shape)
print("Original dtype:", model._fit_X.dtype)

# --------------------------------------------------
# Convert training data from float64 -> float32
# --------------------------------------------------

print("\nConverting training data to float32...")

X_float32 = np.asarray(
    model._fit_X,
    dtype=np.float32
)

print("New dtype:", X_float32.dtype)

# --------------------------------------------------
# Create a new KNN model
# --------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

new_model = KNeighborsClassifier(
    n_neighbors=model.n_neighbors,
    weights=model.weights,
    algorithm=model.algorithm,
    leaf_size=model.leaf_size,
    p=model.p,
    metric=model.metric,
    n_jobs=-1
)

# --------------------------------------------------
# Fit lightweight model
# --------------------------------------------------

print("\nFitting lightweight KNN...")

new_model.fit(
    X_float32,
    model._y
)

# --------------------------------------------------
# Save
# --------------------------------------------------

print("\nSaving lightweight model...")

joblib.dump(
    new_model,
    NEW_MODEL,
    compress=3
)

print("\n===================================")
print("       MODEL CREATED SUCCESSFULLY")
print("===================================")

print("Saved:", NEW_MODEL)

size_mb = (
    os.path.getsize(NEW_MODEL)
    / (1024 * 1024)
)

print(
    f"New model size: {size_mb:.2f} MB"
)

print("===================================")