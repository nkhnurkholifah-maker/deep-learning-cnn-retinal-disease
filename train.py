import os

NOTEBOOK = "notebooks/deep_learning_cnn_retinal_disease.ipynb"

print("Executing notebook (train wrapper): top to bottom...")
os.system(
    f"jupyter nbconvert --to notebook --execute {NOTEBOOK} "
    f"--output executed_train.ipynb"
)
