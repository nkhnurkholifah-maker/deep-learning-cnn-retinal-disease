import os

NOTEBOOK = "notebooks/deep_learning_cnn_retinal_disease.ipynb"

print("Executing notebook (eval wrapper): top to bottom...")
os.system(
    f"jupyter nbconvert --to notebook --execute {NOTEBOOK} "
    f"--output executed_eval.ipynb"
)
