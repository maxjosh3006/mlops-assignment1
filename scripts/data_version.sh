#!/bin/bash

#init git repogit init
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/maxjosh3006/mlops-assignment1.git
git push -u origin main


# Initialize DVC
dvc init

# Add data directory to DVC
dvc add data/housing.csv

# Add and commit DVC files
git add data/.gitignore data/housing.csv.dvc
git commit -m "Add initial dataset version"

# Make changes to dataset (this will be done by experiment.py)
# ... 

# Add new version of dataset
# dvc add data/iris.csv
# git add data/iris.csv.dvc
# git commit -m "Update dataset version"

# Push DVC changes
dvc push 


# echo "# mlops-assignment1" >> README.md
# git add README.md
# git commit -m "Add README"
# git push origin main