from setuptools import setup, find_packages

## reading line by line annd storing them into a single variable
with open("/Users/luis/Documents/Programming/dev/0769_Beginner_Advanced_MLOPS_GCP_CICD/venv_0769_Beginner_Advanced_MLOPS_GCP_CICD/0769_02_Reservation_Prediction/Project01/requirements.txt") as f:
    requirements = f.read().splitlines()
    #print(requirements)

setup(
    name="MLOPS_Project_001",
    version="0.1",
    author="Luismbpr",
    packages=find_packages(),
    install_requires=requirements,
)
