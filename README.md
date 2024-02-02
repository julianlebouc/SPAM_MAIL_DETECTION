# Spam Mail Prediction with MLFlow
## Julian LEBOUC, Nicolas TACHET, Abel COUTOLLEAU, Andy HARAN

# How to run?

Clone the repository

```bash
https://github.com/julianlebouc/SPAM_MAIL_DETECTION
```

## Option 1 : With MLFlow Projects
Please be aware some dependencies might be missing !
### Training Pipeline :
```bash
mlflow run path/to/git/repo -e main
```

### App testing :
```bash
mlflow run path/to/git/repo -e app
```

## Option 2 : Manual install and use

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n spamdetection python=3.8 -y
```

```bash
conda activate spamdetection
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Training pipeline
```bash
python main.py
```

### STEP 04- App Testing
```bash
python app.py
```

## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui
