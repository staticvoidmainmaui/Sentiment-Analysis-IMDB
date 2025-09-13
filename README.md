\# Sentiment Analysis on IMDB (Work in Progress)



A hands-on ML project to classify movie reviews as positive or negative using a simple and handwritten TF-IDF + Logistic Regression baseline.  

This repo is a starting point and something personalized for AI Projects at UF.



IMDB Reviews received from friend, Mahdi Haque.  

Guidelines coming from inspiration of other repositories.



---



\## Quickstart



```bash

python -m venv .venv

\# Windows

.\\.venv\\Scripts\\Activate

\# macOS/Linux

\# source .venv/bin/activate

pip install -r requirements.txt

python train.py



---





```

\## Status



Work in progress; baseline runs and prints accuracy + demo predictions.



\*\*Sample Output (tiny sample space):\*\*



Accuracy: 0.333

&nbsp;             precision    recall  f1-score   support



&nbsp;          0      0.000     0.000     0.000         2

&nbsp;          1      0.333     1.000     0.500         1



&nbsp;   accuracy                          0.333         3

&nbsp;  macro avg      0.167     0.500     0.250         3

weighted avg      0.111     0.333     0.167         3



> 'This was amazing, I loved every minute.' -> positive

> 'The worst film I have ever seen.' -> positive



\## Roadmap



\- Expand dataset and preprocessing (full IMDB dataset)  



