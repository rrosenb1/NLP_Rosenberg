# rrosenb1_msia490_2019
MSiA 490 - Text Analytics projects and notes

# How to Run API
1. Clone project branch
2. Install any necessary dependencies. Package runs primarily on standard data science packages sklearn and pandas, and on standard NLP package NLTK.
3. Download data from https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm and place in the same directory as the project repo.
4. Run intake.py, clean.py, and build_models.py to get the required .csv files, model files, and vectorizer file. This process should take 10-15 minutes to run locally, depending on your machine.
5. Run api.py in your GUI. Leave running and open a Terminal browser.
6. Test API with the code chunk:
  curl -X GET http://127.0.0.1:5000/ -d query='that movie was boring'
7. Replace query call in the curl code with anything you wish and enjoy seeing which astrological sign you are predicted to have!
