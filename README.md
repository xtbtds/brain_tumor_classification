This is a ***FastAPI + React JS + nginx*** application for brain tumor classification. 
It uses trained xgboost model with 75% accuracy. You can see how it was trained in [**notebook.ipynb**](https://github.com/xtbtds/brain_tumor_prediction/blob/main/backend/notebook.ipynb)  

# Table of contents
* [Usage](#usage)


**USAGE:**  
- `git clone https://github.com/xtbtds/brain_tumor_classification_mlzoomcamp`
- `cd <project folder>`
- `docker-compose up`
- wait for docker-compose to build and run images.
- go to `http://localhost` 
- click "select" button to select an image from your computer, then click upload to upload it to the service.
- wait a little bit (5-10 s) and you'll see the probabilies of different deceases

![](app-usage-gif.gif) 

**Note:** you don't need to download the whole repo if you don't want to. Another way to run the app is to use [this  docker-compose file](https://github.com/xtbtds/brain_tumor_classification/blob/main/pulled/docker-compose.yml). It pulles already built images from docker hub. Copy this file to your local machine and run `docker-compose up`. 
