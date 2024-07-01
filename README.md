# Lung Cancer Classification
This project is a FastAPI-based API for lung cancer classification. It provides endpoints to classify chest CT images into 4 categories based on input images. 

# Steps
* first I have maden an effort to train the model by my own model and a pre-trained model
* I have saved the trained model resulted from the pre-trained model
* Make the API project through FastAPI and load the model 

# Classes
* Adenocarcinoma
* Large Cell Carcinoma
* Normal
* Squamous Cell Carcinoma

## Results
  
<table>
<tr>
<td><img src="results/result1.png"></td>
<td><img src="results/result2.png"></td> 
</tr>
</table>


## Run The Project


Build the Docker image:
```bash
  docker build -t lung-cancer-app 
```
Run the Docker container:
```bash
  docker run -p 8000:8000 lung-cancer-app
```
Open your browser and acceess project through this URL:
 http://127.0.0.1:8000/ 