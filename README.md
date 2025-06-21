# TinyLeafNet-IoT - Automated Tomato Leaf Disease Detection and Alert System using Internet of Things and TinyML
TinyLeafNet-IoT is a lightweight TinyML-based IoT system for real-time tomato leaf disease detection on microcontrollers. TinyML model, trained on the <a href="https://github.com/spMohanty/PlantVillage-Dataset">PlantVillage dataset</a>, achieves 97.2% accuracy and streams results via MQTT to a cloud dashboard. The solution offers low-power, cost-effective disease monitoring for precision agriculture.
<br/><br/>
## System Pipeline
<img src="images/system pipeline.png"/>

## Model Architecture

<img src="images/tinymlleafnet-iot_architecture.png"/>

## Results
<img src="images/results.png"/>

## How to use this repository?
### Dataset Prepararion
Download the dataset from <a href="https://github.com/spMohanty/PlantVillage-Dataset">PlantVillage dataset</a> and upload it to Edge Impulse and label it for image classification model.<br/><br/>
If you wish to train the model in jupyter notebook, export labelled dataset from Edge Impulse. Make sure your dataset has following directory structure for custom training.
```
tomato-leaf-disease-export/
├── training/
│   ├── image files
│   └── info.labels
└── testing/
    ├── image files
    └── info.labels
```
### Inferencing
To test **TinyLeafNet-IoT** model (trained using Edge Impulse), download the TensorFlow Lite <a href="https://github.com/tim3in/TinyLeafNet-IoT/blob/main/models/TinyLeafNet-IoT_EdgeImpulse.lite">Model</a> and run the code in <code>TinyLeafNet_IoT_Inferencing.ipynb</code>. <br/><br/>
<br/></br>
### Training
- Use `TinyLeafNet-IoT_Model_Architecture_EdgeImpulse.py` in the `/code` directory to train model in Edge Impulse.
- Use `TinyLeafNet_IoT_Custom_Training.ipynb` in the `/code` directory to train model in Jupyter Notebook.
<br/>
*Note:* The model is trained to detect five classes:<br/>
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Yellow Leaf Curl Virus
