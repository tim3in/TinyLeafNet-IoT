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
- Download the dataset from <a href="https://github.com/spMohanty/PlantVillage-Dataset">PlantVillage dataset</a> and upload it to <a href="https://edgeimpulse.com/">Edge Impulse</a> and label it for image classification model.
- If you wish to train the model in jupyter notebook, export labelled dataset from <a href="https://edgeimpulse.com/">Edge Impulse</a>. Make sure your dataset has following directory structure for custom training.
```
tomato-leaf-disease-export/
├── training/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── ...
│   └── info.labels
└── testing/
    ├── image_1.jpg
    ├── image_2.jpg
    ├── ...
    └── info.labels
```
Edge Impulse uses following JSON label format for image classification models.<br/>

```json
{
  "version": 1,
  "files": [
    {
      "path": "image_1.jpg",
      "name": "image_1",
      "category": "training",
      "label": {
        "type": "label",
        "label": "Yellow_Leaf_Curl_Virus"
      }
    },
    {
      "path": "image_2.jpg",
      "name": "image_2",
      "category": "training",
      "label": {
        "type": "label",
        "label": "Late_Blight"
      }
    }
  ]
}
```
### Training
- To train model with <a href="https://edgeimpulse.com/">Edge Impulse</a> choose `MobileNetV2 96x96 0.35` base model in Transfer Learning settings. Then in the Neural Network setting, switch to keras (expert) mode that will open code window. Use Neural Network code from `TinyLeafNet-IoT_Model_Architecture_EdgeImpulse.py` file to add the custom classification head and update training settings and hyperparameters.
- For custom training use `TinyLeafNet_IoT_Custom_Training.ipynb` file in the `/code` directory to train model in Jupyter Notebook.

### Inferencing
- To test **TinyLeafNet-IoT** model (trained using Edge Impulse), download the TensorFlow Lite <a href="https://github.com/tim3in/TinyLeafNet-IoT/blob/main/models/TinyLeafNet-IoT_EdgeImpulse.lite">Model</a> and run the code in `TinyLeafNet_IoT_Inferencing.ipynb`.
- To test **TinyLeafNet-IoT** model (trained using jupyter notebook), download the <a href="https://github.com/tim3in/TinyLeafNet-IoT/blob/main/models/TinyLeafNet-IoT_Custom.h5">Model Weights</a> and run the inference code provided in `TinyLeafNet_IoT_Custom_Training.ipynb`.

*Note:* The model is trained to detect five classes:<br/>

- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Yellow Leaf Curl Virus
