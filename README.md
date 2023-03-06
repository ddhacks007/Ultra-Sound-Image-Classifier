# Ultra-Sound-Image-Classifier

### walkthrough video https://www.youtube.com/watch?v=FVgKaKrnFJo&ab_channel=deepakraju

Ultra sound image classifier for classifying Lung pathalogies
<li>Covid</li>
<li> Pneumonia</li>
<li> Healthy</li>

In this repository I have trained & validated three popular models
<li> VGG_CAM for getting visual attentions for interpretability (Train-Acc: 89%, Test-Acc: 91%) </li>
<li> BALANCED_VGG_CAM for mitigating the problem of im-balanced classes. (Train-Acc: 93.9%, Test-Acc: 91.7%) </li>
<li> EmergencyNet for reducing model complexity and finding potential trade-offs between run-time and accuracy. </li>

Accuracy of EmergencyNet on different Fusion Techniques
<li> Max: Train-Acc: 80% Test-Acc: 90.7% </li>
<li> Avg: Train-Acc: 99% Test-Acc: 90.6% </li>
<li> Concatenate: Train-Acc: 99.2% Test-Acc: 88.6% </li>
<li> Add: Train-Acc: 99.7% Test-Acc: 88.6% </li>

<br> Go through the ipython notebook which is present inside the models folder for statistical details of the models and results </br>

#### There are some raw-data which couldn't be automated for downloading so please make sure you download the raw-data.zip from the below google-drive link
https://drive.google.com/file/d/1IlXOBp6m-dDqLKrcVxBUFxsbYpVbPGBX/view?usp=share_link 

#### run the below command to download and process the data which will take care of all the processing and conversion for you
<li> export PYTHONPATH=$PYTHONPATH:. </li>
<li> source DataExtracter/.env </li>
<li> python3 DataExtracter/Main.py </li>

### The videos will be converted to frames and will be moved to the data/output folder

#### Go to the models directory 
You can run the model of your choice by using the following commands below
<li> python3 VGG_CAM.py -d ../data/output/ </li>
<li> python3 BALANCED_VGG_CAM.py -d ../data/output/ </li>
<li> python3 EMERGENCY_NET.py -d ../data/output/ </li>


#### Go to the ipynb notebook present in the models directory for running and visualising the results


