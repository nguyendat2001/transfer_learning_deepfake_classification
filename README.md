## abstract

Fake images bring fake news that causes great consequences for victims, causes many bad effects in society, and causes economic damage. With the help of information technology, these fake images can be created by embedding and retouching the victim's face so sophisticated that it is difficult to distinguish with the naked eye. This study proposes methods to detect fake faces in images using well-known convolutional neural network architectures, including Inception, EfficientNet, MobileNet, and Xception. The study is carried out on datasets consisting of more than 20,000 images to perform fake face detection with the considered transfer learning techniques. Experimental results reveal that Xception has achieved the best performance in most cases while MobileNet has reached promising performance, although it owns the fewest parameters.

## result 

The architecture proposed is trained for 100 epochs with a batch size of 8. So that the training process can converge faster, we use RELU activation to evaluate the performance of a model. Besides, We use two metrics, including the accuracy score, the area under the curve (AUC) of the receiver operating characteristics (ROC), and the F1-score that metrics measure the classification results of two classes, fake and real. Accuracy is the fraction of predictions our model got right. AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is the probability that the model ranks a random positive example more highly than a random negative example. F1-Score is the harmonic mean of precision and recall. F1-Score is the best metric to evaluate a model in which the dataset is unbalanced.

|		|		|	Xception	|	MobileNet	|	EfficientNet	|	InceptionV3	|	Meso-4	|	Meso-inception-4	|
|-------|------------|---------|-------|------------|---------|-------|------------|
|		|	Loss	|	0.0338	|	0.0233	|	0.06126	|	6.32302	|	NA	|	NA	|
|		|	Accuracy	|	0.99727	|	0.99259	|	0.98885	|	0.64499	|	NA	|	NA	|
|Training|	AUC	|	0.99856	|	0.99885	|	0.93471	|	0.70654	|	NA	|	NA	|
|		|	Loss	|	0.47293	|	0.84083	|	1.86926	|	6.34912	|	NA	|	NA	|
|Testing|	Accuracy	|	0.93803	|	0.91056	|	0.81233	|	0.64298	|	0.891	|	0.917	|
|       |	AUC	|	0.97638	|	0.94557	|	0.81073	|	0.692	|	NA	|	NA	|


## Installation

install environment (recommended)
<details><summary> <b>Expand</b> </summary>

``` shell
# pip install required packages
pip install requirement.txt
```

</details>

## make data 

load data from folder and save it to csv form 

``` shell
python create_csv.py --dataset_dir ./data/train --output_dir data.csv
```

## Training

Data preparation

``` shell
python train.py --savedir_base ./output --datadir data.csv --base vgg19 --batch_size 32 --im_size 512 --epochs 100 --opt adam
```

## test best model
after obtain best model we started evaluation model by test.py



<div align="center">
    <a href="./">
        <img src="" width="59%"/>
    </a>
</div>


## experiments


