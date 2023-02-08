
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


