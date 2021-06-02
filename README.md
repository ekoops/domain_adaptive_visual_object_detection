# Domain Adaptive Visual Object Detection

This is the reference repository for the [paper](domain_adaptive_visual_object_detection.pdf)

## Colab notebook
The Colab notebook used in this work is available [here](progettoML.ipynb).
Notice that, in order to replicate this work on the Colab platform,
the instructions available in [Usage](#usage) regarding folders and symlinks creation
or environment variables definition, has to be adapted
with appropriate actions on colab notebook and on the connected drive.

## Usage

### Download datasets
Please go to `datasets` directory and follow the instructions. After you have
done the Pascal VOC datasets and the Clipart1k dataset will be available
in `datasets/VOC` and `datasets/CLIPART`

### Download SSD implementation
The following command will download our customize SSD PyTorch implementation based
on [lufficc/SSD](https://github.com/lufficc/SSD)
```
bash download_SSD.sh
```

### Download CycleGAN implementation
The following command will download our customize CycleGAN PyTorch implementation based
on [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
```
bash download_CycleGAN.sh
```

### Download AdaIN implementation
The following command will download our customize AdaIN PyTorch implementation based
on [naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)
```
bash download_AdaIN.sh
```

### Train the baseline SSD model
The baseline SSD model training is perfomed over the Pascal VOC 2007 and Pascal VOC 2012 trainval splits.

#### Set the VOC_ROOT environment variable
In order to train the baseline SSD model, it is necessary to set the VOC_ROOT
environment variable telling the SSD implementation the parent folder in which
the Pascal VOC 2007 and Pascal VOC 2012 datasets are located. You can set it by using a command
like the following:

```
export VOC_ROOT=/absolute/path/to/this/repo/datasets/VOC
```

Please note that it is necessary to download in advance the datasets.

#### Perform the baseline SSD model training
Move inside the SSD folder and use the following command to train a
baseline SSD model using the configurations in `SSD/configs/vgg_ssd300_voc0712.yaml`:
```
python train.py --config-file configs/vgg_ssd300_voc0712.yaml
```
After this step, the obtained model will be available at `SSD/outputs/vgg_ssd300_voc0712/model_final.pth`

#### Use a pre-trained baseline SSD model
As an alternative, you can directly [download](https://drive.google.com/file/d/1-RiuI0qsv6ohVtGMGiUhApMWszpzR7mz/view?usp=sharing) a pre-trained baseline
SSD model and put it inside the folder `SSD/outputs/vgg_ssd300_voc0712`.


### Test the baseline SSD model
In order to test the baseline SSD model over the Clipart1k dataset, the [baseline SSD model training](#train-the-baseline-ssd-model) should be performed first.
Notice that is also necessary to [set the VOC_ROOT environment variable](#set-the-voc_root-environment-variable).

#### Set the CLIPART_ROOT environment variable
It is necessary to set the CLIPART_ROOT environment variable telling the SSD implementation the parent folder in which
the clipart dataset is located. You can set it by using a command
like the following:

```
export CLIPART_ROOT=/absolute/path/to/this/repo/datasets/CLIPART
```

#### Perform the baseline SSD model test
Move inside the `SSD` folder and use the following command to test the baseline SSD model over the Clipart1k dataset:
```
python test.py --config-file configs/vgg_ssd300_clipart.yaml
```
The above command will use the `SSD/configs/vgg_ssd300_clipart.yaml` config file.
After this step, the results will be available inside the folder
`SSD/outputs/vgg_ssd300_voc0712/inference/clipart_test`.

### Train CycleGAN
#### Prepare the environment
CycleGAN has to infer a mapping function from the source domain represented by
the VOC samples to the target domain represented by the Clipart1k samples.
The source domain dataset will be the merge of the Pascal VOC 2007 and 2012 trainval splits
whereas the target domain dataset will be the Clipart1k train split.
In order to allow this setting, it is necessary to let the CycleGAN implementation know which
are the folder containing the Pascal VOC dataset and the Clipart1k dataset. This can be done by
creating symlinks to the  `datasets/VOC` and `datasets/CLIPART` inside the `CycleGAN/inputs` folder:
```
ln -s datasets/VOC CycleGAN/inputs/trainA
ln -s datasets/CLIPART CycleGAN/inputs/trainB
```
Moreover, it is necessary to let the CycleGAN implementation know which are the right merges/splits
that has to be used in the training by defining the `trainA_FILTER_FILE` and `trainB_FILTER_FILE` environment variables:
```
export trainA_FILTER_FILE=/absolute/path/to/this/repo/datasets/VOC_trainval_merge.txt
export trainB_FILTER_FILE=/absolute/path/to/this/repo/datasets/CLIPART/clipart/ImageSets/Main/train.txt
```

#### Perform the CycleGAN training
Move inside the `CycleGAN` folder and use the following command to train CycleGAN:
```
python train.py --name VOC0712_ClipArt1k --dataroot inputs --display_id -1
  --verbose --save_latest_freq 500000 --save_epoch_freq 1
  --n_epochs 10 --n_epochs_decay 10 --lr 0.00001
```
Notice that, the above setting will save the trained model every 1 epoch.
In order to stop and restart the CycleGAN training, a command like the following has to be issued:
```
python train.py --name VOC0712_ClipArt1k --dataroot inputs --display_id -1
  --verbose --save_latest_freq 500000 --save_epoch_freq 1 --epoch_count 5
  --n_epochs 10 --n_epochs_decay 10 --lr 0.00001 --continue_train
```
The above command will let the training restart from the epoch 5.
Please refer to the [CycleGAN implementation](https://github.com/ekoops/pytorch-CycleGAN-and-pix2pix)
for more information about the used script parameters.

#### Use a pre-trained CycleGAN model
As an alternative, you can directly [download](https://drive.google.com/file/d/1AXNgju3XbHzaZEF6f3rvs8QOOnm8-Vtd/view?usp=sharing)
a pre-trained CycleGAN model. The zip content has to be put inside the folder `CycleGAN/checkpoints/VOC0712_ClipArt1k`.


### DT step
#### Prepare the environment
In the DT step the Pascal VOC2007 and Pascal VOC2012 trainval split are domain-transferred to the
ClipArt1k domain.
In order to allow this setting, it is necessary to let the CycleGAN implementation know which
is the folder containing the Pascal VOC datasets that has to be used in the test phase.
This can be done by creating symlinks to the `datasets/VOC` inside the `CycleGAN/inputs` folder:
```
ln -s datasets/VOC CycleGAN/inputs/testA
```
Moreover, it is necessary to let the CycleGAN implementation know which are the right merge
that has to be domain-transferred by defining the `testA_FILTER_FILE` environment variable:
```
export testA_FILTER_FILE=/absolute/path/to/this/repo/datasets/VOC_trainval_merge.txt
```
Finally, in order to perform the transferring only in one-verse, an alias for the trained model
has to be created:
```
ln -s CycleGAN/checkpoints/VOC0712_ClipArt1k/latest_net_G_A.pth CycleGAN/checkpoints/VOC0712_ClipArt1k/latest_net_G.pth
```

#### Perform the DT step
Move inside the `CycleGAN` folder and use the following command to perform the DT step:
```
python test.py --name VOC0712_ClipArt1k --model test --dataroot inputs/testA
  --preprocess none --verbose --no_dropout --num_test 16551 --results_dir ../datasets/ --no_real
```
After the completion, all the transferred images will be stored in `datasets/VOC0712_ClipArt1k`


### Baseline SSD fine-tuning with domain-transferred images and testing
#### Setting VOC_ROOT and CLIPART_ROOT environment variables
Since the domain-transferred VOC images has to be used instead of the original ones, 
the environment variables has to be set as follows:
```
export VOC_ROOT=/absolute/path/to/this/repo/datasets/VOC0712_ClipArt1k
export CLIPART_ROOT=/absolute/path/to/this/repo/datasets/CLIPART
```
#### Preparing the domain-transferred images
CycleGAN produces in the DT step a folder `datasets/VOC0712_ClipArt1k` with a similar to the `datasets/VOC` one.
However, `datasets/VOC0712_ClipArt1k` does not contain the annotation needed for the SSD fine-tuning.
In order to provide valid annotations, it is sufficient to reuse the same annotations of the
`datasets/VOC/VOC2007` and `datasets/VOC/VOC2012`folders by creating the following symbolic links:

```
ln -s datasets/VOC/VOC2007/SegmentationObject datasets/VOC0712_ClipArt1k/VOC2007/SegmentationObject
ln -s datasets/VOC/VOC2007/SegmentationClass datasets/VOC0712_ClipArt1k/VOC2007/SegmentationClass
ln -s datasets/VOC/VOC2007/ImageSets datasets/VOC0712_ClipArt1k/VOC2007/ImageSets
ln -s datasets/VOC/VOC2007/Annotations datasets/VOC0712_ClipArt1k/VOC2007/Annotations

ln -s datasets/VOC/VOC2012/SegmentationObject datasets/VOC0712_ClipArt1k/VOC2012/SegmentationObject
ln -s datasets/VOC/VOC2012/SegmentationClass datasets/VOC0712_ClipArt1k/VOC2012/SegmentationClass
ln -s datasets/VOC/VOC2012/ImageSets datasets/VOC0712_ClipArt1k/VOC2012/ImageSets
ln -s datasets/VOC/VOC2012/Annotations datasets/VOC0712_ClipArt1k/VOC2012/Annotations
```

#### Prepare the baseline SSD model for the fine-tuning
In order to allow the loading of the baseline SSD model for the fine-tuning,
the SSD implementation must locate the correct model `SSD/outputs/vgg_ssd300_voc0712/model_final.pth`.
Use the following two commands in order to create the right directories with the proper loading files:
```
mkdir SSD/outputs/vgg_ssd300_voc0712toclipart_ft
mkdir SSD/outputs/vgg_ssd300_voc0712toclipart_ft2
mkdir SSD/outputs/vgg_ssd300_voc0712toclipart_ft3
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712toclipart_ft/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712toclipart_ft2/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712toclipart_ft3/last_checkpoint.txt
```

#### Perform the baseline SSD model fine-tuning using the domain transferred images
In this implementation we provide three config files that can be used in order to replicate
the paper results obtained with 520, 1100 and 10000 iterations:
```
vgg_ssd300_voc0712toclipart_ft.yaml
vgg_ssd300_voc0712toclipart_ft2.yaml
vgg_ssd300_voc0712toclipart_ft3.yaml
```
Move inside the SSD folder and use the following commands to perform the two fine-tunings:
```
python train.py --config-file configs/vgg_ssd300_voc0712toclipart_ft.yaml
python train.py --config-file configs/vgg_ssd300_voc0712toclipart_ft2.yaml
python train.py --config-file configs/vgg_ssd300_voc0712toclipart_ft3.yaml
```
After this step, the obtained models will be available at
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft/model_final.pth`,
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft2/model_final.pth` and 
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft3/model_final.pth`.
The test results over the Clipart1k test split will be available in
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft/inference/clipart_test`, 
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft2/inference/clipart_test` and
`SSD/outputs/vgg_ssd300_voc0712toclipart_ft3/inference/clipart_test`.

#### Use an already fine-tuned SSD model
As an alternative, you can directly [download](https://drive.google.com/file/d/1-0E4di5RUl9I_Ix6WjqIXxTSLJq53qBb/view?usp=sharing)
an already fine-tuned SSD model and put it inside the folder
`SSD/outputs/SSD/outputs/vgg_ssd300_voc0712toclipart_ft`.

### Baseline SSD fine-tuning with style-transferred images and testing
#### Download VGG and decoder
The style transferring is performed using AdaIN. AdaIN need to use a pre-trained
model in order to perform style-transferring. Download the pre-trained
[VGG core](https://drive.google.com/file/d/19AVULdHwON36SQB07gMmXYe9QSp6cY6a/view?usp=sharing)
and [encoder](https://drive.google.com/file/d/1_cn49w4wzhGjxUd_q_pb-qgGvhZySPow/view?usp=sharing)
and put them in the `AdaIN/models` folder.

#### Prepare the environment
The `VOC_ROOT` and `CLIPART_ROOT` environment variable has to be
set in the following way:
```
export VOC_ROOT=/absolute/path/to/this/repo/datasets/VOC
export CLIPART_ROOT=/absolute/path/to/this/repo/datasets/CLIPART
```
Moreover, the following symbolic link has to be created:
```
ln -s AdaIN SSD/AdaIN
```

#### Prepare the baseline SSD model for the fine-tuning
In order to allow the loading of the baseline SSD model for the fine-tuning,
the SSD implementation must locate the correct model `SSD/outputs/vgg_ssd300_voc0712/model_final.pth`.
Use the following two commands in order to create the right directories with the proper loading files:
```
mkdir SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft
mkdir SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft2
mkdir SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft3
mkdir SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft4
mkdir SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft5
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft2/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft3/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft4/last_checkpoint.txt
echo "outputs/vgg_ssd300_voc0712/model_final.pth" > SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft5/last_checkpoint.txt
```

#### Perform the baseline SSD model fine-tuning using the style-transferred images
In this implementation we provide 5 configuration files that can be used in order to replicate
the paper results:
```
vgg_ssd300_voc0712_AdaINst_ft
vgg_ssd300_voc0712_AdaINst_ft2
vgg_ssd300_voc0712_AdaINst_ft3
vgg_ssd300_voc0712_AdaINst_ft4
vgg_ssd300_voc0712_AdaINst_ft5
```
The above configuration files uses different combination of `SOLVER.MAX_ITER`,
`ADAIN.LOADER.TRANSFER_RATIO` and `ADAIN.MODEL.ALPHA` parameters. 
Move inside the SSD folder and use the following commands to perform the fine-tunings:
```
python train.py --config-file configs/vgg_ssd300_voc0712_AdaINst_ft.yaml --enable_style_transfer
python train.py --config-file configs/vgg_ssd300_voc0712_AdaINst2_ft.yaml --enable_style_transfer
python train.py --config-file configs/vgg_ssd300_voc0712_AdaINst3_ft.yaml --enable_style_transfer
python train.py --config-file configs/vgg_ssd300_voc0712_AdaINst4_ft.yaml --enable_style_transfer
python train.py --config-file configs/vgg_ssd300_voc0712_AdaINst5_ft.yaml --enable_style_transfer
```
After this step, the obtained models will be available at
```
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft/model_final.pth
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft2/model_final.pth
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft3/model_final.pth
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft4/model_final.pth
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft5/model_final.pth
```
The test results over the Clipart1k test split will be available in
```
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft/inference/clipart_test
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft2/inference/clipart_test
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft3/inference/clipart_test
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft4/inference/clipart_test
SSD/outputs/vgg_ssd300_voc0712_AdaINst_ft5/inference/clipart_test
```