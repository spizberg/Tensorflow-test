# Enlaps technical test

Hello you!

You are about to do a technical test to be recruited as an intern in the R&D team to Enlaps, congrats!

This test should not be too difficult and aims at evaluating if you are familiar with the following technologies:
- Docker
- Tensorflow
- Basic ML understanding

## Goals

In this repo you'll find a deep neural network model saved as a tensorflow lite binary. It performs flower classification on images. 

The network is fairly basic, don't expect amazing results from it, and was trained from [this dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz).

Your goal is to :
- [ ] develop a command line interface (CLI) program in python
- [ ] which should be able to read all images in a folder (given as parameter to the CLI)
- [ ] perform image classification on each image using the provided model
- [ ] print the results in the console
- [ ] package the app in a docker image

You will be evaluated on the quality of your code (comments, formating, organization, ...), not on the actual inference scores. But they should be reasonable, proving you correctly handled the input model.

## Code delivery

Fork this project on your own gitlab account, push your solution and make a merge request once you want to deliver your work.

Please be sure to set your project visibility as `Public` and name the merge request as follows:
```
NAME_FirstName
```

In the process we will delete the MR on the public repo so other candidates won't see it.

## Docker execution command

The expected command line that should work:
```bash
# build image
docker build -t classification_app .
# define the image folder on host
export IMAGE_FOLDER=<YOUR IMG FOLDER>
# run the app
docker run --gpus all -it --rm -v ${IMAGE_FOLDER}:/images/ classification_app /images/ 2>/dev/null
```

## Output format
For each image in the input folder the app should print a line in the form:
```json
{"img_name.jpg": {"score":0.25, "class":"sunflower"}}
```
The score should be a confidence score in [0, 1]. And only the most likely class should be output.

Additionnal traces can be included but the app should offer an option to remove them.

:warning: :warning: :warning: 

The only traces allowed on `stdout` are results from prediction. 
All other traces (info, debug, ...) must be printed on `stderr`.

## Additional informations

The input of the network is a `180x180` RGB image array (`float32` in `[0,255]`).

The output of the network is a raw dense layer output, without normalization. It is up to you to find a way to compute the confidence score in [0, 1].

The network was trained and saved with Tensorflow version `2.7.0`.

No more than 4h hours should be dedicated to this test.

### How was trained the network?

```python
model = Sequential([
  # data augmentation
  layers.RandomFlip("horizontal",
                     input_shape=(img_height,
                                  img_width,
                                  3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  # model architecture
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# and compiled with
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
```

## Bonus

For those of you eager to go further here are some leads:
- Allow images to be loaded from an URL
- Allow images to be specified using a regexp (inference only on `images/*/img_???.jpg`)
- Train and submit your own model (we get to 75% validation accuracy without overfitting, using a 20% split for validation)
- Allow to load an external model, maybe other model formats ?
- you tell us!


