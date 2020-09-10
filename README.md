<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Text Classification with CNNs in PyTorch
The aim of this repository is to show a baseline model for text classification through convolutional neural networks in the PyTorch framework. 

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [The model](#the-model)
* [Files](#files)
* [Code structure](#code-structure)
* [How to use](#how-to-use)
* [License](#licence)
* [Contributing](#contributing)
* [Contact](#contact)

## 1. The model
Here I have to write about the model

## 2. Files
* **Pipfile**: Here you will find the dependencies that the model needs to be run.

* **main.py**: It contains the controller of pipelines (preprocessing and trainig)

* **src**: It contains three directories, which are: ``model``, ``parameters`` and ``preprocessing``.

* **src/model**: It contains two files, ``model.py`` and ``run.py`` which handles the model definition as well as the training/evaluation phase respectively.

* **src/parameters**: It contains a ``dataclass`` which stores the parameters used to preprocess the text, define and train the model. 

* **src/preprocessing**: It contains the functions implemented to load, clean and tokenize the text.

* **data**: It contains the data used to train the depicted model. 

## 3 Code structure
Here I have to talk about how the classes are connected

## 4. How to use
First you will need to install the dependencies and right after you will need to launch the ``pipenv`` virutal environment. So in order to install the dependices, you have to type:

```SH
pipenv install
```

right after you will need to launch the virtual environment such as:

```SH
pipenv shell
```

Then, you can execute the prepropcessing and trainig/evaluation pipelines easily, just typing:

```SH
python main.py
```

If you want to modify some of the parameters, you can modify the ``dataclass`` which has the following form:
```PY
@dataclass
class Parameters:

   # Preprocessing parameeters
   seq_len: int = 35
   num_words: int = 2000
   
   # Model parameters
   embedding_size: int = 64
   out_size: int = 16
   stride: int = 2
   
   # Training parameters
   epochs: int = 10
   batch_size: int = 12
   learning_rate: float = 0.001
```

## 5. License
Here I have to introduce the licence

## 6. Contributing
Here I have to describe the way how to contribute

## 7. Contact
Here I have to add my social networks


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/FernandoLpz/TextClassification-CNN-PyTorch/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/fernando-lopezvelasco/
[product-screenshot]: images/screenshot.png