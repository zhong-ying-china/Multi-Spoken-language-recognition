# [Spoken LangID]
[![Website](https://github.com/google/uis-rnn/workflows/Python%20application/badge.svg)](https://zeemo.ai/)[![Downloads](https://pepy.tech/badge/uisrnn)](https://pepy.tech/project/uisrnn) [![HuggingFace](https://codecov.io/gh/google/uis-rnn/branch/master/graph/badge.svg)](https://codecov.io/gh/google/uis-rnn) [![Documentation](https://img.shields.io/badge/api-documentation-blue.svg)](https://google.github.io/uis-rnn)
## Objective 
Spoken Language Identification (LID) is defined as detecting language from an audio clip by an unknown speaker, regardless of gender, manner of speaking, and distinct age speaker. It has numerous applications in speech recognition[**(our ASR api)**], multilingual machine translations, and speech-to-speech translations. 

Our model currently supports 13 languages: English, Spanish, Italian, French, German, Portuguese, Russian, Turkish, Vietnamese, Indonesian, Chinese, Japanese, and Korean.

## Technology
The model uses convolutional and recurrent neural networks trained on two thousands of hours of speech data(private). Approximately 150 hours of speech supervision per language.
If this technology interests you, please have a look at our API available on [lang]()

<img width='400' height='600' src='https://github.com/zhong-ying-china/Multi-Spoken-language-recognition/blob/main/network.png'><br/>


## Available models and languages
 The figure below shows a ACC (Accuracy) breakdown by languages of the **Fleurs**[(paper)](https://arxiv.org/pdf/2205.12446.pdf) test [**dataset**](https://www.tensorflow.org/datasets/catalog/xtreme_s#xtreme_sfleurstr_tr) using pretrained model.
 
|Lang|English|Spanish|Italian|French|German|Portuguese|Russian|Turkish|Vietnamese|Indonesian|Chinese|Japanese|Korean|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Fleurs|75.39| 95.48|95.37|79.88|96.63|96.84|88.77|86.54|99.07|90.83|91.32|68.31|92.67|

![](https://github.com/zhong-ying-china/Multi-Spoken-language-recognition/blob/main/fleurs.jpg)
     
## Environment Setup
Python 3.7
```
pip install -r requirements.txt
```
<img src='readme_images/fleurs.jpg'>

## Code Implementation
### **Audio Format** 
The wav files have 16KHz sampling rate, single channel, and 16-bit Signed Integer PCM encoding.

### **Features** 
As speech features, 80-dimensional log mel-filterbank outputs were computed from 25ms window for each 10ms. Those log mel-filterbank features were further normalized to have zero mean and unit variance over the training partition of the dataset.

### **Train model**

```
python train.py
```
### **Predict**
```
from vocab.vocab import Vocab
import librosa
vocab = Vocab("vocab/vocab.txt")
model = tf.saved_model.load('saved_models/lang14/pb/2/')
signal, _ = librosa.load(wav_path, sr=16000)
output, prob = model.predict_pb(signal)
language = vocab.token_list[output.numpy()]
print(language, prob.numpy()*100)

```


## API Documentations


## Contributing


## LICESES
