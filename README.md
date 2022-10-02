# audio-classifier

Run training (including preprocessing the audio files) using the command:

```
python train.py --cnn --preprocess
```

Get predictions for a test set using

```
python infer.py --cnn 
```


# adapting for streamed data

model 
- 2 * 1D convolutional layers (+ReLU) (input 1*1*80 frame, output 64*1*80 map)
- LSTM layer (+tanh) (input 1*5120 and memory cell at t-1, output 500)
- Fully connected layer (+softmax, dropout)(input 500, output 3)

For a low-latency streaming model, I’d replace the 2D convolution with 1D convolution, to output a probability distribution at each timestep. This switch incurs a tradeoff between latency and input context, since the time dimension of the receptive field of each convolutional kernel is reduced from 14(?) to 1. To add time context back into the model, the 64 feature maps for each timestep are flattened, and a unidirectional LSTM layer is applied. 

This is followed by a fully connected layer with softmax to output a 1*3 probability distribution for each timestep. The model would be trained with CrossEntropyLoss function computed between the predicted and ground-truth probability distributions.

The lack of multi-category (or category switching) training data would make training an accurate model difficult. The LSTM layer would need to see examples of category switching in order to learn how to weight information from the current timestep and previous timesteps. Artificial switching data could be created by concatenating or cross-fading samples of different categories. Since this data would be very different from real switching data, it’s unclear whether this would improve classification accuracy for real data.
