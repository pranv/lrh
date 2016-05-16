# Learning RNN Hierarchies

## Introduction

Recognizing patterns in sequences requires detecting temporal **events** that are occurring at different levels of an abstraction hierarchy. An event is a simple specific pattern observed over time present in the sequence that is useful for recognizing a much more complex pattern. Lower level events are due to local structure in input stream (example: phonemes). Higher-level events could be a combination of several lower level events and even higher level events from the past (example: mood of the speaker depends on the conversational tone and words he speaks). RNNs can not only see a composition of such events like a regular NN, but they can also see the overall variation of these events over arbitrary gaps in time and hence are very powerful. 

In general vanilla RNNs are not that useful because they forget events from the past(belonging to any level of abstraction). This because of its multiplicative update rule for its hidden state, which is repeated over all the time steps, causes the memories of events to decay. Common and now successful approach to tackle this problem is to use the LSTM family of RNNs, which replace the multiplicative update rule with an additive update. This makes the RNN prone to explosion and makes it unstable, thus a protective gating mechanism is put in place. While this solves the Vanishing Gradients problem, a single LSTM layer won't give the best performance. There is abundant empirical evidence that suggests that stacking LSTMs (and RNNs in general) offers better performance compared to a single LSTM layer with the memory size fixed. If LSTMs can remember everything from the past and if LSTMs are already very deep in time, why stack them at all?

The most intuitive and commonly given reason is that lower RNNs specialize to local events, while the higher level RNNs can focus on more abstract events. For example seq2seq architecture uses 4-stacked LSTMs for the encoder to compress the input sequence to a fixed length vector. Other possible reasons for this include ease of optimization, reduction of number parameters required per cell of memory, increased non linear depth per time step and many more (it is still an open research question). But it is clear that stacking RNNs is essential for good performance on complex tasks.

If we can simultaneously do both of the following:

1. Solving the vanishing gradients problem and, at the same time 

2. Make our models better at handling events in multiple levels of abstraction

Using a single simpler model, such a system would be more efficient than an LSTM. The main objective of this work is to find such a model. Taking inspirations from previous methods and combining them with our novel contributions.

## Background

<br>We can split up our big RNN into multiple smaller RNN modules. A module could either be active or inactive at a particular time step. If a module stays frequently inactive, more memory retention capability it possesses - these are slow modules. If a module stays frequently active, less memory retention capability it possesses - these are fast modules. Thus a combination of slower and faster RNNs can together retain memory for longer durations and thus make recognition of patterns based on temporally distant events possible.

There have been a few attempts to do this in the past. Here we discuss their approaches, strengths and weaknesses:

1. __Chunker/Neural History Compressor (1991):__ It is a stack of simple RNNs. The lowest RNN layer gets actual inputs as input. Higher level take inputs only from the layer below it and give their outputs as inputs to the layer above it. Each of the RNN layers, starting from the lowest RNN are trained to predict the input it is going to receive in the next timestep, based on the history of inputs the RNN has received so far. This is an unsupervised step similar to greedy auto encoder training. The main trick is to activate a RNN at a level in proportion to the extent of failure by the RNN layer below it in predicting its current input. If predictions by lower RNN are frequently correct, then the RNN is rarely on, thus has longer memory. The higher-level RNN is now trained to predict its inputs from layer below it, which is only at timesteps where it failed to predict. This is done iteratively over all RNNs in a stack. *Each RNN layer has now learned to expect what is unexpected to the RNN below it*. Schmidhuber calls this history compression as predictability increases with more layers.
</br>__Pros__: Unsupervised. Triggers for higher RNNs are event driven, meaning RNN in higher layer can come in when an unexpected event occurs and gather all data it needs.
</br>__Cons__: Local predictability is necessary - not always possible. Cannot combine information from multiple levels effectively. Needs layer wise pretraining and then supervised fine tuning.

2. __Clockwork RNN (2014)__: This is a supervised variant of Chunker. RNNs are present in a cluster. Each RNN has a dedicated timer or a clock, which only activates the RNN module once per for every ___P___ time steps. P is chosen to form a hierarchy  (example P_i = 2^i, where i is layer index). Further they restrict connections to go only from slow RNNs to fast RNNs and not vice versa. This is similar to Chunker, except events don’t trigger RNNs, rather clock pulses do. They activate according to a predefined period. This allows for supervised training as RNNs specialize automatically at their level.
</br>__Pros__: Supervised. Has lesser complexity than a vanilla RNN due to restricted activity and connectivity scheme.
</br>__Cons__: The major problem with this is that it requires hand engineered clock periods, which depend on the memory requirements of the task. This varies widely from task to task. Thus a lot of domain knowledge is required to setup the initial hierarchy. Further, it is not a trivial task to set this up. If there is a lot of gap between activities of 2 RNN modules, the slower RNN could miss the contents in faster RNN's memory as it would decay with time. Lastly, the connection scheme between modules is not good, both intuitively and in practice. 

<hr>

## Proposed Method

We cast the learning process as a combination of 

1. learning to do the task and 
2. learning the hierarchical interconnected RNN architecture

That is to come up with a model that can:

1. Learn the hierarchy

2. Learn how the modules are interconnected

3. Learn to activate based on events

The methods for the last two have been developed and tested. The first one is tricky and this repo only tries to do that. Described how below


### Learning the hierarchy
<br>
This is the most important and extremely challenging aspect of designing the model. Clock frequencies (inverse of clock periods) are a really good characteristic of a RNN's position in a hierarchy. A RNN having a small frequency naturally has to depend on contents stored in other (faster) RNNs as it rarely gets any input. Accordingly, this low frequency RNN learns to operate at a more abstract form of inputs, thus forming the higher levels of the hierarchy. Conversely, fast clocks form lower levels of the hierarchy. This makes the clock frequency a sufficient parameter that describes the RNN's position in a hierarchy. Thus, learning clock periods of a set of RNNs is equivalent to learning the hierarchy.

This is more powerful than it seems. Learning a symmetric set of clock frequencies between 2 sets of RNNs is equivalent to learning the seq2seq model itself for example (see figure below). A stack of RNNs with continuously decreasing frequencies forms a abstraction pyramid. If this is combined with another set of RNNs, which is connected only to the top most RNN, but with continuously increasing frequencies, we now have a crude seq2seq model. 

![](https://cloud.githubusercontent.com/assets/8753078/11612806/46e59834-9c2e-11e5-8309-7a93aa72383c.png)

(Not claiming that this learning capability has been achieved, but just showing the representational power)

Learning clock frequencies is not as trivial as learning just another parameter. Clocks used in Clockwork RNN were binary clocks. If we move to a smoother version of it, i. e the sine wave, we now have to learn the frequency of this sine wave. This sine wave represents the activation pattern of a RNN module in the hierarchy.

Unfortunately, learning frequency directly is not possible. This is because of extremely large amount of local minima that is present. Consider the following example: current wave frequency is 1/4, but the required wave frequency is 1/8. If the frequency slightly decreases, to say 1/5 this frequency is actually worse than 1/4 as there is less agreement between 1/8  and 1/5 compared to 1/8 and 1/4. That is there is a local minima wherever there is an LCM between the clock periods. Thus learning frequency directly is not possible (learnt it the hard way weeks before ICML deadline)

Instead of operating in amplitude-time domain, we move to amplitude-frequency domain. That is express the wave we want as a set of DCT coefficients. Perform inverse DCT to get the wave and use it to activate the modules. The error derivatives are also transferred to frequency domain during backward pass using DCT. This does not have the above problem of minima. 

This can be viewed as regularization of activities in the frequency domain. There can be many ways to restrict the learnt frequency to have just one major frequency component:

1. L1 penalty over coefficients

2. Softmax over the coefficients for discriminative choosing of frequencies.

<hr>

The code in this repo is only for this purpose. The others are not here, but in a separate repo. They have been independently tested to work, but not as a whole unit.

Note: Due to some reasons, binary clocks seemed like a better fit. So instead of DCT bases, binary bases are used and this whole "transform" is just implemented as a dot product of a vector and a matrix.

