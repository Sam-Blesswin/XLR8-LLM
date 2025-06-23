# Neural Network Character-Level Word Generator

This is a **learning project** inspired by foundational work in neural language models ([Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)). It builds a simple character-level neural network from scratch using PyTorch to generate new words, learning from a dataset of \~17,000 real words.


## 📂 Project Overview

### Dataset

* A plain text file of \~17,000 words.
* The model learns character patterns to predict the next character in a word.

### Problem Setup

* We treat words as sequences of characters (a-z plus `.` for start and end of word).
* There are **27 unique characters**, represented using **2-dimensional embeddings** for simplicity.
* We use a **context window (block size) of 3**, meaning the model looks at 3 characters to predict the next one.

### Concepts
This is an entry-level project intended to build intuition around:

* Embeddings
* Neural network basics
* Cross entropy loss
* Batching
* Hyperparameter tuning


## 🧱 Core Components

### 🏗 Model Architecture

* **Input**: A block of 3 characters (context)
* **Step 1**: Map each character to its embedding
* **Step 2**: Flatten and feed into a small neural network
* **Step 3**: Output a probability distribution over the next character

Think of this as teaching the model: “Given `emm`, what’s the most likely next character?”

---

## 🔄 How It Works: Simplified Example

Given the name `"emma"`, we process it character by character using a **context window of size 3** to predict the next character. We add special padding (e.g. `"."`) to represent the start of the word.

### Step-by-step Breakdown:

1. Add padding: `"emma"` becomes `".emma."`
2. Create input-target pairs using 3-character windows:

| Input (Context) | Target (Next Char) |
| --------------- | ------------------ |
| `...`           | `e`                |
| `..e`           | `m`                |
| `.em`           | `m`                |
| `emm`           | `a`                |
| `mma`           | `.` (end of word)  |

Each 3-character context is passed into the model to predict the next character.

### What the Model Learns:

* The probability of a character appearing after a given sequence of 3 characters
* For example: after seeing `emm`, the model learns that `a` is a common next letter

This approach teaches the model to learn valid patterns in name construction, entirely from data.

## 🌱 How We Generate New Names (Sampling)

Once the model is trained, we can use it to generate entirely new names by **sampling one character at a time**.

### 🧠 The Final Output Layer

* The last layer of the neural network has **27 neurons**
* Each neuron represents one possible **next character** (26 letters + `.`)
* The output is a **probability distribution** over those 27 characters (i.e., what the model thinks is likely to come next)

### 🔁 Sampling Step-by-Step

1. **Start with a context** of 3 special start tokens: `[".", ".", "."]`
2. Pass this context into the model → get a vector of 27 probabilities
3. Use those probabilities to **randomly sample** the next character
4. Shift the context window by 1 and add the new character
5. Repeat until the model predicts the end-of-word character `"."`

### Example

Let’s say we start with `"..."`, and the model outputs:

```text
a: 0.05, b: 0.01, ..., e: 0.20, ..., z: 0.00, .: 0.02
```

* The most likely next character might be `"e"`
* But we sample randomly using this distribution—so `"e"` is more likely, but `"a"` or `"."` could still happen

This randomness makes every generated name unique, and not just a "most likely" copy of training data.

---

### 🔤 Embeddings

Rather than representing characters using sparse **one-hot vectors**, we use a **learned lookup table** called an embedding matrix `C`.

* **Shape**: `[27, D]` where `D` is the embedding dimension (starts with `D=2`)
* Each character is assigned a dense vector like `[0.32, -0.14]` that evolves during training
* This allows the model to learn **similarities between characters** (e.g., vowels might cluster)

We index into it using:

```python
emb = C[X]  # where X are the character indices
```

### 🧮 Why Use Cross-Entropy Loss?

`F.cross_entropy()` in PyTorch is:

* **Efficient**: Combines `log_softmax` and `nll_loss` in one function.
* **Numerically stable**: Avoids overflow by offsetting large log values internally. Avoids potential issues with very large positive logs (e.g., `inf` values)
* **Offset invariant**: Adding a constant to logits doesn't change output probabilities. Resilient to large or offset logits (e.g., when values are shifted, softmax outputs stay the same)

### 🗃️ Batching

Training on multiple samples at once (mini-batch) allows:

* Better gradient estimates
* GPU acceleration
* Smoother learning

### 🎯 Learning Rate

* Controls how big the steps are during optimization.
* If too high: model diverges. If too low: training is very slow.
* We determine the learning rate by experimentation—trying different values and observing loss trends. Typically, values like `0.1`, `0.01`, `0.001` are tested, and loss curves are plotted to find a stable yet fast-learning rate.

### 🛠️ Fine-Tuning and Hyperparameters

You can tweak and tune the model by adjusting:

| Parameter         | Description                                |
| ----------------- | ------------------------------------------ |
| `weights`         | Initialized randomly; trained via backprop |
| `bias`            | Initialized randomly; trained via backprop |
| `neurons`         | Size of hidden layer (e.g., 100)           |
| `learning rate`   | Speed of learning (e.g., 0.1)              |
| `training cycles` | Number of iterations over data             |

Experiment with:

* Different embedding sizes: 2 → 10 → 
* Deeper networks or different non-linearities
* More context (block size > 3)

We observed that increasing the embedding dimension and neuron count **improves performance**, but also **increases training time** and **risk of overfitting**, especially on a small dataset like `names.txt`.

---

