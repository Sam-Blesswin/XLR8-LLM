# 📘 Bigram Name Generator

This notebook walks through building a simple **character-level language model** to generate human-like names. It has two parts:

1. **Manual Bigram Counting** — a no-learning, statistics-based approach.
2. **Neural Network** — a tiny learning model that figures out bigrams on its own.

Both approaches lead to surprisingly similar results, illustrating the deep connection between counting and learning.

---

## 🧩 Part 1: Manual Bigram Counting

### 🧠 What We Do

We read a dataset of names and break them down into **bigrams** — pairs of consecutive characters (e.g., `'e' → 'm'` in "emma"). We add special `.` tokens to mark the beginning and end of each name.

For example, `"emma"` becomes:

* `.` → `e`
* `e` → `m`
* `m` → `m`
* `m` → `a`
* `a` → `.`

We count all such pairs across all names.

### Bigram Counting

* A bigram is a pair of consecutive characters (like 'a' followed by 'n').
* The model first counts how often each character is followed by another across all names. This gives us an idea of transition frequencies between characters.

### 📊 What We Get

* A matrix of counts showing how often each character follows another.
* These counts are normalized to create **probabilities** of transitions.
* We use these to randomly generate new names by sampling character-by-character.

### ✅ Why It Works

It’s direct, interpretable, and needs no training — just counting and normalizing. It’s a great baseline.

---

## 🧠 Part 2: Neural Network

### 🔍 What We Build

We construct a tiny neural network that learns the same bigram relationships — but instead of counting explicitly, it **learns a weight matrix** that maps each input character to a probability distribution over the next character.

Steps:

1. Convert each character to a **one-hot vector**.
2. Multiply by a **weight matrix** to get a vector of scores (logits).
3. Apply **softmax** to get probabilities.
4. Use **negative log-likelihood loss** to evaluate predictions.
5. Update the weights using **gradient descent**.

After enough training, the network generates names almost indistinguishable from those in Part 1.

---

## 🔁 Comparison: Manual vs Neural

Even though one method just counts and the other learns, they’re mathematically close:

| Concept        | Part 1: Manual Counting | Part 2: Neural Network      |
| -------------- | ----------------------- | --------------------------- |
| Basis          | Raw bigram counts       | Learned weights (logits)    |
| Probabilities  | Normalized counts       | Softmax of logits           |
| Training       | None                    | Gradient descent on loss    |
| Output quality | Surprisingly good       | Very similar                |
| Bonus trick    | `exp(logits)` ≈ counts  | Shows they're deeply linked |

### 🤯 Insight

The neural network doesn’t learn something magical — it learns **something equivalent to counting bigrams**, just encoded differently (log space). When you `exp(logits)`, the results align closely with the actual counts from Part 1.

---

## 🧠 Key Concepts

* **Bigram**: Pair of consecutive characters.
* **One-Hot Encoding**: A binary vector to represent each character.
* **Softmax**: Converts logits to probabilities.
* **Negative Log-Likelihood**: Measures how "wrong" the model is.

---
