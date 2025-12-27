# Neural Network From Scratch in C++ (Quadratic Regression)

This project is a **from-scratch implementation of a Neural Network in C++**, built **without using any ML libraries**.
Every component — from matrix operations to backpropagation — is manually implemented to deeply understand how neural networks work internally.

---

## 🚀 What This Project Does

* Implements a **fully connected feedforward neural network**
* Learns a **non-linear function (quadratic regression: y = x²)**
* Uses **ReLU activation** and **Mean Squared Error (MSE)** loss
* Trains using **manual backpropagation and gradient descent**
* Achieves **near-zero training loss**

---

## 🧠 Why This Project Matters

Most ML projects rely on frameworks like PyTorch or TensorFlow.
This project avoids all abstractions and focuses on:

* Understanding **how forward propagation works**
* Implementing **backward propagation mathematically**
* Managing **weights, biases, and gradients manually**
* Designing a **layer-based architecture using OOP in C++**

This demonstrates **strong fundamentals in ML + C++**, not just library usage.

---

## 🏗️ Architecture Overview

### Core Components

* **Matrix Class**

  * Handles 2D numerical data
  * Supports indexing, transpose, and multiplication

* **Layer Base Class**

  * Abstract class enforcing `forward`, `backward`, and `update`

* **Dense Layer**

  * Fully connected layer
  * Stores weights, biases, and gradients
  * Implements backpropagation

* **ReLU Layer**

  * Applies non-linearity
  * Passes gradients only where input > 0

* **Mean Squared Error (MSE) Loss**

  * Computes loss and its derivative

* **Model Class**

  * Chains layers together
  * Handles forward pass, backward pass, and parameter updates

---

## 📐 Model Used

```
Input (1)
   ↓
Dense (1 → 8)
   ↓
ReLU
   ↓
Dense (8 → 8)
   ↓
ReLU
   ↓
Dense (8 → 1)
   ↓
Output
```

---

## 📊 Training Details

* **Dataset**

  ```
  x = [-2, -1, 0, 1, 2]
  y = x²
  ```

* **Loss Function**: Mean Squared Error

* **Optimizer**: Gradient Descent

* **Learning Rate**: 0.01

* **Epochs**: 1000

* **Batch Size**: 1 (SGD)

---

## 📉 Sample Training Output

```
Epoch 0   | Loss: 32.76
Epoch 300 | Loss: 0.01
Epoch 600 | Loss: 1.8e-05
Epoch 900 | Loss: 2.8e-08
```

---

## 🧪 Test Result

```
Prediction for x = 3
Output ≈ 7.0
Expected = 9
```

> Note: Neural networks are **interpolators**, not extrapolators.
> The model was trained only on values between `-2` and `2`, so extrapolation beyond this range is expected to be imperfect.

---

## ⚙️ How to Run

### Compile

```bash
g++ NN.cpp -o NN
```

### Run

```bash
./NN
```

(Works best on **Linux / WSL / MinGW-w64**)

---

## 🧩 Key Learning Outcomes

* Implemented **backpropagation from scratch**
* Understood the role of **weights, biases, and gradients**
* Learned why **ReLU extrapolates poorly**
* Designed an **OOP-based ML framework in C++**
* Gained confidence building ML systems without libraries

---

## 🔮 Future Improvements

* Add batching support
* Implement better activations (`tanh`, `sigmoid`)
* Add optimizers (Momentum, Adam)
* Save & load model weights
* Support multiple loss functions
* Plot predictions vs actual curve

---

## 📌 Disclaimer

This project is **educational by design**, not meant to replace optimized ML frameworks.
Its purpose is to **prove understanding**, not performance.

---

## 🙌 Author

Built with curiosity, debugging, and persistence.
If you’re learning ML internals — this is the hard (and rewarding) way.

---

⭐ If you found this useful, consider starring the repository!
