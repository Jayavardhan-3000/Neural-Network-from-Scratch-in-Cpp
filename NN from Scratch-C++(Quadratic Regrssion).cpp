#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <cstdlib>

class Matrix {
    int rows_, cols_;
    std::vector<double> data_;
public:
//Created a default constructor
    Matrix() : rows_(0), cols_(0), data_() {}
//Creates a r x c mmatrix filled with 0.0's
    Matrix(int r, int c) : rows_(r), cols_(c), data_(r* c, 0.0) {}
//This let's us to write data_(i,j) instead of data_[r * cols_ + c];
    //This allows changes in the matrix
    double& operator()(int r, int c) {
        return data_[r * cols_ + c];
    }
    //This returns the data strictly forbidding any changes and being read only won't allow copying and hence it's much quicker
    double operator()(int r, int c) const {
        return data_[r * cols_ + c];
    }
    //Used them to return the rows and columns sizes if called
    int rows() const { return rows_; }
    int cols() const { return cols_; }
};

//Multiplication of the Matrices
Matrix multiply(const Matrix& A, const Matrix& B) {
//Dimensions are not matching --> Throw an error .... YEET....
    if (A.cols() != B.rows())
        throw std::runtime_error("Matrix shape mismatch");
//Created a matrix C of type Matrix
    Matrix C(A.rows(), B.cols());
//Multiplication, not nerdy enough implement optimal multiplications
    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < B.cols(); j++)
            for (int k = 0; k < A.cols(); k++)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}
//Transposes the given matrix
Matrix transpose(const Matrix& A) {
    Matrix T(A.cols(), A.rows());
    for (int i = 0; i < A.rows(); i++)
        for (int j = 0; j < A.cols(); j++)
            T(j, i) = A(i, j);
    return T;
}
// Class for Layer Bases
class Layer {
public:
    //Abstracted cause they have to be implemented at any cost by his students
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual void update(double lr) = 0;
    //Deconstructor...to freee up the memory
    virtual ~Layer() {}
};
//  A class for Dense Layers
class DenseLayer : public Layer {
// Created a matrix for weights, bias, input(temporarily), gradients of weights and biases
    Matrix W, b;
    Matrix input_cache;
    Matrix dW, db;
public:
    DenseLayer(int in, int out) : W(in, out), b(1, out), dW(in, out), db(1, out) {
//Initialise with random weights
        for (int i = 0; i < in; i++)
            for (int j = 0; j < out; j++)
                W(i, j) = ((double)rand() / RAND_MAX - 0.5) * 0.1;
    }
//Forward method
    Matrix forward(const Matrix& input) override {
        // Y = W.X+B
        input_cache = input;
        Matrix out = multiply(input, W);
        for (int j = 0; j < out.cols(); j++)
            out(0, j) += b(0, j);
        return out;
    }
    //Back propagation method
    Matrix backward(const Matrix& grad_output) override {
        dW = multiply(transpose(input_cache), grad_output);
        for (int j = 0; j < db.cols(); j++)
            db(0, j) = grad_output(0, j);
        return multiply(grad_output, transpose(W));
    }
    //Need to update the values
    void update(double lr) override {
        for (int i = 0; i < W.rows(); i++)
            for (int j = 0; j < W.cols(); j++)
                W(i, j) -= lr * dW(i, j); //lr = learning rate
        for (int j = 0; j < b.cols(); j++)
            b(0, j) -= lr * db(0, j);
    }
};
// Activation Layer
// Here we chose ReLU cause it's easy to write. 
//ReLU makes values less than 0-> 0 and the other -> 1
class ReLULayer : public Layer {
    Matrix input_cache;
public:
    Matrix forward(const Matrix& input) override {
        input_cache = input;
        Matrix out = input;
        for (int i = 0; i < out.rows(); i++)
            for (int j = 0; j < out.cols(); j++)
                out(i, j) = std::max(0.0, out(i, j));
        return out;
    }
    Matrix backward(const Matrix& grad_output) override {
        Matrix grad = grad_output;
        for (int i = 0; i < grad.rows(); i++)
            for (int j = 0; j < grad.cols(); j++)
                grad(i, j) *= (input_cache(i, j) > 0);
        return grad;
    }
//Nothing bro could do to update
    void update(double) override {}
};
//A Loss function: Weights and biases get changed according to the losses (how far were the predictions from the expected outputs
class MeanSquaredError {
public:
    // mean((y_true - y_pred)^2)
    double forward(const Matrix& y_pred, const Matrix& y_true) {
        double loss = 0.0;
        int n = y_true.cols();
        for (int j = 0; j < n; j++) {
            double diff = y_true(0, j) - y_pred(0, j);
            loss += diff * diff;
        }
        return loss / n;
    }
    // 2 * (y_pred - y_true) / n
    Matrix backward(const Matrix& y_pred, const Matrix& y_true) {
        int n = y_true.cols();
        Matrix grad(1, n);
        for (int j = 0; j < n; j++)
            grad(0, j) = 2.0 * (y_pred(0, j) - y_true(0, j)) / n;
        return grad;
    }
};
//A class for Model
class Model {
    //Maintains aquisition and aslo automatically gets erased after the scope expires
    std::vector<std::unique_ptr<Layer>> layers;
public:
    void add(std::unique_ptr<Layer> l) {
        //add the layers into the vectors
        layers.push_back(std::move(l));
    }
    Matrix forward(const Matrix& x) {
        Matrix out = x;
        for (auto& l : layers)
            //L1->L2->L3->....->Ln
            out = l->forward(out);
        return out;
    }
    void backward(Matrix grad) {
        //Ln->....->L3->L2->L1
        for (int i = layers.size() - 1; i >= 0; i--)
            grad = layers[i]->backward(grad);
    }
    void update(double lr) {
        for (auto& l : layers)
            l->update(lr);
    }
};

int main() {
    //Sets seed to maintain reductibility
    srand(42);
    //Creating layers and activation layers
    Model model;
    model.add(std::make_unique<DenseLayer>(1, 8));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<DenseLayer>(8, 8));
    model.add(std::make_unique<ReLULayer>());
    model.add(std::make_unique<DenseLayer>(8, 1));
    
    //The Loss Functon
    MeanSquaredError loss;
    double lr = 0.01;
    //The DATASET (xs -> train, ys -> labels)
    std::vector<double> xs = { -2, -1, 0, 1, 2 };
    std::vector<double> ys;
    //Making the labels, the actual outputs
    for (double x : xs)
        ys.push_back(x * x);
    //Training
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0.0;
        //Had to set them up as matrices, as the data througout the code is matrices
        for (size_t i = 0; i < xs.size(); i++) {
            Matrix x(1, 1), y(1, 1);
            x(0, 0) = xs[i];
            y(0, 0) = ys[i];
            //Sent them through forward
            Matrix pred = model.forward(x);
            total_loss += loss.forward(pred, y);//Adding up the loss for each batch (the batch size is 1, causes I haven't implemented them)
            //Back Propagation
            Matrix grad = loss.backward(pred, y);
            model.backward(grad);
            model.update(lr);
        }
        //Displays what's happening
        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss << "\n";
    }
    //Testing
    Matrix test(1, 1);
    test(0, 0) = 3.0;
    Matrix out = model.forward(test);
    std::cout << "\nPrediction for x=3: " << out(0, 0) << " (expected 9)\n";
    return 0;
}
