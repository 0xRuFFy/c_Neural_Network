#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <math.h>
#include <stdio.h>

#define INPUT_SIZE 4  // first bit, second bit, and, or, nand, xor,
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1  // result of binary operation
#define LEARNING_RATE 0.01
#define OUTPUT_BIAS 0.0001  // to avoid 0 output in relu activation function : else output of 0 will never be updated

typedef struct {
    double (*function)(double);
    double (*derivative)(double);
} ActivationFunction;

typedef struct {
    double input[INPUT_SIZE];
    double expected[OUTPUT_SIZE];
} TrainingPair;

typedef struct {
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];

    double hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE];

    double hidden_error[HIDDEN_SIZE];
    double output_error[OUTPUT_SIZE];

    double hidden_delta[HIDDEN_SIZE];
    double output_delta[OUTPUT_SIZE];

    ActivationFunction activation_function[2];
} NeuralNetwork;

void init_weights(NeuralNetwork *nn);
void init_network(NeuralNetwork *nn, ActivationFunction hidden_activation_function,
                  ActivationFunction output_activation_function);
void set_input(NeuralNetwork *nn, double input[INPUT_SIZE]);
void feed_forward(NeuralNetwork *nn);
void back_propagate(NeuralNetwork *nn, double expected[OUTPUT_SIZE]);
void train(NeuralNetwork *nn, double input[INPUT_SIZE], double expected[OUTPUT_SIZE]);
void train_on_data(NeuralNetwork *nn, TrainingPair data[], int data_size);
void train_on_data_for_epochs(NeuralNetwork *nn, TrainingPair data[], int data_size, int epochs);
void print_network(NeuralNetwork *nn);

extern ActivationFunction sigmoid;
extern ActivationFunction relu;

#endif
