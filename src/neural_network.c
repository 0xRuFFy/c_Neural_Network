#include "neural_network.h"

#include "util.h"

void init_weights(NeuralNetwork *nn) {
    int i, j;

    for (i = 0; i < INPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            nn->hidden_weights[i][j] = randrange(-1, 1);
        }
    }
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < OUTPUT_SIZE; j++) {
            nn->output_weights[i][j] = randrange(-1, 1);
        }
    }
}

void init_network(NeuralNetwork *nn, ActivationFunction hidden_activation_function,
                  ActivationFunction output_activation_function) {
    int i;

    for (i = 0; i < INPUT_SIZE; i++) {
        nn->input[i] = 0;
    }
    for (i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden[i] = 0;
    }
    for (i = 0; i < OUTPUT_SIZE; i++) {
        nn->output[i] = 0;
    }
    for (i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden_error[i] = 0;
    }
    for (i = 0; i < OUTPUT_SIZE; i++) {
        nn->output_error[i] = 0;
    }
    for (i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden_delta[i] = 0;
    }
    for (i = 0; i < OUTPUT_SIZE; i++) {
        nn->output_delta[i] = 0;
    }

    init_weights(nn);
    nn->activation_function[0] = hidden_activation_function;
    nn->activation_function[1] = output_activation_function;
}

void print_network(NeuralNetwork *nn) {
    int i;

    printf("Input: ");
    for (i = 0; i < INPUT_SIZE; i++) {
        printf("%f ", nn->input[i]);
    }
    printf("\n");
    printf("Hidden Weights: ");
    for (i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            printf("%f ", nn->hidden_weights[i][j]);
        }
    }
    printf("\n");
    printf("Hidden: ");
    for (i = 0; i < HIDDEN_SIZE; i++) {
        printf("%f ", nn->hidden[i]);
    }
    printf("\n");
    printf("Output Weights: ");
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%f ", nn->output_weights[i][j]);
        }
    }
    printf("\n");
    printf("Output: ");
    for (i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", nn->output[i]);
    }
    printf("\n");
}

void set_input(NeuralNetwork *nn, double input[INPUT_SIZE]) {
    int i;

    for (i = 0; i < INPUT_SIZE; i++) {
        nn->input[i] = input[i];
    }
}

void feed_forward(NeuralNetwork *nn) {
    double sum = 0;
    int i, j;

    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < INPUT_SIZE; j++) {
            sum += nn->input[j] * nn->hidden_weights[j][i];
        }
        nn->hidden[i] = nn->activation_function[0].function(sum);
    }

    sum = 0;
    for (i = 0; i < OUTPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += nn->hidden[j] * nn->output_weights[j][i];
        }
        nn->output[i] = nn->activation_function[1].function(sum);
    }

    for (i = 0; i < OUTPUT_SIZE; i++) {
        nn->output[i] += OUTPUT_BIAS;
    }
}

void back_propagate(NeuralNetwork *nn, double expected[OUTPUT_SIZE]) {
    int i, j;

    for (i = 0; i < OUTPUT_SIZE; i++) {
        nn->output_error[i] = expected[i] - nn->output[i];
        nn->output_delta[i] = nn->output_error[i] * nn->activation_function[1].derivative(nn->output[i]);
    }

    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < OUTPUT_SIZE; j++) {
            nn->hidden_error[i] += nn->output_delta[j] * nn->output_weights[i][j];
        }
        nn->hidden_delta[i] = nn->hidden_error[i] * nn->activation_function[0].derivative(nn->hidden[i]);
    }

    // Update weights
    // Hidden -> Output
    for (i = 0; i < HIDDEN_SIZE; i++) {
        for (j = 0; j < OUTPUT_SIZE; j++) {
            nn->output_weights[i][j] += nn->hidden[i] * nn->output_delta[j] * LEARNING_RATE;
        }
    }

    // Input -> Hidden
    for (i = 0; i < INPUT_SIZE; i++) {
        for (j = 0; j < HIDDEN_SIZE; j++) {
            nn->hidden_weights[i][j] += nn->input[i] * nn->hidden_delta[j] * LEARNING_RATE;
        }
    }
}

void train(NeuralNetwork *nn, double input[INPUT_SIZE], double expected[OUTPUT_SIZE]) {
    set_input(nn, input);
    feed_forward(nn);
    back_propagate(nn, expected);
}

double __sum(double *arr, int size) {
    int i;
    double sum = 0;

    for (i = 0; i < size; i++) {
        sum += arr[i];
    }

    return sum;
}

void train_on_data(NeuralNetwork *nn, TrainingPair data[], int data_size) {
    int i;

    double sum_error = 0;

    for (i = 0; i < data_size; i++) {
        train(nn, data[i].input, data[i].expected);
        sum_error += __sum(nn->output_error, OUTPUT_SIZE);
    }
    printf("Average error: %2.6f\r", sum_error / data_size);
}

void train_on_data_for_epochs(NeuralNetwork *nn, TrainingPair data[], int data_size, int epochs) {
    int i;

    for (i = 0; i < epochs; i++) {
        printf("Epoch %4d: ", i);
        train_on_data(nn, data, data_size);
    }
    printf("\n");
}

double __sigmoid(double x) { return 1 / (1 + exp(-x)); }

double __sigmoid_derivative(double x) { return x * (1 - x); }

double __relu(double x) { return x > 0 ? x : 0; }

double __relu_derivative(double x) { return x > 0 ? 1 : 0; }

ActivationFunction sigmoid = {
    .function = &__sigmoid,
    .derivative = &__sigmoid_derivative,
};

ActivationFunction relu = {
    .function = &__relu,
    .derivative = &__relu_derivative,
};
