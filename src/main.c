#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural_network.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

TrainingPair data[] = {
    {{0, 0, 1, 0}, {0}}, {{1, 0, 1, 0}, {0}}, {{0, 1, 1, 0}, {0}}, {{1, 1, 1, 0}, {1}},
    {{0, 0, 0, 1}, {0}}, {{1, 0, 0, 1}, {1}}, {{0, 1, 0, 1}, {1}}, {{1, 1, 0, 1}, {1}},
};

int main(void) {
    srand(time(NULL));

    NeuralNetwork nn;
    init_network(&nn, sigmoid, relu);
    train_on_data_for_epochs(&nn, data, ARRAY_SIZE(data), 1000000);

    for (unsigned long i = 0; i < ARRAY_SIZE(data); i++) {
        set_input(&nn, data[i].input);
        feed_forward(&nn);
        printf("%d %c %d = %d >> %d (%f)\n", (int)data[i].input[0], data[i].input[2] ? '&' : '|', (int)data[i].input[1],
               (int)data[i].expected[0], nn.output[0] > 0.5 ? 1 : 0, nn.output[0]);
    }

    return 0;
}
