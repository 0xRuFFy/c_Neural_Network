#include "util.h"

double randrange(double min, double max) { return (max - min) * ((double)rand() / (double)RAND_MAX) + min; }
