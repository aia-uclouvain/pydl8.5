#ifndef PREDEFINED_ERROR_FUNCTIONS_H
#define PREDEFINED_ERROR_FUNCTIONS_H

#include <math.h>
#include "rCover.h"

struct QuantileResult {
    double * predictions = nullptr;
    float * errors = nullptr;

    QuantileResult(double * predictions, float * errors) : predictions(predictions), errors(errors) {}

    ~QuantileResult() {
        if (predictions)
            delete[] predictions;
        if (errors)
            delete[] errors;
    }
};

class QuantileLossComputer {
    int n_quantiles;
    double *h;
    int *h_low;
    int *h_up;
    double *y_low;
    double *under;
    double *above;



    public: 
        QuantileLossComputer(int n_quantiles);

        ~QuantileLossComputer() {
            delete[] h;
            delete[] h_low;
            delete[] h_up;
            delete[] y_low;
            delete[] under;
            delete[] above;
        }

        QuantileResult * quantile_tids_errors(RCover* cover);
};


float sse_tids_error(RCover* cover);


#endif