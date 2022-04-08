#ifndef PREDEFINED_ERROR_FUNCTIONS_H
#define PREDEFINED_ERROR_FUNCTIONS_H

#include <math.h>
#include "rCover.h"

class QuantileLossComputer {
    int n_quantiles;
    double *h;
    int *h_low;
    int *h_up;
    double *y_low;
    double *under;
    double *above;
    double *y_pred;



    public: 
        QuantileLossComputer(int n_quantiles);

        ~QuantileLossComputer() {
            delete[] h;
            delete[] h_low;
            delete[] h_up;
            delete[] y_low;
            delete[] under;
            delete[] above;
            delete[] y_pred;
        }

        Error * quantile_tids_errors(RCover* cover);
};


float sse_tids_error(RCover* cover);


#endif