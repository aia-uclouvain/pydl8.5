#include "predefined_error_functions.h"

float sse_tids_error(RCover* cover) {
    RCover::iterator it;


    // Computing the mean of the targets
    float sum = 0.;
    int count = 0;
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        sum += cover->dm->getY(idx);
        count += 1;
    }

    float centroid = sum/count;
    float sse = 0.;

    // summing up squared errors to the centroid
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        float delta = cover->dm->getY(idx) - centroid;
        sse += delta*delta;
    }    

    return sse;
}

QuantileLossComputer::QuantileLossComputer(int n_quantiles) : n_quantiles(n_quantiles) {
    h = new double[n_quantiles];
    h_low = new int[n_quantiles];
    h_up = new int[n_quantiles];
    y_low = new double[n_quantiles];
    y_pred = new double[n_quantiles];
    under = new double[n_quantiles];
    above = new double[n_quantiles];
}




float* QuantileLossComputer::quantile_tids_errors(RCover* cover) {
    RCover::iterator it;

    int N = cover->getSupport();

    int n_quantiles = cover->dm->getNQuantiles();

    float * errors = new float[n_quantiles];

    float h_tmp;
    int i;
    for (i = 0; i < n_quantiles; i++) {
        h_tmp = (N-1) * cover->dm->getQuantile(i);
        h[i] = h_tmp;
        h_up[i] = ceil(h_tmp);
        h_up[i] = (h_up[i] >= N) ? N-1 : h_up[i];

        h_low[i] = floor(h_tmp);
        
        y_low[i] = -1;
        y_pred[i] = -1;
        under[i] = 0.;
        above[i] = 0.;
    }

    int sub_idx = 0;
    int idx;

    int idx_for_low_val = 0;
    int idx_for_up_val = 0;

    int idx_for_low_sums = 0;
    int idx_for_up_sums = -1;

    double y_cur;


    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        idx = it.value;
        y_cur = cover->dm->getY(idx);
        
        if (idx_for_low_sums < n_quantiles) {
            under[idx_for_low_sums] += y_cur;

            if (sub_idx == h_low[idx_for_low_sums]) {
                idx_for_low_sums += 1;
                idx_for_up_sums += 1;
            }
        }

        if (idx_for_up_sums >= 0) {
            if (!((h_up[idx_for_up_sums] != h_low[idx_for_up_sums]) && (sub_idx == h_up[idx_for_up_sums]))) {
                above[idx_for_up_sums] += y_cur;
            }
        }

        if (idx_for_low_val < n_quantiles) {
            if (sub_idx == h_low[idx_for_low_val]) {
                y_low[idx_for_low_val] = y_cur;

                idx_for_low_val += 1;
            }
        }

        if (idx_for_up_val < n_quantiles) {
            if (sub_idx == h_up[idx_for_up_val]) {
                y_pred[idx_for_up_val] = y_low[idx_for_up_val] + (h[idx_for_up_val] - h_low[idx_for_up_val]) * (y_cur - y_low[idx_for_up_val]);

                idx_for_up_val += 1;
            }
        }

        sub_idx += 1;
    }    

    double sum = 0.;
    for (int i = n_quantiles - 1; i >= 0; i--) {
        above[i] += sum;
        sum = above[i];

        under[i] = (h_low[i] + 1) * y_pred[i] - under[i];
        above[i] = (N - (h_low[i] + 1)) * y_pred[i] - above[i];
        
        float q_i = cover->dm->getQuantile(i); 
        errors[i] = under[i] * q_i + above[i] * (q_i - 1.);
    }

    return errors;
}

