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
    under = new double[n_quantiles];
    above = new double[n_quantiles];
    y_pred = new double[n_quantiles];
}



Error * QuantileLossComputer::quantile_tids_errors(RCover* cover) {
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

    // Indices going from 0 to N-1 (pseudo indices of the cover)
    int sub_idx = 0;

    // True indices of the full y array
    int idx;

    int idx_for_low_val = 0;
    int idx_for_up_val = 0;

    int idx_for_low_sums = 0;
    int idx_for_up_sums = -1;

    double y_cur;

    for (it = cover->end(true); it.wordIndex >= 0; ++it) {
        idx = it.value;
        y_cur = cover->dm->getY(idx);

        // Add the current element to the sum of the elements above the current quantile
        if (idx_for_up_sums >= 0) {
            // If for this quantile h_up == h_low and the current index is equal to those values, we do not add the current element to the above count as it was already counted in the below count
            if (h_up[idx_for_up_sums] == h_low[idx_for_up_sums]) {
                if (sub_idx > h_up[idx_for_up_sums]) {
                    above[idx_for_up_sums] += y_cur;
                }
            } else if (sub_idx >= h_up[idx_for_up_sums]) {
                above[idx_for_up_sums] += y_cur;
            }
        }

        if (idx_for_low_sums < n_quantiles) {
            under[idx_for_low_sums] += y_cur;

            // If we get to the quantile value we switch to the next one
            if (sub_idx == h_low[idx_for_low_sums]) {
                idx_for_low_sums += 1;
                idx_for_up_sums += 1;

                // If for this quantile h_up == h_low and the current index is equal to those values, we do not add the current element to the above count as it was already counted in the below count
                if (h_up[idx_for_up_sums] == h_low[idx_for_up_sums]) {
                    if (sub_idx > h_up[idx_for_up_sums]) {
                        above[idx_for_up_sums] += y_cur;
                    }
                } else if (sub_idx >= h_up[idx_for_up_sums]) {
                    above[idx_for_up_sums] += y_cur;
                }
                
                // The sum of the elements below the next quantiles is the sum of the below the previous quantile + the elements in between
                if (idx_for_low_sums < n_quantiles)
                    under[idx_for_low_sums] += under[idx_for_up_sums];
            }
        }

        if (idx_for_low_val < n_quantiles) {
            // If the current value is equal to h_low for the quantile, we save it and switch to the next quantile
            if (sub_idx == h_low[idx_for_low_val]) {
                y_low[idx_for_low_val] = y_cur;

                idx_for_low_val += 1;
            }
        }

        if (idx_for_up_val < n_quantiles) {
            // If the current value is equal to h_up for the quantile, we compute y_pred and switch to the next quantile
            if (sub_idx == h_up[idx_for_up_val]) {
                y_pred[idx_for_up_val] = y_low[idx_for_up_val] + (h[idx_for_up_val] - h_low[idx_for_up_val]) * (y_cur - y_low[idx_for_up_val]);
 
                idx_for_up_val += 1;
            }
        }

        sub_idx += 1;
    }    

    double sum = 0.;
    for (int i = n_quantiles - 1; i >= 0; i--) {
        // In reverse order, we must add the sum of the elements above the quantile above itself to the above count as they were not accounted in the previous loop
        above[i] += sum;
        sum = above[i];

        // We compute sum(y_p - y) for the elements above and under the predicted value
        under[i] = (h_low[i] + 1) * y_pred[i] - under[i];
        above[i] = (N - (h_low[i] + 1)) * y_pred[i] - above[i];
        
        // The final error is the sum of the above and below differences multiplied by q and q-1 respectively
        float q_i = cover->dm->getQuantile(i); 
        errors[i] = under[i] * q_i + above[i] * (q_i - 1.);
    }

    return errors;
}

