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

float* quantile_tids_errors(RCover* cover) {
    RCover::iterator it;

    int N = cover->getSupport();
    int n_quantiles = cover->dm->getNQuantiles();

    float * h = new float[n_quantiles];
    int * h_up = new int[n_quantiles];
    int * h_low = new int[n_quantiles];

    double * y_low = new double[n_quantiles];
    double * y_pred = new double[n_quantiles];
    float * errors = new float[n_quantiles];
    
    double * under = new double[n_quantiles];
    double * above = new double[n_quantiles];

    float h_tmp;
    int i;
    for (i = 0; i < n_quantiles; i++) {
        h_tmp = (N-1) * cover->dm->getQuantile(i);
        h[i] = h_tmp;
        h_up[i] = ceil(h_tmp);
        h_low[i] = floor(h_tmp);
        
        under[i] = -1.;
        above[i] = 0.;
    }

    int sub_idx = 0;
    int idx;

    int idx_for_low = 0;
    int idx_for_up = -1;

    double y_cur;

    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        idx = it.value;
        y_cur = cover->dm->getY(idx);


        if (idx_for_low < n_quantiles) {
            under[idx_for_low] += y_cur;

            if (sub_idx == h_low[idx_for_low]) {
                y_low[idx_for_low] = y_cur;

                idx_for_low += 1;
                idx_for_up += 1;
            }
        }

        if (idx_for_up >= 0) {
            if (sub_idx == h_up[idx_for_up]) {                
                y_pred[idx_for_up] = y_low[idx_for_up] + (h[idx_for_up] - h_low[idx_for_up]) * (y_cur - y_low[idx_for_up]);
            }

            above[idx_for_up] += y_cur;
            
        }

        sub_idx += 1;
    }    

    double sum = 0.;
    for (int i = n_quantiles - 1; i >= 0; i--) {
        above[i] += sum;
        sum = above[i];

        under[i] = (h_low[i] + 1) * y_pred[i] - under[i];
        above[i] = (N - h_up[i]) * y_pred[i] - above[i];
        
        float q_i = cover->dm->getQuantile(i); 
        errors[i] = under[i] * q_i + above[i] * (q_i - 1.);
    }

    return errors;
}

// float quantile_tids_error_slow(RCover* cover) {
//     RCover::iterator it;

//     // Computing the quantile for this cover
//     int N = cover->getSupport();
//     double q = (double) cover->dm->getQ();

//     double h = (N-1)*q;

//     int h_up = ceil(h);
//     int h_low = floor(h);

//     int sub_idx = 0;
//     double y_sorted[N];

//     for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
//         int idx = it.value;
//         double y_val = cover->dm->getY(idx);

//         y_sorted[sub_idx] = y_val;

//         sub_idx += 1;
//     }    

//     std::sort(y_sorted, y_sorted+N);

//     // for (int i = 0; i< N; i++) {
//     //     std::cout << y_sorted[i] << ", ";
//     // }
//     // std::cout << endl;
    


//     double y_pred = y_sorted[h_low] + (h - h_low) * (y_sorted[h_up] - y_sorted[h_low]);

//     double loss = 0.;
//     for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
//         int idx = it.value;
//         double y_val = cover->dm->getY(idx);

//         double delta = y_pred - y_val;
//         loss += std::fmax(q*delta, (q-1.)*delta);
//     }    
    
//     return loss;
// }