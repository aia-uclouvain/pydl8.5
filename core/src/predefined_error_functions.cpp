#include "predefined_error_functions.h"

double sse_tids_error(RCover* cover) {
    RCover::iterator it;


    // Computing the mean of the targets
    double sum = 0.;
    int count = 0;
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        sum += cover->dm->getY(idx);
        count += 1;
    }

    double centroid = sum/count;
    double sse = 0.;

    // summing up squared errors to the centroid
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        double delta = cover->dm->getY(idx) - centroid;
        sse += delta*delta;
    }    

    return sse;
}

double quantile_tids_error(RCover* cover) {
    RCover::iterator it;

    // Computing the quantile for this cover
    int N = cover->getSupport();
    double q = (double) cover->dm->getQ();

    double h = (N-1)*q;

    int h_up = ceil(h);
    int h_low = floor(h);

    double y_low, y_pred;

    double under = 0.;
    double above = 0.;

    int sub_idx = 0;

    //std::cout << " = [";
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        double y_val = cover->dm->getY(idx);



        if (sub_idx < h_low) {
            under -= y_val;
        } else if (sub_idx == h_low) {
            y_low = y_val;
        } else if (sub_idx == h_up) {
            y_pred = y_low + (h - h_low) * (y_val - y_low);

            under += h_low * y_pred;
            under += y_pred - y_low;
            
            above += y_pred - y_val;
        } else if (sub_idx > h_up) {
            above += y_pred - y_val;          
        }

        //std::cout << y_val << ", ";

        sub_idx += 1;
    }    

    //std::cout << "]" << endl;
    double loss = under * q + above * (q-1.);
    
    return loss;
}

double quantile_tids_error_slow(RCover* cover) {
    RCover::iterator it;

    // Computing the quantile for this cover
    int N = cover->getSupport();
    double q = (double) cover->dm->getQ();

    double h = (N-1)*q;

    int h_up = ceil(h);
    int h_low = floor(h);

    int sub_idx = 0;
    double y_sorted[N];

    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        double y_val = cover->dm->getY(idx);

        y_sorted[sub_idx] = y_val;

        sub_idx += 1;
    }    

    std::sort(y_sorted, y_sorted+N);

    // for (int i = 0; i< N; i++) {
    //     std::cout << y_sorted[i] << ", ";
    // }
    // std::cout << endl;
    


    double y_pred = y_sorted[h_low] + (h - h_low) * (y_sorted[h_up] - y_sorted[h_low]);

    double loss = 0.;
    for (it = cover->begin(true); it.wordIndex < cover->limit.top(); ++it) {
        int idx = it.value;
        double y_val = cover->dm->getY(idx);

        double delta = y_pred - y_val;
        loss += std::fmax(q*delta, (q-1.)*delta);
    }    
    
    return loss;
}