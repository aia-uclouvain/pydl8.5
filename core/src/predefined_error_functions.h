#ifndef PREDEFINED_ERROR_FUNCTIONS_H
#define PREDEFINED_ERROR_FUNCTIONS_H

#include <math.h>
#include "rCover.h"


float sse_tids_error(RCover* cover);

float* quantile_tids_errors(RCover* cover);

#endif