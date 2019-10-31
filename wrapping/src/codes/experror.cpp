#include "experror.h"
#include <math.h>
#include <stdio.h>

ExpError::ExpError()
{
}


ExpError::~ExpError()
{
}

/* The code below was copied from C4.5 source code */


/*************************************************************************/
/*									 */
/*  Compute the additional errors if the error rate increases to the	 */
/*  upper limit of the confidence level.  The coefficient is the	 */
/*  square of the number of standard deviations corresponding to the	 */
/*  selected confidence level.  (Taken from Documenta Geigy Scientific	 */
/*  Tables (Sixth Edition), p185 (with modifications).)			 */
/*									 */
/*************************************************************************/

float Val[] = {  0,  0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.40, 1.00},
      Dev[] = {4.0,  3.09,  2.58,  2.33, 1.65, 1.28, 0.84, 0.25, 0.00};      
      
ExpError_C45::ExpError_C45 ( float CF ):ExpError (), CF ( CF ) { 
      /*  Compute and retain the coefficient value, interpolating from
  the values in Val and Dev  */

  int i;

  i = 0;
  while ( CF > Val[i] ) i++;

  Coeff = Dev[i-1] +
      (Dev[i] - Dev[i-1]) * (CF - Val[i-1]) /(Val[i] - Val[i-1]);
  Coeff = Coeff * Coeff;
}

Error ExpError_C45::addError ( Support total, Error error, Support datasize )
{
  float Val0, Pr;
  float N = total, e = error;

  if ( e < 1E-6 )
  {
    return N * (1 - exp(log(CF) / N));
  }
  else
    if ( e < 0.9999 )
  {
    Val0 = N * (1 - exp(log(CF) / N));
    return Val0 + e * (addError(total, 1.0,datasize) - Val0);
  }
  else
    if ( e + 0.5 >= N )
  {
    return 0.67 * (N - e);
  }
  else
  {
    Pr = (e + 0.5 + Coeff/2
        + sqrt(Coeff * ((e + 0.5) * (1 - (e + 0.5)/N) + Coeff/4)) )
             / (N + Coeff);
    return (N * Pr - e);
  }
}


double P0[] = {
  -5.99633501014107895267E1,
  9.80010754185999661536E1,
  -5.66762857469070293439E1,
  1.39312609387279679503E1,
  -1.23916583867381258016E0,
};

double Q0[] = {
    /* 1.00000000000000000000E0,*/
  1.95448858338141759834E0,
  4.67627912898881538453E0,
  8.63602421390890590575E1,
  -2.25462687854119370527E2,
  2.00260212380060660359E2,
  -8.20372256168333339912E1,
  1.59056225126211695515E1,
  -1.18331621121330003142E0
};

double P1[] = {
  4.05544892305962419923E0,
  3.15251094599893866154E1,
  5.71628192246421288162E1,
  4.40805073893200834700E1,
  1.46849561928858024014E1,
  2.18663306850790267539E0,
  -1.40256079171354495875E-1,
  -3.50424626827848203418E-2,
  -8.57456785154685413611E-4,
};


double Q1[] = {
  /*  1.00000000000000000000E0,*/
  1.57799883256466749731E1,
  4.53907635128879210584E1,
  4.13172038254672030440E1,
  1.50425385692907503408E1,
  2.50464946208309415979E0,
  -1.42182922854787788574E-1,
  -3.80806407691578277194E-2,
  -9.33259480895457427372E-4
};

double  P2[] = {
  3.23774891776946035970E0,
  6.91522889068984211695E0,
  3.93881025292474443415E0,
  1.33303460815807542389E0,
  2.01485389549179081538E-1,
  1.23716634817820021358E-2,
  3.01581553508235416007E-4,
  2.65806974686737550832E-6,
  6.23974539184983293730E-9,
};

double  Q2[] = {
  /*  1.00000000000000000000E0,*/
  6.02427039364742014255E0,
  3.67983563856160859403E0,
  1.37702099489081330271E0,
  2.16236993594496635890E-1,
  1.34204006088543189037E-2,
  3.28014464682127739104E-4,
  2.89247864745380683936E-6,
  6.79019408009981274425E-9
};


  
ExpError_J48::ExpError_J48 ( float CF ):ExpError (), CF ( CF ) { 
      /*  Compute and retain the coefficient value, interpolating from
  the values in Val and Dev  */
  z = normalInverse(1.0 - CF);
}

float ExpError_J48::normalInverse(double y0) { 

  double x, y, z, y2, x0, x1;
  int code;

  double s2pi = sqrt(2.0*M_PI);

  code = 1;
  y = y0;
  if( y > (1.0 - 0.13533528323661269189) ) { /* 0.135... = exp(-2) */
    y = 1.0 - y;
    code = 0;
  }

  if( y > 0.13533528323661269189 ) {
    y = y - 0.5;
    y2 = y * y;
    x = y + y * (y2 * polevl( y2, P0, 4)/p1evl( y2, Q0, 8 ));
    x = x * s2pi; 
    return(x);
  }

  x = sqrt( -2.0 * log(y) );
  x0 = x - log(x)/x;

  z = 1.0/x;
  if( x < 8.0 ) /* y > exp(-32) = 1.2664165549e-14 */
    x1 = z * polevl( z, P1, 8 )/p1evl( z, Q1, 8 );
  else
    x1 = z * polevl( z, P2, 8 )/p1evl( z, Q2, 8 );
  x = x0 - x1;
  if( code != 0 )
    x = -x;
  return( x );
}

double ExpError_J48::p1evl( double x, double coef[], int N ) {

  double ans;
  ans = x + coef[0];

  for(int i=1; i<N; i++) ans = ans*x+coef[i];

  return ans;
}

double ExpError_J48::polevl( double x, double coef[], int N ) {

  double ans;
  ans = coef[0];

  for(int i=1; i<=N; i++) ans = ans*x+coef[i];

  return ans;
}

Error ExpError_J48::addError( Support N, Error  e, Support datasize)
{
  
     // Check for extreme cases at the low end because the
    // normal approximation won't work
  if (e < 1) {
    // no error in training data...
      // Base case (i.e. e == 0) from documenta Geigy Scientific
      // Tables, 6th edition, page 185
    double base = N * (1 - exp(log(CF)* 1.0 / (double) N));
    if (e == 0) {
      return base;
    }

      // Use linear interpolation between 0 and 1 like C4.5 does
    return base + e * (addError(N, 1, datasize) - base);
  }

    // Use linear interpolation at the high end (i.e. between N - 0.5
    // and N) because of the continuity correction
  if (e + 0.5 >= (double) N) {

      // Make sure that we never return anything smaller than zero
    if ( N - e < 0 )
      return 0;
    return N - e;
  }
  // Get z-score corresponding to CF
  
  
    // Compute upper limit of confidence interval
  double  f = (( (double) e ) + 0.5) / (double) N;
  double r = (f + (z * z) / (double) (2.0 * N) +
        z * sqrt((f / (double) N) -
        (f * f / (double) N) +
        (z * z / (double) (4.0 * N * N)))) /
        (1.0 + (z * z) / (double) N);

  return (r * N) - e;

}