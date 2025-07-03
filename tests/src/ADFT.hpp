#ifndef ADFT_H_
#define ADFT_H_

/**
 * Augmented Dickey-Fuller Test
 *
 * Based on the idea of testing for the presence of a unit root
 * in an autoagressive time series sample.
 *
 * The role of the ADF hypothesis test is to consider the null hypothesis that
 * gamma = 0, which would indicte that the process is a random walk and thus
 * non mean reverting
 *
 * If the hyppothesis that gamma = 0 can be rejected then the following movement
 * of the price series is propotional to the current price and thus it is unlikely
 * to be a random walk.
 */

#include "coreTools.hpp"
#include <cmath>

namespace tests {

    namespace adf {

        void adfuller(xt::xarray<double> x, int maxlag = 0, std::string regression = "c",
                      std::string autolag = "AIC", bool store = false, bool regresults = false) {
            /**
             * x : 1d array of test data
             *
             * maxLag : int, Maximum lag which is included in the test
             *          default value of 12*(nobs/100)^{1/4} is used when 0.
             *
             * regression : {"c", "ct", "ctt", "n"}
             *          constant and trend order to include in regression
             *
             *          * "c" : constant only
             *          * "ct" : constant and trend
             *          * "ctt" : constant, linear, and quadratic trend
             *          * "n" : no constant, no trend
             *
             * autolag : {"AIC", "BIC", "t-stat", ""}
             *          Method to use when automatically determining the lag length among the
             *          values 0, 1, ..., maxlag.
             *
             *          * If "AIC" (default) or "BIC", then the number of lags is chosen
             *            to minimize the corresponding information criterion.
             *          * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
             *            lag until the t-statistic on the last lag length is significant
             *            using a 5%-sized test.
             *          * If None, then the number of included lags is set to maxlag.
             * store : bool
             *         If true then a result instance is returned as well as the ADF stats
             * regresults : bool
             *         If true then return the full regression results
             */

            // initial lines ensure type correctness in python function
            // will ignore for now are correctness is ensured by type definition for params

            // check that data is none constant
            if (xt::amax(x) == xt::amin(x)) {
                std::cout << "Invalid input, x is constant" << std::endl;
                return;
            }

            // store regression results so store must be true
            if (regresults)
                store = true;

            // nobs is a return value regarding the number of observations
            // used for the ADF regression and calculation of the critical values.
            std::size_t nobs = x.shape()[0];

            // ntrend seems to be used in the maxlag calc if
            // maxlag is calculated rather than inputted and it is smaller than
            // using ntrend
            int ntrend = regression.length();

            if (maxlag == 0) {
                // from Greene referencing Schwert 1989
                maxlag = static_cast<int>(ceil(12.0 * pow(static_cast<double>(nobs) / 100.0, 1.0 / 4.0)));
                // -1 for the diff
                // may need floor for first arg
                maxlag = std::min(static_cast<int>(nobs) / 2 - ntrend - 1, maxlag);
                if (maxlag < 0) {
                    std::cout << "Sample size is to short to use the selected regression component" << std::endl;
                    return;
                }
            }else if (maxlag > nobs / 2 - ntrend - 1) {
                std::cout << "maxlag must be less than (nobs / 2 - 1 - ntrend) where ntrend is the number of" <<
                    " included deterministic regressors." << std::endl;
            }

            // get the discrete difference along the given axis
            xt::xarray<double> xdiff = xt::diff(x);
            xt::xarray<double> xdall = tools::lagmat(xdiff, maxlag, "both", "in");

            nobs = xdall.shape()[0];

            auto xdallBlock = xt::view(xdall, xt::all(), 0);
            xdallBlock = xt::view(x, xt::range(-nobs-1, -1));

            xt::xarray<double> xdshort = xt::view(xdiff, xt::range(-nobs, 0));

            xt::xarray<double> fullRHS;
            if (autolag == "AIC" || autolag == "BIC" || autolag == "t-stat") {
                if (regression == "c" || regression == "ct" || regression == "ctt") {
                    // implement add trend function
                    //fullRHS = tools::addTrend(xdall, regression, true);
                }else {
                    fullRHS = xdall;
                }
            }


        }
    }
}

#endif // ADFT_H_
