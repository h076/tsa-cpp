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

#include "RegressionModel.hpp"
#include "coreTools.hpp"
#include "modelHelpers.hpp"
#include "MacKinnonValues.hpp"
#include <cmath>
#include <stdexcept>

#include <xtensor/containers/xadapt.hpp>

namespace tests {

    namespace adf {

        struct ADFResult {
            double adfstat;
            double pvalue;
            int usedlag;
            std::size_t nobs;
            std::map<std::string, double> critvalues;
            double icbest;
        };

        ADFResult adfuller(xt::xarray<double> x, int maxlag = 0, std::string regression = "c",
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
                throw std::invalid_argument("Invalid input, x is constant");
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
            int ntrend;
            if (regression == "n") ntrend = 0;
            else if (regression == "c") ntrend = 1;
            else if (regression == "ct") ntrend = 2;
            else if (regression == "ctt") ntrend = 3;
            else throw std::invalid_argument("Invalid regression type");

            if (maxlag == 0) {
                // from Greene referencing Schwert 1989
                maxlag = static_cast<int>(ceil(12.0 * pow(static_cast<double>(nobs) / 100.0, 1.0 / 4.0)));
                // -1 for the diff
                // may need floor for first arg
                maxlag = std::min(static_cast<int>(nobs) / 2 - ntrend - 1, maxlag);
                if (maxlag < 0) {
                    throw std::runtime_error("Sample size is to short to use the selected regression component");
                }
            }else if (maxlag > nobs / 2 - ntrend - 1) {
                std::cout << "maxlag must be less than (nobs / 2 - 1 - ntrend) where ntrend is the number of" <<
                    " included deterministic regressors." << std::endl;
            }

            //std::cout << "Checking maxlag : " << maxlag << std::endl;

            // get the discrete difference along the given axis
            xt::xarray<double> xdiff = xt::diff(x);
            //std::cout << "Checking xdiff : " << xt::adapt(xdiff.shape()) << std::endl;
            //std::cout << xdiff << std::endl;

            xt::xarray<double> xdall = tools::lagmat(xdiff, maxlag, "both", "in");
            //std::cout << "Checking xdall : " << xt::adapt(xdall.shape()) << std::endl;
            //std::cout << xdall << std::endl;

            nobs = xdall.shape()[0];

            xt::view(xdall, xt::all(), 0) = xt::view(x, xt::range(x.shape()[0] - nobs - 1, x.shape()[0] - 1));
            //std::cout << "Checking xdall v2 : " << xt::adapt(xdall.shape()) << std::endl;
            //std::cout << xdall << std::endl;

            xt::xarray<double> xdshort = xt::view(xdiff, xt::range(-nobs, xt::all()));
            //std::cout << "Checking xdshort : " << xt::adapt(xdshort.shape()) << std::endl;
            //std::cout << xdshort << std::endl;

            xt::xarray<double> fullRHS;
            int usedlag;
            double icbest;
            if (autolag == "AIC" || autolag == "BIC" || autolag == "t-stat") {

                if (regression == "c" || regression == "ct" || regression == "ctt") {
                    fullRHS = tools::addTrend(xdall, regression, true);
                }else {
                    fullRHS = xdall;
                }

                //std::cout << "Checking fullRHS : " << xt::adapt(fullRHS.shape()) << std::endl;
                //std::cout << fullRHS << std::endl;

                int startLag = fullRHS.shape(1) - xdall.shape(1) + 1;

                //std::cout << "Checking startLag : " << startLag << std::endl;

                tools::autoLagResult autoRes = tools::autoLag(linModels::OLS, fullRHS, xdshort,
                                                              startLag, maxlag, autolag);

                //std::cout << "Checking autolag results ..." << std::endl;
                //std::cout << "icbest : " << autoRes.icbest << std::endl;
                //std::cout << "bestlag : " << autoRes.bestLag << std::endl;

                icbest = autoRes.icbest;
                int bestlag = autoRes.bestLag;

                bestlag -= startLag;

                // rerun OLS with best autolag
                xdall = tools::lagmat(xdiff, bestlag, "both", "in");

                //std::cout << "Checking xdall v3 : " << xt::adapt(xdall.shape()) << std::endl;
                //std::cout << xdall << std::endl;

                nobs = xdall.shape(0);
                auto x_len = x.shape()[0];
                if (x_len < nobs + 1) {
                    throw std::runtime_error("Not enough observations after lagging");
                }
                xt::view(xdall, xt::all(), 0) = xt::view(x, xt::range(x_len - nobs - 1, x_len - 1));
                xdshort = xt::view(xdiff, xt::range(static_cast<int>(-nobs), xt::all()));

                //std::cout << "Checking xdshort v2 : " << xt::adapt(xdshort.shape()) << std::endl;
                //std::cout << xdshort << std::endl;

                usedlag = bestlag;
            } else {
                usedlag = maxlag;
                icbest = -1.0;
            }

            linModels::RegressionResult resols;
            if (regression != "n") {
                auto rhs = tools::addTrend(xt::view(xdall, xt::all(), xt::range(0, usedlag + 1)),
                                           regression, false);
                resols = linModels::getModelOfType(linModels::OLS, rhs, xdshort)->fit();
            } else {
                auto rhs = xt::view(xdall, xt::all(), xt::range(0, usedlag + 1));
                resols = linModels::getModelOfType(linModels::OLS, rhs, xdshort)->fit();
            }

            double adfstat = resols.tValues[0];

            //std::cout << "Checking ADF stat : " << adfstat << std::endl;

            double pvalue = tools::mackinnon::p_value(adfstat, regression, 1);

            //std::cout << "checking P value : " << pvalue << std::endl;

            xt::xarray<double> critvalues = tools::mackinnon::crit_value(1, regression, nobs);

            std::map<std::string, double> crits;
            crits["1%"] = critvalues[0];
            crits["5%"] = critvalues[1];
            crits["10%"] = critvalues[2];

            return {adfstat, pvalue, usedlag, nobs, crits, icbest};
        }
    }
}

#endif // ADFT_H_
