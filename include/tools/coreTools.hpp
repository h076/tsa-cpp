#ifndef TOOLS_H_
#define TOOLS_H_

#include <stdexcept>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/io/xio.hpp>

#include "../models/linear/RegressionModel.hpp"
#include "../models/linear/modelHelpers.hpp"

namespace tools {

    // lagmat will return a 2d tensor of lags
    xt::xtensor<double, 2> lagmat(xt::xtensor<double, 2>& x, int maxlag, std::string trim, std::string original) {
        /*
         * x : xarray<double>, input at most 2D
         *
         * maxlag : int, all lags from zero to maxlag are included
         *
         * trim : std::string, the trimming method to use
         *     * 'forward' : trim invalid observations in front.
         *     * 'backward' : trim invalid initial observations.
         *     * 'both' : trim invalid observations on both sides.
         *     * 'none' : no trimming of observations.
         *
         * original : std::string, how the original is treated
         *     * 'ex' : drops the original array returning only the lagged values.
         *     * 'in' : returns the original array and the lagged values as a single
         *              array.
         */

        // should check strings ...

        // Compute shape of tensor
        std::size_t nobs = x.shape(0);
        std::size_t nvar = x.shape(1);

        // validate maxlag
        if (maxlag < 0)
            throw std::invalid_argument("lagmat: maxlag must be non-negative");

        if (static_cast<std::size_t>(maxlag) >= nobs)
            throw std::invalid_argument("lagmat: maxlag must be < nobs");

        // validate trim
        if (trim != "forward" && trim != "backward" && trim != "both"    && trim != "none")
            throw std::invalid_argument("lagmat: trim must be 'forward', 'backward', 'both', or 'none'");

        // Validate original option
        if (original != "ex" && original != "in")
            throw std::invalid_argument("lagmat: original must be 'ex' or 'in'");

        std::size_t dropidx = (original == "ex") ? nvar : 0;

        // fill new zeros xtensor
        xt::xtensor<double, 2> lm = xt::zeros<double>({nobs + static_cast<std::size_t>(maxlag),
                nvar * (static_cast<std::size_t>(maxlag) + 1)});

        // equivelant of :
        // for k in range(0, (maxlag + 1)):
        //     lm [maxlag - k : nobs+maxlag-k, nvar*(maxlag-k) : nvar*(maxlag-k+1)] = x
        for(int k=0; k<maxlag+1; ++k) {
            std::size_t r0 = static_cast<std::size_t>(maxlag) - k;
            std::size_t r1 = nobs + r0;
            std::size_t c0 = (static_cast<std::size_t>(maxlag) - k) * nvar;
            std::size_t c1 = nvar + c0;

            // take block view from zeros array
            auto block = xt::view(lm, xt::range(r0, r1), xt::range(c0, c1));
            block = x;
        }

        std::size_t startobs = (trim == "none" || trim == "forward") ? 0 : static_cast<std::size_t>(maxlag);
        std::size_t stopobs = (trim == "none" || trim == "backward") ? lm.shape()[0] : nobs;

        xt::xtensor<double, 2> lags = xt::view(lm, xt::range(startobs, stopobs), xt::range(dropidx, lm.shape()[1]));
        return lags;
    }

    // Handle 1D tensors for lagmat
    xt::xtensor<double, 2> lagmat(xt::xtensor<double, 1>& x, int maxlag, std::string trim, std::string original) {
        xt::xtensor<double, 2> x2 = xt::expand_dims(x, 1);
        return lagmat(x2, maxlag, trim, original);
    }

    // Prepends/appends columns for constant and/or (linear, quadratic) trend to the design matrix.
    // For example a 2D array of (n, p) will be reshaped to (n, p + k), k being the number of trend values
    xt::xtensor<double, 2> addTrend(const xt::xtensor<double, 2>& x, std::string trend, bool prepend) {
        /*
         *
         * X : 2D array
         *
         * trend : string
         *     * "c" add constant only
         *     * "t" add trend only
         *     * "ct" add constant and linear trend
         *     * "ctt" add constant, linear trend and quadratic trend
         *
         * prepend : bool
         *     If true, prepends the new data to the columns of X.
         *
         */

        // turn trend to lower case
        std::transform(trend.begin(), trend.end(), trend.begin(), ::tolower);

        // check trend is valid
        if (trend != "c" && trend != "ct" && trend != "ctt" && trend != "cttt")
            throw std::invalid_argument("tools::addTrend : Trend " + trend + " is invalid.");

        // Compute num of objects
        std::size_t nobs = x.shape(0);

        // build sub trends 1, 2, ..., nobs
        xt::xtensor<double, 1> constant = xt::arange<double>(1.0, double(nobs) + 1.0);
        xt::xtensor<double, 2> lin = xt::expand_dims(constant, 1);
        xt::xtensor<double, 2> quad = xt::expand_dims(xt::eval(xt::pow(constant, 2)), 1);

        // build trend array of shape
        xt::xtensor<double, 2> trendarr;
        if (trend == "c") {
            // constant only
            std::vector<std::size_t> tdrs = {nobs, 1};
            trendarr = xt::ones<double>(tdrs);
        } else if (trend == "t") {
            // trend only
            trendarr = lin;
        } else if (trend == "ct") {
            // constant + linear
            std::vector<std::size_t> tdrs = {nobs, 1};
            trendarr = xt::concatenate(xt::xtuple(xt::ones<double>(tdrs), lin), 1);
        } else {
            // constant + linear + quadratic
            std::vector<std::size_t> tdrs = {nobs, 1};
            trendarr = xt::concatenate(xt::xtuple(xt::ones<double>(tdrs), lin, quad), 1);
        }

        // concat along axis=1
        xt::xtensor<double, 2> result;
        if (prepend) {
            result = xt::concatenate(xt::xtuple(trendarr, x), 1);
        } else {
            result = xt::concatenate(xt::xtuple(x, trendarr), 1);
        }

        return result;
    }

    // Handle tensor of 1D as input for addTrend
    xt::xtensor<double, 2> addTrend(const xt::xtensor<double, 1>& x, std::string trend, bool prepend) {
        xt::xtensor<double, 2> x2 = xt::expand_dims(x, 1);
        return addTrend(x2, trend, prepend);
    }

    struct autoLagResult {
        double icbest;
        int bestLag;
    };

    // Returns the result for the lag length that maximises info criterion
    autoLagResult autoLag(linModels::modelType mod, const xt::xtensor<double, 2>& X, const xt::xtensor<double, 1>& y,
                            int startLag, int maxLag, std::string method) {

        /*
         * mod : linModels::modelType
         *     - Model class type
         *
         * X : xarray
         *     - Input nobs by (startlag + maxlag) array containing lags and possibly other variables
         *
         * y : xarray
         *     - nobs array containing y variable
         *
         * startLag : int
         *     - The first zero-indexed column to hold a lag
         *
         * maxLag : int
         *     - The highest lag order for lag length selection
         *
         * method : string {"aic", "bic", "t-stat"}
         *     - aic : Akaike Information Criterion
         *     - bic : Bayes Information Criterion
         *     - t-stat : Based on last lag
         */

        /*
         * Returns ...
         *
         * icbest : float
         *     - Best information criteria
         *
         * bestlag : int
         *     - Lag length that maximises information crtiterion
         */

        // dictionary storing key-value (lag, results) pairs
        std::unordered_map<int, linModels::RegressionResult> results;
        std::transform(method.begin(), method.end(), method.begin(), ::tolower);

        // Loop over lags from startLag to startLag + maxLag (inclusive)
        for(int lag = startLag; lag < startLag + maxLag + 1; lag++) {
            std::unique_ptr<linModels::RegressionModel> modInstance = linModels::getModelOfType(mod, xt::view(X, xt::all(), xt::range(0, lag)), y);
            results[lag] = modInstance->fit(); // Store model result
        }

        double icbest; // Best information criterion
        int bestLag; // Corresponding lag

        // Select lag with lowest AIC
        if (method == "aic") {
            auto best = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) {
                return a.second.aic < b.second.aic;
            });

            icbest = best->second.aic;
            bestLag = best->second.lag;
        }
        // Select lag with lowest BIC
        else if (method == "bic") {
            auto best = std::min_element(results.begin(), results.end(), [](const auto& a, const auto& b) {
                return a.second.bic < b.second.bic;
            });

            icbest = best->second.bic;
            bestLag = best->first;
        }
        // Select highest lag where last t-stat is statistically significant
        else if (method == "t-stat") {
            double stop = 1.6448536269514722; // 95% critical value

            bestLag = startLag + maxLag;
            icbest = 0.0;

            // Iterate backwards from largest to smallest lag
            for(int lag = startLag + maxLag; lag > startLag - 1; lag--) {
                icbest = std::abs(results[lag].tValues.back());
                bestLag = lag;
                if (std::abs(icbest) >= stop)
                    break; // break for first lag with significant t-stat
            }
        } else {
            throw std::invalid_argument("tools::autoLag : Invalid method.");
        }

        return {icbest, bestLag};
    }
}

#endif // TOOLS_H_
