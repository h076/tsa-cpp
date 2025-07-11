#ifndef MACKINNONCRITVALUES_H_
#define MACKINNONCRITVALUES_H_

#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <cmath>
#include <limits>

namespace tools {

    namespace mackinnon {

        using xt::xarray;

        const double inf = std::numeric_limits<double>::infinity();

        const xarray<double> tau_star_nc = {-1.04, -1.53, -2.68, -3.09, -3.07, -3.77};
        const xarray<double> tau_min_nc  = {-19.04,-19.62,-21.21,-23.25,-21.63,-25.74};
        const xarray<double> tau_max_nc  = {inf,1.51,0.86,0.88,1.05,1.24};

        const xarray<double> tau_star_c  = {-1.61, -2.62, -3.13, -3.47, -3.78, -3.93};
        const xarray<double> tau_min_c   = {-18.83,-18.86,-23.48,-28.07,-25.96,-23.27};
        const xarray<double> tau_max_c   = {2.74,0.92,0.55,0.61,0.79,1.0};

        const xarray<double> tau_star_ct  = {-2.89, -3.19, -3.50, -3.65, -3.80, -4.36};
        const xarray<double> tau_min_ct   = {-16.18,-21.15,-25.37,-26.63,-26.53,-26.18};
        const xarray<double> tau_max_ct   = {0.7,0.63,0.71,0.93,1.19,1.42};

        const xarray<double> tau_star_ctt = {-3.21,-3.51,-3.81,-3.83,-4.12,-4.63};
        const xarray<double> tau_min_ctt  = {-17.17,-21.1,-24.33,-24.03,-24.33,-28.22};
        const xarray<double> tau_max_ctt  = {0.54,0.79,1.08,1.43,3.49,1.92};

        const xarray<double> small_scaling = {1.0, 1.0, 1e-2};
        const xarray<double> large_scaling = {1.0, 1e-1, 1e-1, 1e-2};
        const xarray<double> z_large_scaling = {1.0, 1e-1, 1e-2, 1e-3, 1e-5};

        // Tau NC small-p coefficients
        const xarray<double> tau_nc_smallp = xt::view(
            xarray<double>{
                {0.6344,1.2378,3.2496},
                {1.9129,1.3857,3.5322},
                {2.7648,1.4502,3.4186},
                {3.4336,1.4835,3.19},
                {4.0999,1.5533,3.59},
                {4.5388,1.5344,2.9807}
            }, xt::all()) * small_scaling;

        // Tau C small-p coefficients
        const xarray<double> tau_c_smallp = xt::view(
            xarray<double>{
                {2.1659,1.4412,3.8269},
                {2.92,1.5012,3.9796},
                {3.4699,1.4856,3.164},
                {3.9673,1.4777,2.6315},
                {4.5509,1.5338,2.9545},
                {5.1399,1.6036,3.4445}
            }, xt::all()) * small_scaling;

        // Tau CT small-p coefficients
        const xarray<double> tau_ct_smallp = xt::view(
            xarray<double>{
                {3.2512,1.6047,4.9588},
                {3.6646,1.5419,3.6448},
                {4.0983,1.5173,2.9898},
                {4.5844,1.5338,2.8796},
                {5.0722,1.5634,2.9472},
                {5.53,1.5914,3.0392}
            }, xt::all()) * small_scaling;

        // Tau CTT small-p coefficients
        const xarray<double> tau_ctt_smallp = xt::view(
            xarray<double>{
                {4.0003,1.658,4.8288},
                {4.3534,1.6016,3.7947},
                {4.7343,1.5768,3.2396},
                {5.214,1.6077,3.3449},
                {5.6481,1.6274,3.3455},
                {5.9296,1.5929,2.8223}
            }, xt::all()) * small_scaling;

        // Tau NC large-p coefficients
        const xarray<double> tau_nc_largep = xt::view(
            xarray<double>{
                {0.4797, 9.3557, -0.6999, 3.3066},
                {1.5578, 8.5580, -2.0830, -3.3549},
                {2.2268, 6.8093, -3.2362, -5.4448},
                {2.7654, 6.4502, -3.0811, -4.4946},
                {3.2684, 6.8051, -2.6778, -3.4972},
                {3.7268, 7.1670, -2.3648, -2.8288}
            }, xt::all()) * large_scaling;

        // Tau C large-p coefficients
        const xarray<double> tau_c_largep = xt::view(
            xarray<double>{
                {1.7339, 9.3202, -1.2745, -1.0368},
                {2.1945, 6.4695, -2.9198, -4.2377},
                {2.5893, 4.5168, -3.6529, -5.0074},
                {3.0387, 4.5452, -3.3666, -4.1921},
                {3.5049, 5.2098, -2.9158, -3.3468},
                {3.9489, 5.8933, -2.5359, -2.7210}
            }, xt::all()) * large_scaling;

        // Tau CT large-p coefficients
        const xarray<double> tau_ct_largep = xt::view(
            xarray<double>{
                {2.5261, 6.1654, -3.7956, -6.0285},
                {2.8500, 5.2720, -3.6622, -5.1695},
                {3.2210, 5.2550, -3.2685, -4.1501},
                {3.6520, 5.9758, -2.7483, -3.2081},
                {4.0712, 6.6428, -2.3464, -2.5460},
                {4.4735, 7.1757, -2.0681, -2.1196}
            }, xt::all()) * large_scaling;

        // Tau CTT large-p coefficients
        const xarray<double> tau_ctt_largep = xt::view(
            xarray<double>{
                {3.0778, 4.9529, -4.1477, -5.9359},
                {3.4713, 5.9670, -3.2507, -4.2286},
                {3.8637, 6.7852, -2.6286, -3.1381},
                {4.2736, 7.6199, -2.1534, -2.4026},
                {4.6679, 8.2618, -1.8220, -1.9147},
                {5.0009, 8.3735, -1.6994, -1.6928}
        }, xt::all()) * large_scaling;

        // Star z-stats
        const xarray<double> z_star_nc  = {-2.9, -8.7, -14.8, -20.9, -25.7, -30.5};
        const xarray<double> z_star_c   = {-8.9, -14.3, -19.5, -25.1, -29.6, -34.4};
        const xarray<double> z_star_ct  = {-15.0, -19.6, -25.3, -29.6, -31.8, -38.4};
        const xarray<double> z_star_ctt = {-20.7, -25.3, -29.9, -34.4, -38.5, -44.2};

        const xarray<double> z_nc_smallp = {
            {0.0342, -0.6376, 0.0, -0.03872},
            {1.3426, -0.7680, 0.0, -0.04104},
            {3.8607, -2.4159, 0.51293, -0.09835},
            {6.1072, -3.7250, 0.85887, -0.13102},
            {7.7800, -4.4579, 1.00056, -0.14014},
            {4.0253, -0.8815, 0.0, -0.04887}
        };

        const xarray<double> z_c_smallp = {
            {2.2142, -1.7863, 0.32828, -0.07727},
            {1.1662, 0.1814, -0.36707, 0.0},
            {6.6584, -4.3486, 1.04705, -0.15011},
            {3.3249, -0.8456, 0.0, -0.04818},
            {4.0356, -0.9306, 0.0, -0.04776},
            {13.9959, -8.4314, 1.97411, -0.22234}
        };

        const xarray<double> z_ct_smallp = {
            {4.6476, -2.8932, 0.5832, -0.0999},
            {7.2453, -4.7021, 1.1270, -0.15665},
            {3.4893, -0.8914, 0.0, -0.04755},
            {1.6604, 1.0375, -0.53377, 0.0},
            {2.0060, 1.1197, -0.55315, 0.0},
            {11.1626, -5.6858, 1.21479, -0.15428}
        };

        const xarray<double> z_ctt_smallp = {
            {3.6739, -1.1549, 0.0, -0.03947},
            {3.9783, -1.0619, 0.0, -0.04394},
            {2.0062, 0.8907, -0.51708, 0.0},
            {4.9218, -1.0663, 0.0, -0.04691},
            {5.1433, -0.9877, 0.0, -0.04993},
            {23.6812, -14.6485, 3.42909, -0.33794}
        };

        const xarray<double> z_nc_largep = xt::view(
            xarray<double>{
                {0.4927, 6.9060, 13.2331, 12.0990, 0.0},
                {1.5167, 4.6859, 4.2401, 2.7939, 7.9601},
                {2.2347, 3.9465, 2.2406, 0.8746, 1.4239},
                {2.8239, 3.6265, 1.6738, 0.5408, 0.7449},
                {3.3174, 3.3492, 1.2792, 0.3416, 0.3894},
                {3.7290, 3.0611, 0.9579, 0.2087, 0.1943}
            }, xt::all()) * z_large_scaling;

        const xarray<double> z_c_largep = xt::view(
            xarray<double>{
                {1.7170, 5.5243, 4.3463, 1.6671, 0.0},
                {2.2394, 4.2377, 2.4320, 0.9241, 0.4364},
                {2.7430, 3.6260, 1.5703, 0.4612, 0.5670},
                {3.2280, 3.3399, 1.2319, 0.3162, 0.3482},
                {3.6583, 3.0934, 0.9681, 0.2111, 0.1979},
                {4.0379, 2.8735, 0.7694, 0.1433, 0.1146}
            }, xt::all()) * z_large_scaling;

        const xarray<double> z_ct_largep = xt::view(
            xarray<double>{
                {2.7117, 4.5731, 2.2868, 0.6362, 0.5},
                {3.0972, 4.0873, 1.8982, 0.5796, 0.7384},
                {3.4594, 3.6326, 1.4284, 0.3813, 0.4325},
                {3.8060, 3.2634, 1.0689, 0.2402, 0.2304},
                {4.1402, 2.9867, 0.8323, 0.1600, 0.1315},
                {4.4497, 2.7534, 0.6582, 0.1089, 0.0773}
            }, xt::all()) * z_large_scaling;

        const xarray<double> z_ctt_largep = xt::view(
            xarray<double>{
                {3.4671, 4.3476, 1.9231, 0.5381, 0.6216},
                {3.7827, 3.9421, 1.5699, 0.4093, 0.4485},
                {4.0520, 3.4947, 1.1772, 0.2642, 0.2502},
                {4.3311, 3.1625, 0.9126, 0.1775, 0.1462},
                {4.5940, 2.8739, 0.7070, 0.1181, 0.0838},
                {4.8479, 2.6447, 0.5647, 0.0827, 0.0518}
        }, xt::all()) * z_large_scaling;

        // Polynomial evaluation: same as numpy.polyval
        inline double polyval_scalar(const std::vector<double>& coeffs, double x) {
            double result = 0.0;
            for (auto c : coeffs)
                result = result * x + c;

            return result;
        }

        // Standard normal CDF using error function
        inline double norm_cdf(double x, double mu = 0.0, double sigma = 1.0) {
           return 0.5 * (1.0 + std::erf((x - mu) / (sigma * std::sqrt(2.0))));
        }

        inline double p_value(double teststat, const std::string& regression = "c", int N = 1) {

            // Maps from regression strings to tau arrays
            static const std::unordered_map<std::string, const xarray<double>*> tau_max_map = {
                {"nc", &tau_max_nc}, {"c", &tau_max_c}, {"ct", &tau_max_ct}, {"ctt", &tau_max_ctt}
            };

            static const std::unordered_map<std::string, const xarray<double>*> tau_min_map = {
                {"nc", &tau_min_nc}, {"c", &tau_min_c}, {"ct", &tau_min_ct}, {"ctt", &tau_min_ctt}
            };

            static const std::unordered_map<std::string, const xarray<double>*> tau_star_map = {
                {"nc", &tau_star_nc}, {"c", &tau_star_c}, {"ct", &tau_star_ct}, {"ctt", &tau_star_ctt}
            };

            static const std::unordered_map<std::string, const xarray<double>*> tau_smallp_map = {
                {"nc", &tau_nc_smallp}, {"c", &tau_c_smallp}, {"ct", &tau_ct_smallp}, {"ctt", &tau_ctt_smallp}
            };

            static const std::unordered_map<std::string, const xarray<double>*> tau_largep_map = {
                {"nc", &tau_nc_largep}, {"c", &tau_c_largep}, {"ct", &tau_ct_largep}, {"ctt", &tau_ctt_largep}
            };

            const xarray<double> * maxstat = tau_max_map.at(regression);
            const xarray<double> * minstat = tau_min_map.at(regression);
            const xarray<double> * starstat = tau_star_map.at(regression);

            if (teststat > (*maxstat)[N-1]) {
                return 1.0;
            } else if (teststat < (*minstat)[N-1]) {
                return 0.0;
            }

            const xarray<double> * tau_coef;
            if (teststat <= (*starstat)[N-1]) {
                tau_coef = tau_smallp_map.at(regression);
            } else {
                tau_coef = tau_largep_map.at(regression);
            }

            auto coef_view = xt::view(*tau_coef, xt::all());
            auto coef_arr = xt::xarray<double>(coef_view);
            auto flipped = xt::eval(xt::flip(coef_arr));
            std::vector<double> coeffs(flipped.begin(), flipped.end());
            return norm_cdf(polyval_scalar(coeffs, teststat));
        }


        const xt::xarray<double> tau_nc_2010 = {{{-2.56574, -2.2358, -3.627, 0},
                                         {-1.94100, -0.2686, -3.365, 31.223},
                                         {-1.61682, 0.2656, -2.714, 25.364}}};

        const xt::xarray<double> tau_c_2010 = {
        {{-3.43035, -6.5393, -16.786, -79.433}, {-2.86154, -2.8903, -4.234, -40.040}, {-2.56677, -1.5384, -2.809, 0}},
        {{-3.89644, -10.9519, -33.527, 0},      {-3.33613, -6.1101, -6.823, 0},      {-3.04445, -4.2412, -2.720, 0}},
        {{-4.29374, -14.4354, -33.195, 47.433}, {-3.74066, -8.5632, -10.852, 27.982},{-3.45218, -6.2143, -3.718, 0}},
        {{-4.64332, -18.1031, -37.972, 0},      {-4.09600, -11.2349, -11.175, 0},    {-3.81020, -8.3931, -4.137, 0}},
        {{-4.95756, -21.8883, -45.142, 0},      {-4.41519, -14.0405, -12.575, 0},    {-4.13157, -10.7417, -3.784, 0}},
        {{-5.24568, -25.6688, -57.737, 88.639}, {-4.70693, -16.9178, -17.492, 60.007},{-4.42501, -13.1875, -5.104, 27.877}},
        {{-5.51233, -29.5760, -69.398, 164.295},{-4.97684, -19.9021, -22.045, 110.761},{-4.69648, -15.7315, -5.104, 27.877}},
        {{-5.76202, -33.5258, -82.189, 256.289},{-5.22924, -23.0023, -24.646, 144.479},{-4.95007, -18.3959, -7.344, 94.872}},
        {{-5.99742, -37.6572, -87.365, 248.316},{-5.46697, -26.2057, -26.627, 176.382},{-5.18897, -21.1377, -9.484, 172.704}},
        {{-6.22103, -41.7154, -102.680, 389.33},{-5.69244, -29.4521, -30.994, 251.016},{-5.41533, -24.0006, -7.514, 163.049}},
        {{-6.43377, -46.0084, -106.809, 352.752},{-5.90714, -32.8336, -30.275, 249.994},{-5.63086, -26.9693, -4.083, 151.427}},
        {{-6.63790, -50.2095, -124.156, 579.622},{-6.11279, -36.2681, -32.505, 314.802},{-5.83724, -29.9864, -2.686, 184.116}}};

        const xt::xarray<double> tau_ct_2010 = {
        {{-3.95877, -9.0531, -28.428, -134.155}, {-3.41049, -4.3904, -9.036, -45.374}, {-3.12705, -2.5856, -3.925, -22.380}},
        {{-4.32762, -15.4387, -35.679, 0}, {-3.78057, -9.5106, -12.074, 0}, {-3.49631, -7.0815, -7.538, 21.892}},
        {{-4.66305, -18.7688, -49.793, 104.244}, {-4.11890, -11.8922, -19.031, 77.332}, {-3.83511, -9.0723, -8.504, 35.403}},
        {{-4.96940, -22.4694, -52.599, 51.314}, {-4.42871, -14.5876, -18.228, 39.647}, {-4.14633, -11.2500, -9.873, 54.109}},
        {{-5.25276, -26.2183, -59.631, 50.646}, {-4.71537, -17.3569, -22.660, 91.359}, {-4.43422, -13.6078, -10.238, 76.781}},
        {{-5.51727, -29.9760, -75.222, 202.253}, {-4.98228, -20.3050, -25.224, 132.03}, {-4.70233, -16.1253, -9.836, 94.272}},
        {{-5.76537, -33.9165, -84.312, 245.394}, {-5.23299, -23.3328, -28.955, 182.342}, {-4.95405, -18.7352, -10.168, 120.575}},
        {{-6.00003, -37.8892, -96.428, 335.92}, {-5.46971, -26.4771, -31.034, 220.165}, {-5.19183, -21.4328, -10.726, 157.955}},
        {{-6.22288, -41.9496, -109.881, 466.068}, {-5.69447, -29.7152, -33.784, 273.002}, {-5.41738, -24.2882, -8.584, 169.891}},
        {{-6.43551, -46.1151, -120.814, 566.823}, {-5.90887, -33.0251, -37.208, 346.189}, {-5.63255, -27.2042, -6.792, 177.666}},
        {{-6.63894, -50.4287, -128.997, 642.781}, {-6.11404, -36.4610, -36.246, 348.554}, {-5.83850, -30.1995, -5.163, 210.338}},
        {{-6.83488, -54.7119, -139.800, 736.376}, {-6.31127, -39.9676, -37.021, 406.051}, {-6.03650, -33.2381, -6.606, 317.776}}};

        const xt::xarray<double> tau_ctt_2010 = {
        {{-4.37113, -11.5882, -35.819, -334.047}, {-3.83239, -5.9057, -12.490, -118.284}, {-3.55326, -3.6596, -5.293, -63.559}},
        {{-4.69276, -20.2284, -64.919, 88.884}, {-4.15387, -13.3114, -28.402, 72.741}, {-3.87346, -10.4637, -17.408, 66.313}},
        {{-4.99071, -23.5873, -76.924, 184.782}, {-4.45311, -15.7732, -32.316, 122.705}, {-4.17280, -12.4909, -17.912, 83.285}},
        {{-5.26780, -27.2836, -78.971, 137.871}, {-4.73244, -18.4833, -31.875, 111.817}, {-4.45268, -14.7199, -17.969, 101.92}},
        {{-5.52826, -30.9051, -92.490, 248.096}, {-4.99491, -21.2360, -37.685, 194.208}, {-4.71587, -17.0820, -18.631, 136.672}},
        {{-5.77379, -34.7010, -105.937, 393.991}, {-5.24217, -24.2177, -39.153, 232.528}, {-4.96397, -19.6064, -18.858, 174.919}},
        {{-6.00609, -38.7383, -108.605, 365.208}, {-5.47664, -27.3005, -39.498, 246.918}, {-5.19921, -22.2617, -17.910, 208.494}},
        {{-6.22758, -42.7154, -119.622, 421.395}, {-5.69983, -30.4365, -44.300, 345.48}, {-5.42320, -24.9686, -19.688, 274.462}},
        {{-6.43933, -46.7581, -136.691, 651.38}, {-5.91298, -33.7584, -42.686, 346.629}, {-5.63704, -27.8965, -13.880, 236.975}},
        {{-6.64235, -50.9783, -145.462, 752.228}, {-6.11753, -37.056, -48.719, 473.905}, {-5.84215, -30.8119, -14.938, 316.006}},
        {{-6.83743, -55.2861, -152.651, 792.577}, {-6.31396, -40.5507, -46.771, 487.185}, {-6.03921, -33.8950, -9.122, 285.164}},
        {{-7.02582, -59.6037, -166.368, 989.879}, {-6.50353, -44.0797, -47.242, 543.889}, {-6.22941, -36.9673, -10.868, 418.414}}};

        inline xarray<double> polyval(const xt::xarray<double>& coeffs, double x) {
            xarray<double> result = xt::zeros<double>({coeffs.shape()[1]});
            for (size_t i = 0; i < coeffs.shape()[0]; ++i)
                result = result * x + xt::view(coeffs, i, xt::all());

            return result;
        }

        inline xarray<double> crit_value(int N = 1, const std::string& regression = "c", double nobs = inf) {
            if (N < 1 || N > 12) {
                throw std::invalid_argument("N must be between 1 and 12 (inclusive)");
            }
            
            const xarray<double>* tau;
            
            if (regression == "nc") {
                tau = &tau_nc_2010;
            } else if (regression == "c") {
                tau = &tau_c_2010;
            } else if (regression == "ct") {
                tau = &tau_ct_2010;
            } else if (regression == "ctt") {
                tau = &tau_ctt_2010;
            } else {
                throw std::invalid_argument("Unknown regression type: " + regression);
            }

            // N-1 because C++ is 0-indexed
            auto row = xt::view(*tau, N - 1, xt::all(), xt::all());

            if (nobs == inf) {
                // Return asymptotic critical values: first column
                return xt::eval(xt::view(row, xt::all(), 0));
            } else {
                // Reverse coefficients and evaluate polynomial at 1.0 / nobs
                auto coeffs = xt::eval(xt::flip(xt::transpose(row), 0));  // shape: (4, 3) -> flip -> (4, 3)
                return polyval(coeffs, 1.0 / nobs);
            }
        }
    }
}

#endif // MACKINNONCRITVALUES_H_
