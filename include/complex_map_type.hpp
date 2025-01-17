#include <type_traits>
#include <complex>

#ifndef CMT_HPP
#define CMT_HPP 

// Define a template struct to map T to U
template <typename T>
struct map_type;

// Specializations for float and double
template <>
struct map_type<float> {
    using type = float;
};

template <>
struct map_type<double> {
    using type = double;
};

// Specializations for std::complex<float> and std::complex<double>
template <>
struct map_type<std::complex<float>> {
    using type = float;
};

template <>
struct map_type<std::complex<double>> {
    using type = double;
};

#endif