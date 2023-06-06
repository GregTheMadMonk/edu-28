#pragma once

#include "base.hh"

#include <iostream>
#include <random>

namespace edu28 {

/**
 * \brief Normalize probability distribution
 * 
 * Makes it so that \int PdE = 1
 * 
 * \return New P
 */
std::vector<Real> probNormalize(const std::vector<Real>& E, std::vector<Real> P) {
    Real pInt = 0;

    for (std::size_t i = 0; i < E.size() - 1; ++i) {
        pInt += (P[i] + P[i+1]) * (E[i+1] - E[i]) / 2;
    }

    for (auto& p : P) p /= pInt;

    return P;
} // <-- std::vector<Real> probNormalize()

/**
 * \brief Roll a value from uniform distribution in the interval [from, to]
 */
template <typename ValueType>
ValueType uniformRoll(ValueType from, ValueType to) {
    // Set up RNG
    static thread_local std::random_device rd{};
    static thread_local std::mt19937 gen{rd()};

    static thread_local std::uniform_real_distribution<Real> dist(from, to);

    return dist(gen);
} // <-- Real uniformRoll()

/**
 * \brief Rolls a random value with a given probability distribution
 *
 * Assumes distribution defined by `P`, `E` is normalized
 */
Real rollScalar(const std::vector<Real>& E, const std::vector<Real>& P) {
    const auto roll = uniformRoll<Real>(0, 1);

    std::vector<Real> pInt(E.size(), 0);

    std::size_t idx = 0;

    while (idx + 1 < E.size()) {
        pInt[idx + 1] = pInt[idx] + (P[idx + 1] + P[idx]) * (E[idx + 1] - E[idx]) / 2;

        if (pInt[idx + 1] >= roll) break;
        ++idx;
    }

    const auto t = (roll - pInt[idx]) / (pInt[idx + 1] - pInt[idx]);

    return E[idx] + t * (E[idx + 1] - E[idx]);
} // <-- Real rollScalar()

} // <-- namespace edu28
