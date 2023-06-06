#pragma once

#include <thread>

#include "base.hh"
#include "prob.hh"

namespace edu28 {

using Signal = std::tuple<std::vector<Real>, std::vector<Real>>;

/**
 * \brief Yerforms a composition of signals
 *
 * \param signal1 - first signal
 * \param signal2 - second signal
 * \param offset  - offset of the second signal relative to the first one
 * \param amp1    - first signal multiplier
 * \param amp2    - second signal multiplier
 *
 * \throws std::runtime_error if signal grids aren't aligned
*/
Signal composeSignals(
    const Signal& signal1,
    const Signal& signal2,
    Real offset,
    Real amp1,
    Real amp2
) {
    Signal signal = signal1;
    auto& [ X, Y ] = signal;

    const auto& [ X2, Y2 ] = signal2;

    // Modify first signal's amplitude
    for (auto& p : Y) {
        p *= amp1;
    }

    // Calculate index offset
    const std::size_t iOffset = [&X2, &X, offset] {
        for (std::size_t i = 0; i < X2.size(); ++i) {
            if (X2.at(i) - X.at(0) == offset) return i;
        }

        throw std::runtime_error("composeSignals expects `offset` argument to be in the signals' grids");
    } ();

    // Overlap signals
    for (std::size_t i = 0; i < X.size(); ++i) {
        if (i + iOffset >= X.size()) break;

        Y.at(i + iOffset) += Y2.at(i) * amp2;
    }

    return signal;
} // <-- Signal composeSignals()

/**
 * \brief "Integrates" the given signal in the interval
 *
 * Integration is performed by summing signal's `Y` values
 *
 * \param signal  - given signal
 * \param intFrom - _absolute_ left integration boundary
 * \param intTo   - _absolute_ right integration boundary
*/
Real integrateSignal(const Signal& signal, Real intFrom, Real intTo) {
    Real ret = 0;

    const auto& [ X, Y ] = signal;

    for (std::size_t i = 0; i < X.size(); ++i) {
        ret += ((X.at(i) >= intFrom) && (X.at(i) <= intTo)) * Y.at(i); // Branchless :)
    }

    return ret;
} // <-- Real integrateSignals()

/**
 * \brief "Integrates" the given signal in the interval
 *        given via offests relative to the point
 * 
 * \param signal      - given signal
 * \param offsetLeft  - left integration boundary offset relative to the center.
 *                      Same as intLeft = center - offsetLeft
 * \param offsetRight - right integration boundary offset relative to the center.
 *                      Same as intRight = center + offsetRight
 * \param center      - A point inside integration interval that offsets relate to
 */
Real integrateSignalRelative(const Signal& signal, Real offsetLeft, Real offsetRight, Real center = 9) {
    return integrateSignal(signal, center - offsetLeft, center + offsetRight);
} // <-- Real integrateSignalsRelative()

/**
 * \brief Double overlap roll result
 *
 * Includes the result of integration as well as information about rolled values
 */
struct DoubleOverlapRollResult {
    /// \brief Roll signal peak offset
    int offset;
    /// \brief Roll's first signal amplitude
    Real amp1;
    /// \brief Roll's second signal amplitude
    Real amp2;
    /// \brief Result integral
    Real integral;
}; // <-- struct DoubleOverlapRollResult

/**
 * \brief Rolls a double overlapped signal
 *
 * Offset is assumed to have a uniform integer distribution between offsetMin and offsetMax
 * Amplitudes are determined as random values with distribution given by `P` and `E`
 *
 * \param E         - distribution E
 * \param P         - distribution P
 * \param signal    - signal shape
 * \param intLeft   - left integration border (offset relative to 9)
 * \param intRight  - right integration border (offset relative to 9)
 * \param offsetMin - minimum signal peak offset value
 * \param offsetMax - maximum signal peak offset value
 *
 * See \ref integrateSignalRelative() for better explaination of `intLeft` and `intRight`
 */
DoubleOverlapRollResult rollDoubleOverlap(
    const std::vector<Real>& E, const std::vector<Real>& P,
    const Signal& signal,
    Real intLeft, Real intRight,
    int offsetMin = 0, int offsetMax = 42
) {
    const int offset = uniformRoll(offsetMin, offsetMax + 1);
    const Real amp1 = rollScalar(E, P);
    const Real amp2 = rollScalar(E, P);

    return DoubleOverlapRollResult{
        offset, amp1, amp2,
        integrateSignalRelative(composeSignals(signal, signal, offset, amp1, amp2), intLeft, intRight)
    };
} // <-- DoubleOverlapRollResult rollDoubleOverlap()

/// \brief Implementation detail namespace
namespace detail {

    template <typename Func, typename... Args>
    requires std::invocable<Func, Args...>
    std::vector< std::invoke_result_t< Func, Args... > >
    runInBulkHelper(std::size_t bulkSize, Func func, Args... args)
    {
        const auto threads = std::thread::hardware_concurrency();

        using ResultType = std::invoke_result_t<Func, Args...>;

        std::vector<ResultType> ret(bulkSize);
        std::vector<std::thread> workers;

        std::size_t start = 0;
        for (std::size_t thread = 0; thread < threads; ++thread) {
            std::size_t end = (thread == threads - 1) ? bulkSize : (start + bulkSize / threads);

            workers.emplace_back(
                [start, end, &ret, &func, &args...] {
                    for (std::size_t i = start; i < end; ++i) {
                        ret[i] = std::invoke(func, args...);
                    }
                }
            );

            start = end;
        }

        for (auto& t : workers) t.join();

        return ret;
    } // <-- runInBulkHelper()

} // <-- namespace detail

/**
 * \brief Perform \ref rollDoubleOverlap in bulk
 */
std::vector<DoubleOverlapRollResult> rollDoubleOverlapBulk(
    std::size_t bulkSize,
    const std::vector<Real>& E, const std::vector<Real>& P,
    const Signal& signal,
    Real intLeft, Real intRight,
    int offsetMin = 0, int offsetMax = 42
) {
    return detail::runInBulkHelper(
        bulkSize,
        rollDoubleOverlap,
        E, P, signal, intLeft, intRight, offsetMin, offsetMax
    );
} // <-- std::vector<DoubleOverlapRollResult> rollDoubleOverlapBulk()

/**
 * \brief Single signal roll result - integral of the signal
 *
 * \param E         - distribution E
 * \param P         - distribution P
 * \param signal    - signal shape
 * \param intLeft   - left integration border (offset relative to 9)
 * \param intRight  - right integration border (offset relative to 9)
 */
Real rollSingle(
    const std::vector<Real>& E, const std::vector<Real>& P,
    Signal signal,
    Real intLeft, Real intRight
) {
    const Real amp = rollScalar(E, P);

    // Transform signal to the new amplitude
    auto& [ X, Y ] = signal;
    for (auto& y : Y) y *= amp;

    return integrateSignalRelative(signal, intLeft, intRight);
} // <-- std::vector<Real> rollSingle()

/**
 * \brief Perform \ref rollSingle in bulk
 */
std::vector<Real> rollSingleBulk(
    std::size_t bulkSize,
    const std::vector<Real>& E, const std::vector<Real>& P,
    const Signal& signal,
    Real intLeft, Real intRight
) {
    return detail::runInBulkHelper(
        bulkSize,
        rollSingle,
        E, P, signal, intLeft, intRight
    );
} // <-- std::vector<Real> rollSingelBulk()

} // <-- namespace edu28
