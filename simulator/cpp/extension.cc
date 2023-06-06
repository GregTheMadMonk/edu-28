// PyBind / Torch
#include <torch/extension.h>
#include <pybind11/numpy.h>

#include "prob.hh"
#include "signals.hh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "composeSignals",
        edu28::composeSignals,
        py::call_guard<py::gil_scoped_release>(),
        "Compose two signals into one"
    );

    m.def(
        "integrateSignal",
        edu28::integrateSignal,
        py::call_guard<py::gil_scoped_release>(),
        "Integrate the signal in terms of sum"
    );

    m.def(
        "integrateSignalRelative",
        edu28::integrateSignalRelative,
        py::call_guard<py::gil_scoped_release>(),
        "Integrate the signal in terms of sum (relative bounds)"
    );
    m.def( // With default argument
        "integrateSignalRelative",
        [] (const edu28::Signal& signal, edu28::Real intFrom, edu28::Real intTo) -> edu28::Real {
            return edu28::integrateSignalRelative(signal, intFrom, intTo);
        },
        py::call_guard<py::gil_scoped_release>(),
        "Integrate the signal in terms of sum (relative bounds)"
    );

    m.def(
        "probNormalize",
        edu28::probNormalize,
        py::call_guard<py::gil_scoped_release>(),
        "Normalize probability density"
    );

    m.def(
        "rollScalar",
        edu28::rollScalar,
        py::call_guard<py::gil_scoped_release>(),
        "Roll a scalar value according to a distribution"
    );

    py::class_<edu28::DoubleOverlapRollResult>(m, "DoubleOverlapRollResult")
        .def_readonly("offset",   &edu28::DoubleOverlapRollResult::offset)
        .def_readonly("amp1",     &edu28::DoubleOverlapRollResult::amp1)
        .def_readonly("amp2",     &edu28::DoubleOverlapRollResult::amp2)
        .def_readonly("integral", &edu28::DoubleOverlapRollResult::integral)
    ;

    m.def(
        "rollDoubleOverlap",
        edu28::rollDoubleOverlap,
        py::call_guard<py::gil_scoped_release>(),
        "Perform a random double-signal overlap simulation"
    );
    m.def( // With default arguments
        "rollDoubleOverlap",
        [] (
            const std::vector<edu28::Real>& E,
            const std::vector<edu28::Real>& P,
            const edu28::Signal& signal,
            edu28::Real intLeft, edu28::Real intRight
        ) {
            return edu28::rollDoubleOverlap(E, P, signal, intLeft, intRight, 0, 42);
        },
        py::call_guard<py::gil_scoped_release>(),
        "Perform a random double-signal overlap simulation"
    );

    m.def(
        "rollDoubleOverlapBulk",
        edu28::rollDoubleOverlapBulk,
        py::call_guard<py::gil_scoped_release>(),
        "Perform several random double-signal overlap simulations"
    );
    m.def( // With default arguments
        "rollDoubleOverlapBulk",
        [] (
            std::size_t bulkSize,
            const std::vector<edu28::Real>& E,
            const std::vector<edu28::Real>& P,
            const edu28::Signal& signal,
            edu28::Real intLeft, edu28::Real intRight
        ) {
            return edu28::rollDoubleOverlapBulk(bulkSize, E, P, signal, intLeft, intRight, 0, 42);
        },
        py::call_guard<py::gil_scoped_release>(),
        "Perform several random double-signal overlap simulations"
    );

    m.def(
        "rollSingle",
        edu28::rollSingle,
        py::call_guard<py::gil_scoped_release>(),
        "Perform a single signal roll"
    );
    m.def(
        "rollSingleBulk",
        edu28::rollSingleBulk,
        py::call_guard<py::gil_scoped_release>(),
        "Perform several random single signal rolls"
    );

    m.def(
        "toList",
        [] (const std::vector<edu28::DoubleOverlapRollResult>& res) {
            std::vector<std::vector<edu28::Real>> ret(res.size(), std::vector<edu28::Real>(4));

            for (std::size_t i = 0; i < res.size(); ++i) {
                ret[i][0] = res[i].offset;
                ret[i][1] = res[i].amp1;
                ret[i][2] = res[i].amp2;
                ret[i][3] = res[i].integral;
            }

            return ret;
        },
        py::call_guard<py::gil_scoped_release>(),
        "Convert a list of DoubleOverlapResult's to 2D array"
    );

}
