#include <sstream>
#include <string>
#include <boost/log/trivial.hpp>

boost::log::trivial::severity_level string_to_severity_level(const std::string& verbosity_str) {
    if (verbosity_str == "trace") {
        return boost::log::trivial::severity_level::trace;
    } else if (verbosity_str == "debug") {
        return boost::log::trivial::severity_level::debug;
    } else if (verbosity_str == "info") {
        return boost::log::trivial::severity_level::info;
    } else if (verbosity_str == "warning") {
        return boost::log::trivial::severity_level::warning;
    } else if (verbosity_str == "error") {
        return boost::log::trivial::severity_level::error;
    } else if (verbosity_str == "fatal") {
        return boost::log::trivial::severity_level::fatal;
    } else {
        std::ostringstream ss;
        ss << '"' << verbosity_str << "\" is not a valid verbosity level. Try one of: trace, debug, info, warning, error and fatal.";
        throw std::runtime_error(ss.str());
    }
}

