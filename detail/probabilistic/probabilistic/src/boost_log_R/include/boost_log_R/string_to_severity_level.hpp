#ifndef BOOST_LOG_R_STRING_TO_SEVERITY_LEVEL_GUARD
#define BOOST_LOG_R_STRING_TO_SEVERITY_LEVEL_GUARD

#include <string>
#include <boost/log/trivial.hpp>

boost::log::trivial::severity_level string_to_severity_level(const std::string& verbosity_str);

#endif
