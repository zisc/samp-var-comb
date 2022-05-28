#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <boost/log/attributes/value_extraction_fwd.hpp>
#include <boost/log/attributes/value_extraction.hpp>
#include <boost/log/trivial.hpp>
#include <R.h>
#include <Rinternals.h>
#include "boost_log_R/filter.hpp"
#include "boost_log_R/string_to_severity_level.hpp"

void boost_log_R_filter_configuration::verbosity(boost::log::trivial::severity_level verbosity_in) {
    verbosity_prop = verbosity_in;
}

void boost_log_R_filter_configuration::verbosity(const std::string& verbosity_in_str) {
    auto verbosity_in = string_to_severity_level(verbosity_in_str);
    verbosity(verbosity_in);
}

void boost_log_R_filter_configuration::verbosity(SEXP Rverbosity) {
    std::string verbosity_in_str(CHAR(STRING_ELT(Rverbosity,0)));
    verbosity(verbosity_in_str);
}

boost::log::trivial::severity_level boost_log_R_filter_configuration::verbosity(void) const {
    return verbosity_prop;
}

class boost_log_R_filter {
    public:
        typedef bool result_type;
        
        boost_log_R_filter(boost_log_R_filter_configuration config_in): config(std::move(config_in)) { }

        result_type operator() (const boost::log::attribute_value_set& attributes) {
            auto it = attributes.find(boost::log::aux::default_attribute_names::severity());
            if (it != attributes.end()) {
                const boost::log::attribute_value& value = it->second;
                auto record_severity = value.extract<boost::log::trivial::severity_level>();
                if (record_severity && record_severity.get() < config.verbosity()) {
                    return false;
                }
            }
            return true;
        }
        
    private:
        boost_log_R_filter_configuration config;
};

void set_boost_log_R_filter(const boost_log_R_filter_configuration& config) {
    boost::log::core::get()->set_filter(boost::log::filter(boost_log_R_filter(config)));
}

