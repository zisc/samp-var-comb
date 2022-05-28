#ifndef BOOST_LOG_R_FILTER_HPP_GUARD
#define BOOST_LOG_R_FILTER_HPP_GUARD

#include <string>
#include <boost/log/trivial.hpp>
#include <R.h>
#include <Rinternals.h>

class boost_log_R_filter_configuration;
void set_boost_log_R_filter(const boost_log_R_filter_configuration&);

class boost_log_R_filter_configuration {
    public:
        void verbosity(boost::log::trivial::severity_level);
        void verbosity(const std::string&);
        void verbosity(SEXP);
        boost::log::trivial::severity_level verbosity(void) const;
    private:
        boost::log::trivial::severity_level verbosity_prop;
};

#endif


