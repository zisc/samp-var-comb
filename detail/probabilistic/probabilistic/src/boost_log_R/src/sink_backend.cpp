#include <boost/log/attributes/value_extraction_fwd.hpp>
#include <boost/log/attributes/value_extraction.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <R_ext/Print.h>
#include "boost_log_R/sink_backend.hpp"

void R_sink_backend::consume(const boost::log::record_view& record, const string_type& message) {
    const boost::log::attribute_value_set& values = record.attribute_values();
    auto it = values.find(boost::log::aux::default_attribute_names::severity());
    if (it != values.end()) {
        const boost::log::attribute_value& value = it->second;
        auto record_severity = value.extract<boost::log::trivial::severity_level>();
        if (record_severity && record_severity.get() < boost::log::trivial::error) {
            Rprintf("%s\n", message.c_str());
        } else {
            REprintf("%s\n", message.c_str());
        }
    } else {
        Rprintf("%s\n", message.c_str());
    }
}

void initialise_boost_log_R_sink_backend(void) {
    auto sink = boost::make_shared<boost::log::sinks::synchronous_sink<R_sink_backend>>();
    boost::log::core::get()->add_sink(sink);
}

