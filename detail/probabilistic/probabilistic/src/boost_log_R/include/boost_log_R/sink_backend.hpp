#ifndef R_SINK_BACKEND_HPP_GUARD
#define R_SINK_BACKEND_HPP_GUARD

#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/log/sinks/frontend_requirements.hpp>

class R_sink_backend :
    public boost::log::sinks::basic_formatted_sink_backend<
        char,
        boost::log::sinks::synchronized_feeding
    >
{
    public:
        void consume(const boost::log::record_view& record, const string_type& message);
};

void initialise_boost_log_R_sink_backend(void);

#endif

