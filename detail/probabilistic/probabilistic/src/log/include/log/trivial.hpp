#ifndef PROBABILISTIC_LOG_TRIVIAL_GUARD
#define PROBABILISTIC_LOG_TRIVIAL_GUARD

#include <utility>
#include <boost/log/trivial.hpp>

#ifdef NDEBUG
    namespace probabilistic {

        class null_logger {
            public:
                template <typename T>
                null_logger& operator<<(const T& t) {
                    return *this;
                }
        };

    }

    #define PROBABILISTIC_LOG_TRIVIAL_TRACE if (false) ::probabilistic::null_logger()
    #define PROBABILISTIC_LOG_TRIVIAL_DEBUG if (false) ::probabilistic::null_logger()
#else
    #define PROBABILISTIC_LOG_LOCATION_PRELUDE __FILE__ ":" << __LINE__ << ":\n"

    #define PROBABILISTIC_LOG_TRIVIAL_TRACE BOOST_LOG_TRIVIAL(trace) << PROBABILISTIC_LOG_LOCATION_PRELUDE
    #define PROBABILISTIC_LOG_TRIVIAL_DEBUG BOOST_LOG_TRIVIAL(debug) << PROBABILISTIC_LOG_LOCATION_PRELUDE
#endif

#define PROBABILISTIC_LOG_TRIVIAL_INFO BOOST_LOG_TRIVIAL(info)
#define PROBABILISTIC_LOG_TRIVIAL_WARNING BOOST_LOG_TRIVIAL(warning)
#define PROBABILISTIC_LOG_TRIVIAL_ERROR BOOST_LOG_TRIVIAL(error)
#define PROBABILISTIC_LOG_TRIVIAL_FATAL BOOST_LOG_TRIVIAL(fatal)

#endif

