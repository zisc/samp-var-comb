#include_next "R_ext/RS.h"
#undef ERROR
// If we don't undefine the ERROR macro after calling R headers,
// it is expanded somewhere in torch/torch.h causing this error:
// "
// /usr/share/R/include/R_ext/RS.h:55:17: error: expected unqualified-id before ‘)’ token
// #define ERROR   ),error(R_problem_buf);}
// "

