from .logger import logger, print_all, print_custom
from .logger import print_left_of, print_between

logger_forward = logger.log(
    {
        "+="    : print_left_of("+="),
        "-="    : print_left_of("-="),
        "="     : print_left_of("="),
        "if"    : print_between("if", ":"),
        "else"  : print_custom("Taken",True,1),
    }
)

logger_init = logger.log(
    {
        'self.register_buffer("'    : print_between('self.register_buffer("'    , '",', prefix="self.", pre=True),
        'self.register_parameter("' : print_between('self.register_parameter("' , '",', prefix="self.", pre=True),
    }
)
