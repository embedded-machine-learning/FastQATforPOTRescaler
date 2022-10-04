__HIGH_PRES__ = False

"""
Enables high precision mode, network will take longer and will need more memory 
"""

__HIGH_PRES_USE_RUNNING__ = True

"""
In High precision mode use the running stats of the bn in training
"""


__DEBUG__ = False
"""
Enables DEBUG code segments 
"""



from .logger import logger
from .logger import logger_forward,logger_init

logger.log_level=0



# __LOG_VERBOSE__ = 0
# """
# Sets the __LOG_VERBOSE__ score for the whole code
# """

# __LOG_LEVEL_IMPORTANT__ = 0
# __LOG_LEVEL_NORMAL__ = 1
# __LOG_LEVEL_DEBUG__ = 2
# __LOG_LEVEL_HIGH_DETAIL__ = 3
# __LOG_LEVEL_TO_MUCH__ = 4


# def LOG(lvl: int, msg: str, value=None):
#     """
#     LOG Logs something with an according level

#     only if :param: `__VERBOSE__` is equal ore higher

#     :param lvl: The Log level
#     :type lvl: int
#     :param msg: The Log message
#     :type msg: str
#     """
#     global __LOG_VERBOSE__
#     if __LOG_VERBOSE__ >= lvl:
#         if value == None:
#             print(f"LOG level: {lvl}\n{msg}\n================================================================================")
#         else:
#             print(f"LOG level: {lvl}\n{msg}:{value}\n================================================================================")
