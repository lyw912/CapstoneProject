"""

Deep Search Agent

深度搜索 AI 代理（LangGraph 编排）

"""



from .agent import DeepSearchAgent, AnspireSearchAgent, create_agent

from .graph import MediaAgentState, build_media_agent_graph

from .utils.config import Settings



__version__ = "1.0.0"

__author__ = "Deep Search Agent Team"



__all__ = [

    "DeepSearchAgent",

    "AnspireSearchAgent",

    "create_agent",

    "Settings",

    "MediaAgentState",

    "build_media_agent_graph",

]


