from .anthropic import SynapseAnthropic
from .openai import SynapseOpenAI

try:
    from .langchain import SynapseAgentExecutor
except Exception:
    SynapseAgentExecutor = None

try:
    from .llamaindex import SynapseFunctionCallingAgent
except Exception:
    SynapseFunctionCallingAgent = None

try:
    from .crewai import SynapseCrewTaskExecutor
except Exception:
    SynapseCrewTaskExecutor = None

__all__ = [
    "SynapseOpenAI",
    "SynapseAnthropic",
    "SynapseAgentExecutor",
    "SynapseFunctionCallingAgent",
    "SynapseCrewTaskExecutor",
]
