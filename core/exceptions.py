class HelloAgentsException(Exception):
    pass


class LLMException(HelloAgentsException):
    pass


class AgentException(HelloAgentsException):
    pass


class ConfigException(HelloAgentsException):
    pass


class ToolException(HelloAgentsException):
    pass
