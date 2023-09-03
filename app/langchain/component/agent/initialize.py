"""Load agent."""
# NOTE: cf. /usr/local/lib/python3.10/dist-packages/langchain/agents/initialize.py
# Copyright (c) Harrison Chase
# Copyright (c) 2023 Takehito Oshita
from typing import Any, Optional, Sequence

from langchain.agents.agent_types import AgentType
from langchain.agents.loading import AGENT_TO_CLASS
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools.base import BaseTool

from app.langchain.component.agent.agent_executor import CustomAgentExecutor


def initialize_agent(
    tools: Sequence[BaseTool],
    llm: BaseLanguageModel,
    agent: Optional[AgentType] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    agent_kwargs: Optional[dict] = None,
    **kwargs: Any,
) -> CustomAgentExecutor:
    """Load an agent executor given tools and LLM.

    Args:
        tools: List of tools this agent has access to.
        llm: Language model to use as the agent.
        agent: Agent type to use. If None and agent_path is also None, will default to
            AgentType.ZERO_SHOT_REACT_DESCRIPTION.
        callback_manager: CallbackManager to use. Global callback manager is used if
            not provided. Defaults to None.
        agent_path: Path to serialized agent to use.
        agent_kwargs: Additional key word arguments to pass to the underlying agent
        **kwargs: Additional key word arguments passed to the agent executor

    Returns:
        An agent executor
    """
    if agent is None:
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION
    else:
        if agent not in AGENT_TO_CLASS:
            raise ValueError(
                f"Got unknown agent type: {agent}. "
                f"Valid types are: {AGENT_TO_CLASS.keys()}."
            )
        agent_cls = AGENT_TO_CLASS[agent]
        agent_kwargs = agent_kwargs or {}
        agent_obj = agent_cls.from_llm_and_tools(
            llm, tools, callback_manager=callback_manager, **agent_kwargs
        )
    return CustomAgentExecutor.from_agent_and_tools(
        agent=agent_obj,
        tools=tools,
        callback_manager=callback_manager,
        **kwargs,
    )
