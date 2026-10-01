# Google ADK + AMD Instinct - A2A Purchasing Concierge
#
# Reconstructed from Tutorial_Powering_Google_ADK_on_AMD_Platform_and_Local_LLMs.ipynb
# (utils branch). Code cells are verbatim from the notebook; only the step
# numbering, comments and the missing sse-starlette install are added.
#
# PREREQUISITES - both must answer 200 before starting:
#   vllm serve meta-llama/Llama-3.1-8B-Instruct \
#       --enable-auto-tool-choice --tool-call-parser llama3_json \
#       --port 8088 --chat-template vllm/examples/tool_chat_template_llama3.1_json.jinja \
#       --gpu-memory-utilization 0.6
#   ollama serve
#
# Run cells in order. The servers must be started from notebook cells (not a
# separate terminal) so the env vars below reach them.

# ============================================================================ #
# STEP 0 - dependencies  (run this cell first, it is the only real install)
# ============================================================================ #
# NOTE: use a2a-sdk[http-server] - it pulls in starlette + sse-starlette, which
# utils/server.py imports. Do NOT install the bare `a2a` package: on PyPI that
# is an unrelated Azure/AWS service-lookup CLI, not the Agent2Agent SDK.

# ============================================================================ #
# STEP 1 - environment variables
# ============================================================================ #
import os

os.environ["VLLM_MODEL"] = "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
os.environ["OPENAI_API_BASE"] = "http://localhost:8088/v1"
os.environ["OLLAMA_MODEL"] = "ollama_chat/llama3.1:latest"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

print("env set")

# ============================================================================ #
# STEP 2 - shared pydantic schemas
# ============================================================================ #
from typing import Literal, List, Dict, Any
from pydantic import BaseModel


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class OrderItem(BaseModel):
    name: str
    quantity: int
    price: int


class Order(BaseModel):
    order_id: str
    status: str
    order_items: List[OrderItem]


# ============================================================================ #
# STEP 3 - create_burger_order tool
# ============================================================================ #
import uuid
from crewai.tools import tool


@tool("create_order")
def create_burger_order(order_items: List[OrderItem]) -> str:
    """
    Creates a new burger order with the given order items.

    Args:
        order_items: List of order items to be added to the order.

    Returns:
        str: A message indicating that the order has been created.
    """
    try:
        order_id = str(uuid.uuid4())
        order = Order(order_id=order_id, status="created", order_items=order_items)
        print("===")
        print(f"order created: {order}")
        print("===")
    except Exception as e:
        print(f"Error creating order: {e}")
        return f"Error creating order: {e}"
    return f"Order {order.model_dump()} has been created"


# ============================================================================ #
# STEP 4 - BurgerSellerAgent (CrewAI + vLLM)
# ============================================================================ #
from crewai import Agent as crewagent
from crewai import Crew as crewcrew
from crewai import LLM as crewllm
from crewai import Task as crewtask
from crewai import Process as crewprocess


class BurgerSellerAgent:
    TaskInstruction = """
# INSTRUCTIONS

You are a specialized assistant for a burger store.
Your sole purpose is to answer questions about what is available on burger menu and price also handle order creation.
If the user asks about anything other than burger menu or order creation, politely state that you cannot help with that topic and can only assist with burger menu and order creation.
Do not attempt to answer unrelated questions or use tools for other purposes.

# CONTEXT

Received user query: {user_prompt}
Session ID: {session_id}

Provided below is the available burger menu and it's related price:
- Classic Cheeseburger: IDR 85K
- Double Cheeseburger: IDR 110K
- Spicy Chicken Burger: IDR 80K
- Spicy Cajun Burger: IDR 85K

# RULES

- If user want to do something, you will be following this order:
    1. Always ensure the user already confirmed the order and total price. This confirmation may already given in the user query.
    2. Use `create_burger_order` tool to create the order
    3. Finally, always provide response to the user about the detailed ordered items, price breakdown and total, and order ID

- Set response status to input_required if asking for user order confirmation.
- Set response status to error if there is an error while processing the request.
- Set response status to completed if the request is complete.
- DO NOT make up menu or price, Always rely on the provided menu given to you as context.
"""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def invoke(self, query, sessionId) -> str:
        burger_agent = crewagent(
            role="Burger Seller Agent",
            goal=(
                "Help user to understand what is available on burger menu and price "
                "also handle order creation."
            ),
            backstory=("You are an expert and helpful burger seller agent."),
            verbose=False,
            allow_delegation=False,
            tools=[create_burger_order],
            llm=crewllm(
                model=os.environ.get(
                    "VLLM_MODEL", "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
                ),
                api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:8088/v1"),
            ),
        )

        agent_task = crewtask(
            description=self.TaskInstruction,
            output_pydantic=ResponseFormat,
            agent=burger_agent,
            expected_output=(
                "A JSON object with 'status' and 'message' fields."
                "Set response status to input_required if asking for user order confirmation."
                "Set response status to error if there is an error while processing the request."
                "Set response status to completed if the request is complete."
            ),
        )

        crew = crewcrew(
            tasks=[agent_task],
            agents=[burger_agent],
            verbose=False,
            process=crewprocess.sequential,
        )

        inputs = {"user_prompt": query, "session_id": sessionId}
        response = crew.kickoff(inputs)
        return self.get_agent_response(response)

    def get_agent_response(self, response):
        response_object = response.pydantic
        if response_object and isinstance(response_object, ResponseFormat):
            if response_object.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response_object.message,
                }
            elif response_object.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": response_object.message,
                }
            elif response_object.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": response_object.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }


# ============================================================================ #
# STEP 5 - Burger A2A server on 10001
# ============================================================================ #
from utils.server import A2AServer
from utils.a2a_types import AgentCard, AgentCapabilities, AgentSkill, AgentAuthentication
from utils.push_notification_auth import PushNotificationSenderAuth
from utils.task_manager import AgentTaskManager
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUTH_USERNAME = "burgeruser123"
AUTH_PASSWORD = "burgerpass123"


def burger_main(host, port):
    """Starts the Burger Seller Agent server."""
    try:
        capabilities = AgentCapabilities(pushNotifications=True)
        skill = AgentSkill(
            id="create_burger_order",
            name="Burger Order Creation Tool",
            description="Helps with creating burger orders",
            tags=["burger order creation"],
            examples=["I want to order 2 classic cheeseburgers"],
        )
        agent_card = AgentCard(
            name="burger_seller_agent",
            description="Helps with creating burger orders",
            # The URL provided here is for the sake of demo,
            # in production you should use a proper domain name
            url=f"http://{host}:{port}/",
            version="1.0.0",
            authentication=AgentAuthentication(schemes=["Basic"]),
            defaultInputModes=BurgerSellerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=BurgerSellerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=BurgerSellerAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host=host,
            port=port,
            auth_username=AUTH_USERNAME,
            auth_password=AUTH_PASSWORD,
        )

        server.app.add_route(
            "/.well-known/jwks.json",
            notification_sender_auth.handle_jwks_endpoint,
            methods=["GET"],
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        raise


# --- Global variable to hold the server thread reference ---
# This allows you to stop it later from another cell if needed
global burger_thread
burger_thread = None

# --- Main execution in the Jupyter cell ---
if burger_thread is not None and burger_thread.is_alive():
    print("Server is already running.")
else:
    # Define host and port
    server_host = "0.0.0.0"
    server_port = 10001

    # Create and start the thread
    burger_thread = threading.Thread(target=burger_main, args=(server_host, server_port))
    burger_thread.daemon = True  # Allows the main program to exit even if the thread is still running
    burger_thread.start()

    print(
        f"Server thread started. Waiting a moment for server to initialize on http://{server_host}:{server_port}"
    )
    time.sleep(5)  # Give it a few seconds to boot up


# ============================================================================ #
# STEP 6 - CHECKPOINT: test the burger agent
# ============================================================================ #
agent = BurgerSellerAgent()
print(agent)
result = agent.invoke("1 classic cheeseburger pls", "default_session")
print(result)

# ============================================================================ #
# STEP 7 - create_pizza_order tool (LangGraph + Ollama)
# ============================================================================ #
from langchain_ollama import ChatOllama
from langchain_core.tools import tool as lc_tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

memory = MemorySaver(
    serde=JsonPlusSerializer(
        allowed_msgpack_modules=[("__main__", "ResponseFormat")]
    )
)
# memory = MemorySaver()


@lc_tool
def create_pizza_order(order_items: List[OrderItem]) -> str:
    """
    Creates a new pizza order with the given order items.

    Args:
        order_items: List of order items to be added to the order.

    Returns:
        str: A message indicating that the order has been created.
    """
    try:
        order_id = str(uuid.uuid4())
        order = Order(order_id=order_id, status="created", order_items=order_items)
        print("===")
        print(f"order created: {order}")
        print("===")
    except Exception as e:
        print(f"Error creating order: {e}")
        return f"Error creating order: {e}"
    return f"Order {order.model_dump()} has been created"


# ============================================================================ #
# STEP 8 - PizzaSellerAgent (LangGraph + Ollama)
# ============================================================================ #
class PizzaSellerAgent:
    SYSTEM_INSTRUCTION = """
# ROLE

You are an expert pizza seller agent. Your sole purpose is to answer questions about available pizza menu items and their prices, and to handle pizza order creation. If the user asks about anything other than pizza menu or order creation, politely state that you cannot help with that topic and can only assist with pizza menu and order creation. Do not attempt to answer unrelated questions or use tools for other purposes.

# CONTEXT

Received user query: {user_prompt}
Session ID: {session_id}

Here is the pizza menu with prices:
- Pepperoni Pizza: IDR 110K
- Hawaiian Pizza: IDR 110K
- Veggie Pizza: IDR 100K
- BBQ Chicken Pizza: IDR 130K

# RULES

- If user want to do something, you will be following this order:
    1. Always ensure the user already confirmed the order and total price. This confirmation may already given in the user query.
    2. Use `create_pizza_order` tool to create the order
    3. Finally, always provide response to the user about the detailed ordered items, price breakdown and total, and order ID

- Set response status to input_required if asking for user order confirmation.
- Set response status to error if there is an error while processing the request.
- Set response status to completed if the request is complete.
- DO NOT make up menu or price, Always rely on the provided menu given to you as context.
"""
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self.model = ChatOllama(
            model="llama3.1:latest",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        self.tools = [create_pizza_order]
        self.graph = create_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            system_prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def invoke(self, query, sessionId) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        self.graph.invoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }


# ============================================================================ #
# STEP 9 - pizza environment variables
# ============================================================================ #
os.environ["API_KEY"] = "pizza123"


# ============================================================================ #
# STEP 10 - Pizza A2A server on 10000
# ============================================================================ #
def pizza_main(host, port):
    """Starts the Pizza Seller Agent server."""
    try:
        capabilities = AgentCapabilities(pushNotifications=True)
        skill = AgentSkill(
            id="create_pizza_order",
            name="Pizza Order Creation Tool",
            description="Helps with creating pizza orders",
            tags=["pizza order creation"],
            examples=["I want to order 2 pepperoni pizzas"],
        )
        agent_card = AgentCard(
            name="pizza_seller_agent",
            description="Helps with creating pizza orders",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            authentication=AgentAuthentication(schemes=["Bearer"]),
            defaultInputModes=PizzaSellerAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=PizzaSellerAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=PizzaSellerAgent(),
                notification_sender_auth=notification_sender_auth,
            ),
            host=host,
            port=port,
            api_key=os.environ.get("API_KEY"),
        )

        server.app.add_route(
            "/.well-known/jwks.json",
            notification_sender_auth.handle_jwks_endpoint,
            methods=["GET"],
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        raise


global pizza_thread
pizza_thread = None

if pizza_thread is not None and pizza_thread.is_alive():
    print("Pizza server is already running.")
else:
    server_host = "0.0.0.0"
    server_port = 10000
    pizza_thread = threading.Thread(target=pizza_main, args=(server_host, server_port))
    pizza_thread.daemon = True
    pizza_thread.start()
    print(
        f"Server thread started. Waiting a moment for server to initialize on http://{server_host}:{server_port}"
    )
    time.sleep(5)


# ============================================================================ #
# STEP 11 - CHECKPOINT: test the pizza agent
# ============================================================================ #
agent = PizzaSellerAgent()
print(agent)
result = agent.invoke("I will order 3 Pepperoni Pizza", "default_session")
print(result)

# ============================================================================ #
# STEP 12 - root agent environment variables
# ============================================================================ #
import sys

os.environ["PIZZA_SELLER_AGENT_AUTH"] = "pizza123"
os.environ["PIZZA_SELLER_AGENT_URL"] = "http://localhost:10000"
os.environ["BURGER_SELLER_AGENT_AUTH"] = "burgeruser123:burgerpass123"
os.environ["BURGER_SELLER_AGENT_URL"] = "http://localhost:10001"

# ============================================================================ #
# STEP 13 - RemoteAgentConnections
# ============================================================================ #
from typing import Callable
from utils.a2a_types import (
    AgentCard,
    Task,
    TaskSendParams,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from utils.client import A2AClient

TaskCallbackArg = Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
TaskUpdateCallback = Callable[[TaskCallbackArg, AgentCard], Task]

KNOWN_AUTH = {
    "pizza_seller_agent": os.getenv("PIZZA_SELLER_AGENT_AUTH", "api_key"),
    "burger_seller_agent": os.getenv("BURGER_SELLER_AGENT_AUTH", "user:pass"),
}


class RemoteAgentConnections:
    """A class to hold the connections to the remote agents."""

    def __init__(self, agent_card: AgentCard, agent_url: str):
        auth = KNOWN_AUTH.get(agent_card.name, None)
        self.agent_client = A2AClient(agent_card, auth=auth, agent_url=agent_url)
        self.card = agent_card

        self.conversation_name = None
        self.conversation = None
        self.pending_tasks = set()

    def get_agent(self) -> AgentCard:
        return self.card

    async def send_task(
        self,
        request: TaskSendParams,
        task_callback: TaskUpdateCallback | None,
    ) -> Task | None:
        response = await self.agent_client.send_task(request.model_dump())
        merge_metadata(response.result, request)
        # For task status updates, we need to propagate metadata and provide
        # a unique message id.
        if (
            hasattr(response.result, "status")
            and hasattr(response.result.status, "message")
            and response.result.status.message
        ):
            merge_metadata(response.result.status.message, request.message)
            m = response.result.status.message
            if not m.metadata:
                m.metadata = {}
            if "message_id" in m.metadata:
                m.metadata["last_message_id"] = m.metadata["message_id"]
            m.metadata["message_id"] = str(uuid.uuid4())

        if task_callback:
            task_callback(response.result, self.card)
        return response.result


def merge_metadata(target, source):
    if not hasattr(target, "metadata") or not hasattr(source, "metadata"):
        return
    if target.metadata and source.metadata:
        target.metadata.update(source.metadata)
    elif source.metadata:
        target.metadata = dict(**source.metadata)


# ============================================================================ #
# STEP 14 - PurchasingAgent
# ============================================================================ #
import json
import httpx

from google.adk.models.lite_llm import LiteLlm
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from utils.card_resolver import A2ACardResolver
from utils.a2a_types import (
    AgentCard,
    Message,
    TaskState,
    Task,
    TaskSendParams,
    TextPart,
    Part,
)


class PurchasingAgent:
    """The purchasing agent.

    This is the agent responsible for choosing which remote seller agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: List[str],
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            try:
                card = card_resolver.get_agent_card()
                # The URL accessed here should be the same as the one provided in the agent card
                # However, in this demo we are using the URL provided in the key arguments
                remote_connection = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.remote_agent_connections[card.name] = remote_connection
                self.cards[card.name] = card
            except httpx.ConnectError:
                print(f"ERROR: Failed to get agent card from : {address}")
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = "\n".join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model=LiteLlm(model=os.getenv("OLLAMA_MODEL")),
            name="purchasing_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This purchasing agent orchestrates the decomposition of the user purchase request into"
                " tasks that can be performed by the seller agents."
            ),
            tools=[
                self.send_task,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_active_agent(context)
        return f"""You are an expert purchasing delegator that can delegate the user product inquiry and purchase request to the
appropriate seller remote agents.

Execution:
- For actionable tasks, you can use `send_task` to assign tasks to remote agents to perform.
- When the remote agent is repeatedly asking for user confirmation, assume that the remote agent doesn't have access to user's conversation context.
    So improve the task description to include all the necessary information related to that agent
- Never ask user permission when you want to connect with remote agents. If you need to make connection with multiple remote agents, directly
    connect with them without asking user permission or asking user preference
- Always show the detailed response information from the seller agent and propagate it properly to the user.
- If the remote seller is asking for confirmation, rely the confirmation question to the user if the user haven't do so.
- If the user already confirmed the related order in the past conversation history, you can confirm on behalf of the user
- Do not give irrelevant context to remote seller agent. For example, ordered pizza item is not relevant for the burger seller agent
- Never ask order confirmation to the remote seller agent

Please rely on tools to address the request, and don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

If there is an active agent, send the request to that agent with the update task tool.

Agents:
{self.agents}

Current active seller agent: {current_agent["active_agent"]}
"""

    def check_active_agent(self, context: ReadonlyContext):
        state = context.state
        if (
            "session_id" in state
            and "session_active" in state
            and state["session_active"]
            and "active_agent" in state
        ):
            return {"active_agent": f"{state['active_agent']}"}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request):
        state = callback_context.state
        if "session_active" not in state or not state["session_active"]:
            if "session_id" not in state:
                state["session_id"] = str(uuid.uuid4())
            state["session_active"] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            print(f"Found agent card: {card.model_dump()}")
            print("=" * 100)
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_task(self, agent_name: str, task: str, tool_context: ToolContext):
        """Sends a task to remote seller agent

        This will send a message to the remote agent named agent_name.

        Args:
            agent_name: The name of the agent to send the task to.
            task: The comprehensive conversation context summary
                and goal to be achieved regarding user inquiry and purchase request.
            tool_context: The tool context this method runs in.

        Yields:
            A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        state = tool_context.state
        state["active_agent"] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
        if "task_id" in state:
            taskId = state["task_id"]
        else:
            taskId = str(uuid.uuid4())
        sessionId = state["session_id"]
        task: Task
        messageId = ""
        metadata = {}
        if "input_message_metadata" in state:
            metadata.update(**state["input_message_metadata"])
            if "message_id" in state["input_message_metadata"]:
                messageId = state["input_message_metadata"]["message_id"]
        if not messageId:
            messageId = str(uuid.uuid4())
        metadata.update(**{"conversation_id": sessionId, "message_id": messageId})
        request: TaskSendParams = TaskSendParams(
            id=taskId,
            sessionId=sessionId,
            message=Message(
                role="user",
                parts=[TextPart(text=task)],
                metadata=metadata,
            ),
            acceptedOutputModes=["text", "text/plain"],
            # pushNotification=None,
            metadata={"conversation_id": sessionId},
        )
        task = await client.send_task(request, self.task_callback)
        # Assume completion unless a state returns that isn't complete
        state["session_active"] = task.status.state not in [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.FAILED,
            TaskState.UNKNOWN,
        ]
        if task.status.state == TaskState.INPUT_REQUIRED:
            # Force user input back
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.COMPLETED:
            # Reset active agent is task is completed
            state["active_agent"] = "None"

        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(convert_parts(task.status.message.parts, tool_context))
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(convert_parts(artifact.parts, tool_context))
        return response


def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def convert_part(part: Part, tool_context: ToolContext):
    # Currently only support text parts
    if part.type == "text":
        return part.text

    return f"Unknown type: {part.type}"


# ============================================================================ #
# STEP 15 - root_agent
# ============================================================================ #
root_agent = PurchasingAgent(
    remote_agent_addresses=[
        os.getenv("PIZZA_SELLER_AGENT_URL", "http://localhost:10000"),
        os.getenv("BURGER_SELLER_AGENT_URL", "http://localhost:10001"),
    ]
).create_agent()

# ============================================================================ #
# STEP 16 - Gradio UI
# ============================================================================ #
import gradio as gr
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from typing import AsyncIterator
from google.genai import types
from pprint import pformat

APP_NAME = "purchasing_concierge"
USER_ID = "user"
SESSION_ID = "default_session"

SESSION_SERVICE = InMemorySessionService()
PURCHASING_AGENT_RUNNER = Runner(
    session_service=SESSION_SERVICE,
    agent=root_agent,
    app_name=APP_NAME,
)


async def get_response_from_agent(
    message: str,
    history: List[Dict[str, Any]],
) -> AsyncIterator[str]:
    await SESSION_SERVICE.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    content = types.Content(role="user", parts=[types.Part(text=message)])
    events_iterator: AsyncIterator[Event] = PURCHASING_AGENT_RUNNER.run_async(
        session_id=SESSION_ID,
        user_id=USER_ID,
        new_message=content,
    )

    responses = []
    async for event in events_iterator:  # event has type Event
        if event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    formatted_call = f"```python\n{pformat(part.function_call.model_dump(), indent=2, width=80)}\n```"
                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=f"{part.function_call.name}:\n{formatted_call}",
                            metadata={"title": "🛠️ Tool Call"},
                        )
                    )
                elif part.function_response:
                    formatted_response = f"```python\n{pformat(part.function_response.model_dump(), indent=2, width=80)}\n```"

                    responses.append(
                        gr.ChatMessage(
                            role="assistant",
                            content=formatted_response,
                            metadata={"title": "⚡ Tool Response"},
                        )
                    )

        # Key Concept: is_final_response() marks the concluding message for the turn
        if event.is_final_response():
            if event.content and event.content.parts:
                # Extract text from the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                # Handle potential errors/escalations
                final_response_text = (
                    f"Agent escalated: {event.error_message or 'No specific message.'}"
                )
            responses.append(
                gr.ChatMessage(role="assistant", content=final_response_text)
            )
            yield responses
            break  # Stop processing events once the final response is found

        yield responses


if __name__ == "__main__":
    import os
    from IPython.display import IFrame, display

    hostname = os.environ.get("HOSTNAME")

    demo = gr.ChatInterface(
        get_response_from_agent,
        title="Purchasing Concierge",
        description="This assistant can help you to purchase food from remote sellers.",
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=8087,
        share=False,
        inline=False,
        root_path=f"/{hostname}/proxy/8087",
    )

    display(IFrame(src=f"/{hostname}/proxy/8087/", width="100%", height=600))
