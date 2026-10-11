from backend._log import fmt_args
from langgraph.graph import StateGraph,START,END

from backend.agents.checkpointing import create_checkpointer

from backend.agents.advisor import advisor_node
from backend.agents.clinic import clinic_node
from backend.agents.handoff import handoff_destinations
from backend.agents.insurance import insurance_node
from backend.agents.pharmacy import pharmacy_node

from backend.agents.report import report_node
from backend.agents.router import router_node
from backend.agents.state import MainAgentState

_AGENT_MAP={
    "clinic_agent": "clinic_node",
    "insurance_agent": "insurance_node",
    "report_agent": "report_node",
    "advisor_agent": "advisor_node",
    "pharmacy_agent": "pharmacy_node",
}

def _route_to_agent(state: MainAgentState) -> str:
    print(f"--- [graph] _route_to_agent({fmt_args(state=state)}) ---", flush=True)
    agent_id=state.get("next_agent","advisor_agent")
    return _AGENT_MAP.get(agent_id,"advisor_node")

def build_graph(checkpointer=None):
    print(f"--- [graph] build_graph({fmt_args(checkpointer=checkpointer)}) ---", flush=True)
    workflow=StateGraph(MainAgentState)

    workflow.add_node("router",router_node)
    workflow.add_node(
        "clinic_node",
        clinic_node,
        destinations=(*handoff_destinations("clinic_agent"),END),
    )
    workflow.add_node(
        "insurance_node",
        insurance_node,
        destinations=(*handoff_destinations("insurance_agent"),END),
    )
    workflow.add_node(
        "report_node",
        report_node,
        destinations=(*handoff_destinations("report_agent"),END),
    )
    workflow.add_node(
        "advisor_node",
        advisor_node,
        destinations=(*handoff_destinations("advisor_agent"),END),
    )
    workflow.add_node(
        "pharmacy_node",
        pharmacy_node,
        destinations=(*handoff_destinations("pharmacy_agent"),END),
    )

    workflow.add_edge(START,"router")
    workflow.add_conditional_edges(
        "router",
        _route_to_agent,
        {
            "clinic_node": "clinic_node",
            "insurance_node": "insurance_node",
            "report_node": "report_node",
            "advisor_node": "advisor_node",
            "pharmacy_node": "pharmacy_node",
        }
    )

    workflow.add_edge("clinic_node",END)
    workflow.add_edge("insurance_node",END)
    workflow.add_edge("report_node",END)
    workflow.add_edge("advisor_node",END)
    workflow.add_edge("pharmacy_node",END)

    return workflow.compile(checkpointer=checkpointer or create_checkpointer())


master_app=build_graph()

