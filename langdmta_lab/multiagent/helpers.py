from langdmta_lab.base.base_state import State


def route_from_supervisor(state: State) -> str:
    return state["next"]
