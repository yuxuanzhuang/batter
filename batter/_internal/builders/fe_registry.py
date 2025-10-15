from typing import Callable, Dict

BUILD_COMPLEX_REGISTRY: Dict[str, Callable[["AlchemicalFEBuilder"], bool]] = {}

def register_build_complex(component: str):
    def deco(fn: Callable[["AlchemicalFEBuilder"], bool]):
        BUILD_COMPLEX_REGISTRY[component] = fn
        return fn
    return deco

RESTRAINT_REGISTRY: Dict[str, Callable[["BaseBuilder", dict], None]] = {}

def register_restraints(*components: str):
    """Decorator: register a function to build restraints for comp(s)."""
    def deco(fn: Callable[["BaseBuilder", dict], None]):
        for c in components:
            RESTRAINT_REGISTRY[c] = fn
        return fn
    return deco

SIM_FILES_REGISTRY: Dict[str, Callable[["BaseBuilder"], None]] = {}

def register_sim_files(component: str):
    def deco(fn: Callable[["BaseBuilder"], None]):
        SIM_FILES_REGISTRY[component] = fn
        return fn
    return deco