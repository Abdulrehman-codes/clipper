"""LangGraph nodes (§3). One file per stage; each is a `ClipState -> dict` fn.

Nodes are thin: they orchestrate state and delegate the real work to
clipper.video / clipper.youtube / clipper.llm. This keeps the graph readable and
the heavy logic independently testable (§8: 'work as plain functions first').
"""
