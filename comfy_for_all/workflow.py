from pydantic import BaseModel, Field


class Node(BaseModel): ...


class Workflow(BaseModel):
    nodes: dict[str, Node]
