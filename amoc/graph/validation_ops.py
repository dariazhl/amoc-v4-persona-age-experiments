from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from amoc.graph.graph import Graph


class ValidationOps:
    FORBIDDEN_EDGE_LABELS = {"agent_of", "target_of", "patient_of", "relation"}

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def validate_amocv4_constraints(self) -> list[str]:
        violations = []
        for edge in self._graph.edges:
            if edge.label in self.FORBIDDEN_EDGE_LABELS:
                violations.append(
                    f"AMoCv4: Forbidden edge label '{edge.label}': "
                    f"{edge.source_node.get_text_representer()} -> {edge.dest_node.get_text_representer()}"
                )
        return violations

    def sanity_check_readable_triplets(self) -> list[str]:
        violations = []
        for edge in self._graph.edges:
            subj = edge.source_node.get_text_representer()
            verb = edge.label
            obj = edge.dest_node.get_text_representer()

            if not subj or not verb or not obj:
                violations.append(
                    f"AMoCv4: Edge has empty component: '{subj}' --{verb}--> '{obj}'"
                )

            if verb in self.FORBIDDEN_EDGE_LABELS:
                violations.append(
                    f"AMoCv4: Edge uses forbidden label '{verb}': '{subj}' --{verb}--> '{obj}'"
                )

        return violations

    def validate_event_mediation_invariant(self) -> list[str]:
        from amoc.graph.node import NodeType

        violations = []
        for edge in self._graph.edges:
            is_event_involved = (
                edge.source_node.node_type == NodeType.EVENT
                or edge.dest_node.node_type == NodeType.EVENT
            )
            if is_event_involved:
                if not edge.label or edge.label.strip() == "":
                    violations.append(
                        f"EVENT edge has empty label: "
                        f"{edge.source_node.get_text_representer()} --> "
                        f"{edge.dest_node.get_text_representer()}"
                    )

        return violations

    def collect_all_violations(self) -> list[str]:
        violations = []
        violations.extend(self.validate_amocv4_constraints())
        violations.extend(self.sanity_check_readable_triplets())
        violations.extend(self.validate_event_mediation_invariant())
        return violations
