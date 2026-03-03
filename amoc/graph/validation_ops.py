from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from amoc.graph.graph import Graph


class ValidationOps:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

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

        return violations

    def collect_all_violations(self) -> list[str]:
        violations = []
        violations.extend(self.validate_amocv4_constraints())
        violations.extend(self.sanity_check_readable_triplets())
        return violations
