from synapse.dependency_analyzer import DependencyAnalyzer, ToolCall
from synapse.planner import Planner


def test_planner_builds_expected_stages() -> None:
    calls = [
        ToolCall(id="a", name="tool_a"),
        ToolCall(id="b", name="tool_b"),
        ToolCall(id="c", name="tool_c", inputs={"x": "$results.a", "y": "$results.b"}),
        ToolCall(id="d", name="tool_d", inputs={"x": "$results.c"}),
    ]

    graph = DependencyAnalyzer().analyze(calls)
    plan = Planner().plan(graph)

    assert len(plan.stages) == 3
    assert {c.id for c in plan.stages[0].calls} == {"a", "b"}
    assert [c.id for c in plan.stages[1].calls] == ["c"]
    assert [c.id for c in plan.stages[2].calls] == ["d"]
    assert plan.total_calls == 4


def test_planner_computes_critical_path_for_branching_graph() -> None:
    calls = [
        ToolCall(id="root", name="root"),
        ToolCall(id="left", name="left", inputs={"x": "$results.root"}),
        ToolCall(id="right", name="right", inputs={"x": "$results.root"}),
        ToolCall(id="tail", name="tail", inputs={"x": "$results.left"}),
    ]
    graph = DependencyAnalyzer().analyze(calls)
    plan = Planner().plan(graph)

    assert plan.critical_path == ["root", "left", "tail"]


def test_planner_handles_empty_graph() -> None:
    graph = DependencyAnalyzer().analyze([])
    plan = Planner().plan(graph)

    assert plan.stages == []
    assert plan.total_calls == 0
    assert plan.parallelism == 0.0
    assert plan.critical_path == []
