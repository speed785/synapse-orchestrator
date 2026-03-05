from synapse.dependency_analyzer import DependencyAnalyzer, ToolCall  # pyright: ignore[reportImplicitRelativeImport]


def test_analyze_valid_dag_with_implicit_and_explicit_dependencies() -> None:
    analyzer = DependencyAnalyzer()
    calls = [
        ToolCall(id="fetch.user", name="fetch_user"),
        ToolCall(id="fetch.orders", name="fetch_orders"),
        ToolCall(
            id="build",
            name="build_payload",
            inputs={
                "user": "$results.fetch.user",
                "orders": "$results.fetch.orders",
                "email": "$results.fetch.user.email",
            },
            depends_on=["fetch.user"],
        ),
    ]

    graph = analyzer.analyze(calls)

    assert graph.dependencies_of("fetch.user") == set()
    assert graph.dependencies_of("fetch.orders") == set()
    assert graph.dependencies_of("build") == {"fetch.user", "fetch.orders"}
    assert graph.dependents_of("fetch.user") == {"build"}
    assert graph.is_acyclic() is True


def test_analyze_rejects_unknown_dependency() -> None:
    analyzer = DependencyAnalyzer()
    calls = [
        ToolCall(id="a", name="tool_a", inputs={"x": "$results.missing"}),
    ]

    try:
        analyzer.analyze(calls)
        assert False, "Expected ValueError for unknown dependency"
    except ValueError as exc:
        assert "unknown tool ids" in str(exc)


def test_analyze_rejects_cycles() -> None:
    analyzer = DependencyAnalyzer()
    calls = [
        ToolCall(id="a", name="tool_a", inputs={"x": "$results.b"}),
        ToolCall(id="b", name="tool_b", inputs={"x": "$results.a"}),
    ]

    try:
        analyzer.analyze(calls)
        assert False, "Expected ValueError for cycle"
    except ValueError as exc:
        assert "Circular dependency" in str(exc)


def test_analyze_ignores_self_dependency_and_handles_empty_inputs() -> None:
    analyzer = DependencyAnalyzer()
    calls = [
        ToolCall(id="self", name="tool_self", depends_on=["self"]),
        ToolCall(id="next", name="tool_next", inputs={}),
    ]

    graph = analyzer.analyze(calls)

    assert graph.dependencies_of("self") == set()
    assert graph.dependencies_of("next") == set()
    assert set(graph.roots()) == {"self", "next"}


def test_analyze_extracts_references_from_nested_lists_and_tuples() -> None:
    analyzer = DependencyAnalyzer()
    calls = [
        ToolCall(id="a", name="tool_a"),
        ToolCall(id="b", name="tool_b"),
        ToolCall(
            id="join",
            name="join_tool",
            inputs={
                "items": ["$results.a.value", ("$results.b", "x")],
            },
        ),
    ]

    graph = analyzer.analyze(calls)

    assert graph.dependencies_of("join") == {"a", "b"}
