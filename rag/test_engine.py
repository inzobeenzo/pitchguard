from engine import build_index, explain, eval_retrieval, retrieve

coll = build_index()

eval_retrieval(coll, [
    {"query": "does a workload spike increase injury risk?",     "expect_source": "acwr_workload_ratio"},
    {"query": "is higher fastball velocity a UCL risk factor?",  "expect_source": "workload_velocity_review"},
    {"query": "does a lowered release point predict UCL injury?","expect_source": "pitch_tracking_case_control"},
])

result = explain(coll, ["uniform color", "astrological sign"])
assert result["sources"] == [], "should have refused!"
print("refusal test passed:", result["answer"])