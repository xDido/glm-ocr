---
name: Tightly scope exploration from user-provided anchors
description: User prefers narrow, verification-oriented exploration when they've already done the analysis — not broad repo-wide sweeps
type: feedback
originSessionId: 4f68fa82-22d5-414d-8646-1b145bb7159f
---
When the user asks a feasibility/design question and supplies their own analysis with specific file paths, line numbers, or component names (e.g. "the `.detach().cpu().numpy()` hop in runtime_app.py:505"), treat those as anchors — read/grep from there outward to verify or refute specific claims. Do not spawn broad Explore agents that re-catalogue the whole area.

**Why:** In the torch→ONNX feasibility study (2026-04-22), I launched two broad Explore agents to "catalogue all remaining torch usage" and "map pipeline + env + perf context." The user interrupted both and re-scoped the question with their own pre-digested analysis naming exact files, lines, and expected gains. The broad sweep would have duplicated work the user had already done.

**How to apply:**
- If the user gives file paths / line numbers / named functions in the prompt, open *those* first with Read/Grep — don't send a subagent to re-discover them.
- Use subagents when the question is genuinely open-ended ("where would X live?"), not when the user has already identified the target.
- Feasibility studies: verify the user's specific claims, resolve their named open questions, and push back on the one or two numbers that look wrong. Skip the preamble of re-establishing what they already stated.
