# API Indexing V2 Consolidated Plan

This is the canonical improvement plan for API Indexing V2.

It consolidates:
- `potential_improvements.md` (13-item detailed list)
- `potential_improvements_clean.md` (9-item grouped list)

Use the initiative IDs in this document (`V2-01` ... `V2-09`) for planning, tickets, and status tracking.

---

## Goals

1. Increase routine quality and reduce failed iteration loops.
2. Improve knowledge reuse across experiments (avoid rediscovery).
3. Reduce token waste and context bloat in PI/Inspector loops.
4. Improve debuggability, replayability, and operational visibility.

---

## Consolidated Initiatives

| ID | Initiative | Impact | Effort | Depends On |
|---|---|---|---|---|
| V2-01 | Tool Usage Observability | Medium | Low | - |
| V2-02 | PI Execution Visibility | High | Low | - |
| V2-03 | Structured Experiment Output Schemas | High | Medium | V2-01 |
| V2-04 | Proven Artifacts Redesign | High | High | V2-03 |
| V2-05 | Inspector Context Management | Medium | Low | V2-02 |
| V2-06 | Anti-Bot as First-Class Exploration Output | Medium | Low | - |
| V2-07 | Deferred Routine Planning + Scouting Pre-Phase | High | Medium | V2-06 |
| V2-08 | Pipeline Resume + Agent Thread Replay | Medium | High | V2-01 |
| V2-09 | Visual Page Understanding (Screenshots/OCR) | Medium | Medium | - |

---

## Sequencing (Implementation Waves)

### Wave 1 (Immediate, Low Effort / High Leverage)
- V2-01 Tool Usage Observability
- V2-02 PI Execution Visibility
- V2-05 Inspector Context Management
- V2-06 Anti-Bot Output

### Wave 2 (Core Data Model and Knowledge Reuse)
- V2-03 Structured Experiment Output Schemas
- V2-04 Proven Artifacts Redesign

### Wave 3 (Planning and Execution Strategy)
- V2-07 Deferred Planning + Scouting

### Wave 4 (Platform Capability Extensions)
- V2-08 Resume + Replay
- V2-09 Visual Understanding

---

## Initiative Specs

## V2-01: Tool Usage Observability

### Scope
- Track per-agent:
  - tools registered
  - tools called
  - call counts
  - never-called tools
- Emit per-run aggregate report.

### Delivery
- Add counters in `AbstractAgent._execute_tool`.
- Persist per-agent summary with thread dumps.
- Write pipeline-level JSON report in output dir.

### Acceptance Criteria
- Every run emits `tool_usage.json` in pipeline output.
- Report includes all agent types participating in the run.
- Report includes registered vs called counts for each tool.

---

## V2-02: PI Execution Visibility

### Scope
- Improve PI visibility after `submit_routine`:
  - include operation-level metadata in response
  - expose full attempt details on demand

### Delivery
- Add `operations_summary` to `submit_routine` response.
- Include `operations_metadata` in returned payload.
- Add `get_attempt_details(attempt_id)` PI tool, reading persisted attempt records.

### Acceptance Criteria
- PI can identify failing operation index/type without reading raw logs.
- `submit_routine` response includes per-operation timing + status.
- PI can fetch full attempt record by ID in one tool call.

---

## V2-03: Structured Experiment Output Schemas

### Scope
- Make schema-driven worker outputs the default for common experiment types.

### Delivery
- Define canonical schemas:
  - endpoint discovery
  - auth test
  - token tracing
  - navigation/DOM probe
- Add PI guidance and dispatch rules for selecting schema.
- Keep freeform fallback for novel tasks.

### Acceptance Criteria
- At least 80% of experiments in a run include non-null `output_schema`.
- Structured fields (`confirmed`, `endpoint_url`, `method`, auth fields) present for schema-based runs.
- PI can compare same-type experiments programmatically by key fields.

---

## V2-04: Proven Artifacts Redesign

### Scope
- Turn `ProvenArtifacts` into a reliable, typed knowledge layer.

### Delivery
- Replace `list[dict[str, Any]]` artifacts with typed models.
- Extend worker output schema with optional `discovered_artifacts`.
- Auto-upsert artifacts from confirmed experiment outputs.
- Inject proven artifacts into worker dispatch context.
- Enforce PI artifact handling after confirmed experiments (record or explicit skip reason).

### Acceptance Criteria
- Confirmed experiments produce artifact deltas in ledger for supported schema types.
- Workers receive compact proven artifact context at dispatch.
- Runs no longer show near-zero proven fetch/token artifacts when routines ship successfully.

---

## V2-05: Inspector Context Management

### Scope
- Improve inspector signal-to-noise and reduce truncation risk.

### Delivery
- Pass network summary only by default (or compact site facts block).
- Apply smart truncation to execution `data` field only.
- Add compact previous-attempt history summary to inspector prompt.

### Acceptance Criteria
- Inspector prompt size drops materially on large payload attempts.
- No blind prompt truncation of critical context blocks.
- Inspector output references prior attempt deltas when applicable.

---

## V2-06: Anti-Bot as First-Class Output

### Scope
- Surface anti-bot observations from exploration to PI planning/dispatch.

### Delivery
- Add `anti_bot_observations` to `NetworkExplorationSummary`.
- Inject this section prominently into PI prompt context.

### Acceptance Criteria
- Exploration outputs include structured anti-bot observations when detected.
- PI experiment prompts mention anti-bot constraints for affected sites.

---

## V2-07: Deferred Planning + Scouting Pre-Phase

### Scope
- Shift PI from strict upfront full planning to progressive planning with early scouting.

### Delivery
- Add workflow guidance for:
  - initial scouting experiments
  - incremental `plan_routines` updates
  - broader routine archetypes (non-fetch-first)

### Acceptance Criteria
- PI performs scouting before locking full routine set on non-trivial sites.
- New/updated routine specs appear after early experiment evidence.
- Increase in non-API routine patterns where appropriate (DOM/hybrid/nav+fetch).

---

## V2-08: Pipeline Resume + Agent Thread Replay

### Scope
- Resume partial runs and support post-hoc interactive replay.

### Delivery
- Add `--resume` mode in API indexing runner.
- Detect partial state from output dir and continue PI loop.
- Add thread reload path for PI/worker/inspector replay.
- Add inspector re-run from saved attempt records.

### Acceptance Criteria
- Interrupted run can resume from output directory without re-running exploration.
- Agent thread file can be loaded and continued interactively.
- Inspector can be rerun on saved attempt records without re-executing routines.

---

## V2-09: Visual Page Understanding (Screenshots/OCR)

### Scope
- Give workers visual state awareness beyond DOM/JS introspection.

### Delivery
- Add `browser_screenshot` tool.
- Add configurable auto-capture modes (`manual`, `after_navigation`, `every_action`).
- Add optional OCR extraction pipeline.

### Acceptance Criteria
- Workers can retrieve screenshot and reason about visible UI state.
- Failures caused by visual blockers (modals/challenges) are identified explicitly.
- OCR path is feature-flagged and dependency-guarded.

### Dependency Note
- OCR dependencies (`easyocr`, `opencv`) are not currently in project deps; rollout must include packaging/runtime decision.

---

## Crosswalk from Existing Docs

| Consolidated ID | `potential_improvements.md` | `potential_improvements_clean.md` |
|---|---|---|
| V2-01 | #11 Tool Usage Observability | #2 Tool Usage Observability |
| V2-02 | #13 PI Execution Visibility | #8 PI Execution Visibility |
| V2-03 | #12 Structured Experiment Output Schemas | #5 Structured Experiment Output Schemas |
| V2-04 | #4 Workers Access Proven Artifacts + #10 Proven Artifacts Redesign | #6 Proven Artifacts Redesign |
| V2-05 | #3, #7, #8, #9 Inspector Context Issues | #7 Smarter Inspector Context Management |
| V2-06 | #2 Anti-Bot Output | #3 Anti-Bot Output |
| V2-07 | #5 Deferred Planning | #4 Deferred Planning |
| V2-08 | #1 Resume + Replay | #1 Resume + Replay |
| V2-09 | #6 Visual Understanding | #9 Visual Understanding |

---

## Metrics to Track (Minimum Set)

- Experiment schema adoption rate (`structured_experiments / total_experiments`).
- Proven artifact growth rate per run (fetch/token/nav/param counts).
- PI iteration efficiency:
  - average attempts per shipped routine
  - submit→ship cycle count
- Inspector efficiency:
  - average prompt size
  - truncation events
- Tool utilization:
  - registered vs called tool ratio by agent type.

---

## Execution Notes

- Treat this doc as the only planning source for V2 improvements.
- Keep the older two docs as reference context/history.
- When creating issues/PRs, include `V2-XX` ID in title.
