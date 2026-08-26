# Best 15-task GAIA evaluation

This is the best completed 15-task evaluation produced by this project as of 2026-08-25.

## Configuration

- benchmark: GAIA 2023 validation split
- dataset revision: `682dd723ee1e1697e00360edccf2366dc8418dd9`
- model: `gpt-5.6-luna`
- reasoning effort: `max`
- sample: 15 tasks selected with seed 42, balanced as 5 tasks per difficulty level
- votes: 1
- limits: 8 plan steps, 24 logical tool calls, 500 seconds per task
- stage deadlines: Planner 30 seconds, Specialists 410 seconds, Critic 60 seconds

## Result

| Metric | Result |
| --- | ---: |
| Overall | **46.67% (7/15)** |
| Level 1 | **100.00% (5/5)** |
| Level 2 | **40.00% (2/5)** |
| Level 3 | **0.00% (0/5)** |
| Critic approvals | 8 |
| Malformed-response failures | 5 |
| Timeout failures | 2 |
| Total duration | 3,515.613 seconds |
| Median task duration | 208.900 seconds |
| p95 task duration | 500.011 seconds |
| Total tokens | 959,469 |

The source run's public artifacts were independently reloaded, schema-checked, privacy-checked, and
recomputed by `gaia-agent verify-results`. The SHA-256 of the original generated `summary.json` was
`988792c05fc073b6a7b9376feda779fd005da94c75d514a5e21d6d125e9c612e`.

## Scope and privacy

This is a deterministic project evaluation, not a full GAIA leaderboard score. The aggregate
artifact intentionally omits gated task text, reference answers, task-level predictions,
attachments, prompts, raw model/tool output, credentials, and hidden reasoning.
