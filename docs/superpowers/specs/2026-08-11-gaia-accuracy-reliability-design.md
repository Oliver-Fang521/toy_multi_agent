# GAIA Agent Accuracy and Reliability Improvement Design

## Goal

Improve end-to-end GAIA accuracy by preventing a useful task attempt from being
discarded because one internal stage returned malformed structured output. The
change must preserve GAIA's official final-answer scoring and the existing
attachment security boundary.

## Observed failure modes

The previous 15-task runs showed that most failures happened before answer
submission: planner/specialist/critic responses could not be parsed, whole-task
retries repeated paid work, and visual work could be routed to a provider that
cannot inspect images. The internal evidence gate also treated an approved
answer as a failure when evidence was thinner than the benchmark requires.

## Chosen approach

Use a small reliability layer at the three structured-output boundaries:

1. **Stage-local repair.** Planner, specialist, and critic validate their first
   response. If it is missing or invalid, they make at most one inexpensive
   repair call using the same schema, no tools, and an instruction to return
   only the required object. This keeps the retry local instead of rerunning
   the complete Plan → Act → Verify workflow.
2. **Deterministic planner fallback.** If both planner attempts fail, construct a
   conservative plan from the question and attachment metadata. The fallback
   never invents an answer; it only selects a safe lookup, computation, file,
   or visual route. It is observable in the run metadata.
3. **Provider capability routing.** Providers expose whether they support image
   input. Visual specialists are not selected for a provider that cannot see the
   attachment; the deterministic fallback chooses a text/file route when that
   is safe, otherwise the task fails early with a clear attachment error before
   spending more model calls.
4. **Configurable quality policy.** `benchmark` mode (the default for scoring)
   accepts a critic approval and valid candidate under the same GAIA answer
   scorer. `strict` mode retains the stronger internal evidence requirement for
   audit-oriented runs. Neither mode changes the official exact-match scorer.

## Non-goals

- Do not loosen path validation, symlink checks, file-size limits, or MIME
  handling.
- Do not fabricate evidence or turn a missing attachment into a model guess.
- Do not change GAIA's `model_answer` format or exact-match normalization.
- Do not add a second full benchmark run until unit and integration tests show
  the repair paths are working.

## Compatibility and observability

Repair attempts and fallback routes are recorded as stage metadata and retain
the existing error categories. The persisted run configuration includes the
quality policy and provider vision capability so results remain comparable.

## Verification strategy

Tests cover: one-call repair success, deterministic fallback after two invalid
responses, provider vision capability, benchmark versus strict quality policy,
and the unchanged attachment security checks. Existing full tests, Ruff, and
package build must pass before an expensive GAIA evaluation is considered.
