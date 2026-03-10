# Round 2 Labeling Strategy (EN)

## Purpose

This document defines the approved Round 2 labeling logic for the action taxonomy, data filtering, deduplication, ambiguity handling, and batch assignment.

The goal is to improve label quality and usable dataset coverage without breaking continuity with the existing validated set.

## 1. Approved Label Space

Round 2 keeps the same 5 primary action classes:

1. `Object Transfer`
2. `Task Operation` (new display name for the former `Essential Operation`)
3. `Stationary`
4. `Locomotion`
5. `Search`

`Error` is no longer treated as an action class. If a row is invalid, annotators should use `Delete Row`.

## 2. Definitions and Priority Rules

### 2.1 Object Transfer

Choose `Object Transfer` when the main outcome is a change in:

1. object position
2. possession / who is holding it
3. contact state
4. open / closed state

Examples:

1. pick up / put down
2. hand over
3. carry an object
4. move an object
5. open / close a door, drawer, fridge, or tap

### 2.2 Task Operation

Choose `Task Operation` when the main outcome is performing a task on an object or tool, rather than making transfer/state change the dominant result.

Examples:

1. cut
2. stir
3. type
4. write
5. scrub
6. wash
7. operate a tool or device

### 2.3 Search

Choose `Search` when the primary behavior is active visual scanning, checking, or looking for something.

### 2.4 Locomotion

Choose `Locomotion` when movement through space is the main meaning of the narration.

### 2.5 Stationary

Choose `Stationary` when none of the stronger action meanings above applies and the activity is mainly low-motion, holding, waiting, resting, talking, or otherwise non-displacing.

### 2.6 Priority Order

Annotators should use this decision order:

1. First ask: does `Object Transfer` clearly apply?
2. If not, ask: does `Task Operation` clearly apply?
3. If not, ask: is it `Search`?
4. If not, ask: is the main meaning `Locomotion`?
5. Otherwise use `Stationary`

This rule is important:

- If transfer or state change is the dominant outcome, choose `Object Transfer` even if a task component is also present.

That means:

1. `Object Transfer` is intentionally prioritized over `Task Operation` when transfer is the main result.
2. `Locomotion` should not replace `Object Transfer` just because the person is moving while doing it.

## 3. Secondary Label Policy

The secondary label is optional and should be rare.

It is used only when:

1. the main LLM label is acceptable (`Gold`), and
2. another class is also clearly valid at the same time

Allowed secondary options are intentionally restricted to:

1. `Task Operation`
2. `Stationary`

Not allowed as secondary in the standard workflow:

1. `Object Transfer`
2. `Locomotion`
3. `Search`

Reasoning:

1. If `Object Transfer` is truly valid as the dominant action, it should be the main label, not the secondary one.
2. `Locomotion` is usually co-occurring context, not the competing semantic main action.
3. `Search` should remain a direct primary decision, not a routinely dual-labeled fallback.

Typical use cases:

1. Main = `Object Transfer`, secondary = `Task Operation`
   - when transfer is primary, but the task component is still clearly present
2. Main = `Task Operation`, secondary = `Stationary`
   - when a low-motion / hold-like state is also clearly valid

## 4. Out-of-Scope Domain Exclusion

Rows that are outside the assistive-technology target domain should be removed from the main training and testing sets.

### 4.1 Exclude These Domain Families

1. Ball sports and sports-like play
   - dribbling the ball
   - shooting the ball
   - passing the ball
   - kicking the ball
   - catching the ball in a sports/game context
2. Competitive or recreational sports actions
   - goal / hoop / court / racket / net / score contexts
3. Workout-only athletic drills
   - exercise actions that are not relevant to daily assistive-technology use cases

### 4.2 Important Filtering Rule

Do not exclude by verb alone.

Examples:

1. `throw` in cleaning or cooking can still be in-scope
   - e.g. throwing food into a bin should stay
2. `serve` can refer to a kitchen tool, not sports
3. `catch` can be valid if it is catching a normal household item and not sports play

So exclusion must be based on:

1. narration text
2. object mention
3. activity context

It should not be based only on a single keyword.

### 4.3 Scenario-Level Note

The current dataset does not contain a dedicated `sports` scenario field. Sports-like rows are mixed into existing scenarios, so exclusion must happen at the narration level, not by dropping an entire scenario column value.

## 5. Deduplication and Conflict Detection

Use the approved `No Articles` normalization:

1. lowercase
2. remove hashtags
3. remove punctuation
4. remove articles (`a`, `an`, `the`, `some`)
5. trim whitespace

But the planning key must be:

- `(normalized_narration, action)`

This is required because narration-only grouping can incorrectly merge rows that still map to different action labels.

Round 2 should explicitly separate:

1. safe duplicates
   - same `(normalized_narration, action)`
2. conflict duplicates
   - same `normalized_narration`, but different actions

Only the first category is safe for direct propagation / planning.

## 6. Ambiguity Handling

Not all conflicts should be removed in the same way.

Recommended handling:

1. High-agreement conflict groups
   - keep in the main eligible pool
2. Medium-agreement conflict groups
   - move to an `adjudication bucket`
3. Low-agreement conflict groups
   - remove from the main training / testing set and from normal Round 2 dispatch

The adjudication bucket should include:

1. narration groups with multi-label conflicts
2. high-risk narration families such as `moves`, `looks`, `opens`, `closes`, `holds`
3. rows that repeatedly changed labels in the previous validation round
4. low-confidence second-pass LLM outputs

## 7. Round 2 Pool Construction

Build the Round 2 eligible pool in this order:

1. remove out-of-scope domain rows
2. apply normalization
3. deduplicate using `(normalized_narration, action)`
4. remove already-validated groups
5. remove low-agreement ambiguity groups
6. move medium-agreement groups into a separate adjudication bucket
7. keep the remaining clean groups as the main eligible pool

This creates three distinct outputs:

1. main eligible pool
2. adjudication bucket
3. out-of-scope exclusions

## 8. Batch Assignment Logic

Round 2 main assignment should follow a coverage-first strategy.

### 8.1 Main Principle

Prioritize groups that cover the largest number of original rows, so a limited amount of validation yields the largest propagated dataset coverage.

### 8.2 Assignment Target

1. `12` batches
2. `1000` groups per batch
3. `12,000` total main-batch groups

### 8.3 Assignment Rules

1. each `(normalized_narration, action)` group can appear in only one batch
2. do not duplicate the same group across annotators
3. keep the adjudication bucket separate from standard annotator batches
4. assign the main pool by descending group frequency (coverage-first)

This means the standard Round 2 batches are for high-confidence groups, while the harder conflict cases are handled separately by a smaller adjudication workflow.

## 9. Label Studio Workflow

The standard reviewer interface should stay minimal:

1. `verdict`
   - `Gold`
   - `Bad`
   - `Skip`
   - `Delete Row`
2. `corrected_action`
   - shown only when `Bad`
3. `secondary_action`
   - shown only when `Gold`
   - optional
   - limited to `Task Operation` or `Stationary`

Reference config:

[label_studio_round2_5class_review.xml](/Users/huangjunda/Desktop/MIT%202.156/HAR/doc/label_studio_round2_5class_review.xml)

## 10. Bottom Line

Round 2 should:

1. keep the 5-class structure
2. rename `Essential Operation` to `Task Operation`
3. prioritize `Object Transfer` whenever transfer/state change is the main outcome
4. exclude sports / athletic out-of-scope rows from the main train/test
5. use safe dedup with `(normalized_narration, action)`
6. remove low-agreement ambiguity groups from the main pool
7. assign `12 x 1000` main batches with a coverage-first strategy

This preserves continuity with the existing validated data while making the labeling and selection logic stricter and more application-relevant.
