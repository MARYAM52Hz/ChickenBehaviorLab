# Pipeline Orchestration

## Scope

`ChickenBehaviorPipeline` coordinates already-implemented components. It does not implement detection, tracking, pose estimation, skeleton construction, behavior classification, event extraction, or analytics itself.

## Expected stage flow

1. `detection`
2. `tracking`
3. `pose`
4. `skeleton`
5. `behavior`
6. `events`
7. `analytics`

Stage names are conventions. Each stage receives a `PipelineContext`, reads prior results through `context.get_output(name)`, and returns its own output. Dependencies should be declared in `PipelineStage.requires`.

## Adapter boundary

Existing classes may have different method signatures. Create small adapter functions that translate the orchestration contract into the actual component API. Keep video decoding/frame iteration in the video-ingestion layer if the current detector expects individual frames rather than a video path.

## Error behavior

If a stage raises an exception, the run stops and raises `PipelineStageError`. The original exception remains available as `error.__cause__`. Stage timings are returned only for successful runs; failure-stage timing is logged but no partial `PipelineResult` is returned.

## Running tests

From the repository root, with `src` on the Python import path or the package installed:

```bash
python -m unittest discover -s tests/pipeline -v
python -m compileall -q src tests
```

The tests use fake components and do not validate model accuracy, video decoding, real detector weights, or compatibility with concrete project classes.
