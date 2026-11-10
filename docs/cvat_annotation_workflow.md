# CVAT Annotation Workflow

**Status:** Initial workflow contract; the concrete CVAT export adapter will be implemented after inspecting the first project export.

## Role of CVAT

CVAT is the annotation workspace and human-review interface. Its native project/task exports should be preserved as source artifacts. ChickenBehaviorLab converts those exports into the project's internal annotation schema; model code should not depend directly on a particular CVAT export layout.

Do not overwrite the original media or native CVAT exports during conversion.

## Annotation levels

For behavior recognition, annotations need to preserve two different concepts:

1. **Individual identity:** a persistent track identifier for one chicken within one video.
2. **Behavior interval:** a behavior label attached to that individual over a bounded time interval.

A chicken may perform several behaviors during one video. Therefore, a behavior label should represent a time-varying interval, not be treated as a permanent class for the whole chicken track. The same identity should be retained when behavior changes.

Detection boxes, pose keypoints, and behavior intervals are distinct annotation levels. They may be exported together, but the importer should map them into separate internal structures rather than mixing their meanings.

## Internal annotation contract

The current internal JSON contract is version 1.0 and is video-scoped:

- video_id belongs to the annotation document (AnnotationSet).
- Each BehaviorAnnotation contains track_id, behavior_id, start_frame, and end_frame.
- Optional per-annotation fields are annotation_id, annotator_id, confidence, and attributes.
- Frame intervals are inclusive at both ends.
- Track IDs are normalized to a canonical string when data enter the processing/dataset pipeline. Legacy numeric IDs are accepted at the boundary and converted there.
- Track IDs are only unique within their video context. When creating split groups or cross-video summaries, the stable identity is the pair (video_id, track_id).

Example document:

    {
      "schema_version": "1.0",
      "video_id": "video_001",
      "annotations": [
        {
          "annotation_id": "ann_000001",
          "track_id": "track_1",
          "behavior_id": "feeding",
          "start_frame": 100,
          "end_frame": 180,
          "annotator_id": "annotator_01",
          "confidence": 0.95,
          "attributes": {}
        },
        {
          "annotation_id": "ann_000002",
          "track_id": "track_1",
          "behavior_id": "walking",
          "start_frame": 181,
          "end_frame": 230
        }
      ],
      "metadata": {}
    }

The example illustrates the internal target format; it does not prescribe how these fields must be represented inside CVAT.

## Import boundary

The future CVAT adapter should:

1. Read the native CVAT export without modifying it.
2. Resolve the associated video/image asset and its stable project video_id.
3. Extract persistent object/track identity separately from behavior labels.
4. Convert behavior tags/attributes/segments into inclusive frame intervals.
5. Preserve source annotation IDs, annotator information, confidence (when available), and extra attributes.
6. Validate frame bounds against the decoded asset and report any unmapped labels or ambiguous identities.
7. Write a versioned internal AnnotationSet JSON file and a conversion report.

The adapter must not silently invent a behavior label, infer a missing track identity, or shift frame numbers to compensate for a media conversion. If a conversion changes frame rate or frame count, an explicit frame mapping is required.

## Media handling

Keep the supplied videos and any later originals in a read-only raw/ area. If a source container or codec is not convenient for annotation or model processing, create a derivative in processed/ and record at least:

- source asset ID and file hash;
- derivative file name and file hash;
- decoder/encoder and conversion settings;
- original and resulting frame counts;
- original and resulting frame rates and dimensions;
- any frame/timestamp mapping applied.

Do not assume that a transcode preserves frame numbering or timing. Check synchronization before using CVAT frame indices as model frame indices.

## Still images

Still images can be useful for detection/pose labeling, but the current behavior schema is organized around video-scoped track intervals. The image export representation and its mapping into the dataset will be specified separately after reviewing the actual CVAT export. Do not convert an image collection into fake video tracks without an explicit mapping rule.

## First export review checklist

Before implementing the importer, inspect one small CVAT export and confirm:

- export format and version;
- whether the task is image- or video-based;
- where object IDs and track IDs are stored;
- how behavior labels are represented (track labels, shape attributes, tags, or segments);
- whether labels can change over time for one tracked chicken;
- frame indexing convention and handling of outside/occluded frames;
- how source media names and dimensions are referenced.

Only then should the importer be implemented and tested against the exact export structure.
