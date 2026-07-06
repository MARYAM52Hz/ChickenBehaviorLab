# Chicken Behavior Annotation Standard (CBAS)

**Version:** 1.0.0 (Draft)
**Status:** Active Specification

---

# 1. Introduction

The Chicken Behavior Annotation Standard (CBAS) defines the official annotation specification used throughout the ChickenBehaviorLab project.

Its primary objective is to establish a unified, reproducible, and extensible framework for representing chicken behavior from video data.

CBAS standardizes how annotations are created, stored, validated, and exchanged across datasets, experiments, and machine learning models. It provides a common foundation for tasks including chicken detection, tracking, pose estimation, behavior recognition, anomaly detection, flock-level analytics, and mortality prediction.

The standard is designed to remain independent of any specific neural network architecture or software implementation.

---

# 2. Objectives

CBAS has the following objectives:

* Establish a unified annotation framework for poultry behavior analysis.
* Ensure consistency across datasets and experiments.
* Enable reproducible scientific research.
* Support modular machine learning pipelines.
* Facilitate benchmarking between different models.
* Simplify future dataset expansion.
* Promote interoperability with external tools and research projects.

---

# 3. Scope

CBAS supports annotation for the following research tasks:

* Chicken Detection
* Multi-Object Tracking
* Pose Estimation
* Skeleton Representation
* Behavior Recognition
* Temporal Behavior Segmentation
* Behavior Event Detection
* Abnormal Behavior Detection
* Flock-Level Behavior Analysis
* Mortality Risk Prediction

Future versions may extend support to multimodal sensing, including thermal imaging, audio, environmental sensors, and wearable devices.

---

# 4. Design Principles

CBAS is based on the following principles:

* Standard-first design
* Model independence
* Modularity
* Extensibility
* Reproducibility
* Human-readable documentation
* Machine-friendly data structures
* Backward compatibility whenever possible

---

# 5. Annotation Hierarchy

CBAS organizes annotations into seven hierarchical levels.

| Level | Annotation        | Description                                                          |
| ----- | ----------------- | -------------------------------------------------------------------- |
| L1    | Detection         | Chicken localization using bounding boxes or segmentation.           |
| L2    | Tracking          | Persistent identity assignment across frames.                        |
| L3    | Pose              | Anatomical keypoints and skeleton representation.                    |
| L4    | Behavior Label    | Classification of instantaneous behavior.                            |
| L5    | Behavior Event    | Temporal representation of behaviors with start and end times.       |
| L6    | Flock Statistics  | Aggregated behavioral indicators at flock level.                     |
| L7    | Mortality Records | Mortality and health-related outcomes for validation and prediction. |

Each level builds upon the previous one while remaining modular enough to support datasets with partial annotations.

---

# 6. Dataset Organization

A CBAS-compliant dataset should clearly separate raw data, processed data, annotations, metadata, and derived outputs.

Typical dataset components include:

* Raw videos
* Processed videos
* Detection annotations
* Tracking annotations
* Pose annotations
* Skeleton sequences
* Behavior annotations
* Behavior events
* Metadata
* Mortality records

The physical directory layout is implementation-specific but should preserve this logical separation.

---

# 7. Annotation Rules

All annotations should satisfy the following requirements:

* Every annotated entity must have a unique identifier.
* Temporal consistency must be preserved across frames.
* Missing observations must be explicitly represented.
* Confidence scores should be stored whenever available.
* Annotation timestamps must remain synchronized with video frames.
* Validation procedures should detect structural inconsistencies before release.

---

# 8. Naming Conventions

CBAS adopts consistent naming conventions to improve interoperability.

Recommended practices include:

* Stable identifiers
* Lowercase file names
* Human-readable labels
* Versioned annotation files
* ISO 8601 date formats where applicable

---

# 9. Validation

A valid CBAS annotation should satisfy:

* Valid identifiers
* Consistent frame numbering
* Valid skeleton topology
* Valid behavior labels
* Valid event durations
* Metadata completeness
* Temporal consistency

Automatic validation tools are recommended before publishing datasets.

---

# 10. Versioning

CBAS follows Semantic Versioning.

* Major versions introduce structural changes.
* Minor versions introduce backward-compatible features.
* Patch versions correct errors or improve documentation.

Datasets should explicitly specify the CBAS version used during annotation.

---

# 11. Relationship to Other Standards

CBAS serves as the central specification of ChickenBehaviorLab.

It is complemented by:

* Skeleton Standard
* Behavior Standard
* Behavior Event Standard
* Metadata Standard
* Annotation Protocol
* Chicken Behavior Ontology (CBO)

Each companion document expands one aspect of the overall annotation framework.

---

# 12. Future Extensions

Future releases of CBAS may include:

* Multi-animal interaction annotations
* Social relationship annotations
* Environmental event annotations
* Thermal imaging support
* Audio event annotations
* Sensor fusion
* Knowledge graph integration
* Cross-species annotation compatibility

---

# 13. Compliance

A dataset is considered CBAS-compliant when it:

* follows the annotation hierarchy,
* satisfies validation requirements,
* includes mandatory metadata,
* adheres to naming conventions, and
* clearly specifies the CBAS version used.

---

# 14. References

References to scientific publications, public datasets, and related annotation standards will be added in future releases as the framework evolves.

---

# Revision History

| Version       | Date            | Description                                                             |
| ------------- | --------------- | ----------------------------------------------------------------------- |
| 1.0.0 (Draft) | Initial release | First public specification of the Chicken Behavior Annotation Standard. |
