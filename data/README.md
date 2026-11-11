# Research data workspace

Original media and private annotation exports stay outside Git history. Keep
source files immutable; use the manifest to identify each asset and record
its provenance.

Suggested local workspace:

```text
data/
├── raw/
│   ├── videos/
│   └── images/
├── processed/
├── cvat/
│   └── native_exports/
├── annotations/
│   └── canonical/
└── media_manifest.json
```

## Rules

1. Put original videos and still images under `data/raw/`; do not overwrite them.
2. Put converted/transcoded files under `data/processed/`. Register each as a
   derivative asset and point `parent_asset_id` to its original.
3. Preserve native CVAT exports under `data/cvat/native_exports/`. Store
   canonical ChickenBehaviorLab annotations separately under
   `data/annotations/canonical/`.
4. Keep the real `media_manifest.json` at the local workspace root. Paths in it
   are relative to `data/`, not absolute machine-specific paths.
5. Record file size and SHA-256 at registration. Add technical metadata only
   after a successful probe. Unknown codec, dimensions, frame rate, duration,
   or frame count must remain null rather than being guessed.
6. Never store access tokens, passwords, private sharing URLs, or personal
   account information in the manifest. Optional source references must be
   non-secret opaque identifiers only.
7. Git ignores raw/processed media, native CVAT exports, private annotations,
   and the local manifest. The repository tracks only schemas, code, tests,
   documentation, and the example manifest.

The manifest implementation does not decode media. This separation allows
unusual source containers/codecs to be registered and checksummed before a
decoder or conversion workflow has been selected.
