# Ground truth schema for dynamic/potentially-dynamic evaluation

One JSON file per annotated bag segment. Ground truth is **waypoint-based**,
not per-scan: you mark a handful of (time, position, status) points per
real-world object and the evaluator interpolates between them — far cheaper
than labeling every ~0.1s scan by hand, and accurate enough since object
motion between waypoints is close to linear over a few seconds.

```jsonc
{
  "objects": [
    {
      "gt_id": "person_A",          // your own label, never seen by the tracker under test
      "waypoints": [
        // status is a STEP function (holds until the next waypoint);
        // position is LINEARLY INTERPOLATED between consecutive waypoints.
        {"t": 10.0, "x": 2.00, "y": 1.00, "status": "static"},
        {"t": 15.0, "x": 2.00, "y": 1.00, "status": "static"},
        // two waypoints at the same (x,y) with different status = an
        // instantaneous state change (e.g. "starts walking here, at t=15.0"):
        {"t": 15.0, "x": 2.00, "y": 1.00, "status": "dynamic"},
        {"t": 20.0, "x": 4.00, "y": 1.50, "status": "dynamic"},
        {"t": 20.0, "x": 4.00, "y": 1.50, "status": "static"},
        {"t": 28.0, "x": 4.00, "y": 1.50, "status": "static"}
      ]
    }
  ]
}
```

## How to annotate

Use the two camera-track videos already produced for the report (persistent
YOLO/ByteTrack ids, independent of the pipeline being evaluated) as your
visual reference — they already show a timestamp and a person-level id per
camera. For each person/object visible in the bag:

1. Note the wall/bag time they enter frame, their approximate position
   (read off the LiDAR-processed point cloud in rviz, or estimate from the
   camera + known room layout), and their status at that moment.
2. Add a new waypoint every time position changes non-trivially (every
   1-3s while moving is plenty) or status changes (this is the important
   one — get the start/stop timestamps as precise as you can, since
   `debounce latency` is measured directly from these transitions).
3. `status` values: `"static"`, `"dynamic"`, `"potentially_dynamic"` — match
   `jo_msgs/Obstacle`'s own enum names (see `extract_obstacles.py`). Leave
   out `"tracked"` (v1's internal debounce-pending state) from ground truth
   entirely — it's a transient system state, not a real-world fact to
   annotate.

## What counts as agreement

`eval_tracking.py` samples both the ground truth (interpolated/stepped as
above) and the extracted prediction onto a common time grid
(`--frame-period`, default 0.5s), then matches GT objects to predicted
obstacles by position (Hungarian assignment, `--max-dist` cutoff, default
1.0m) independently at each grid timestamp — same protocol as CLEAR MOT /
MOTChallenge. A GT object with no predicted obstacle within `--max-dist` at
a given timestamp counts as a miss (false negative) regardless of status;
only matched pairs are compared on status.
