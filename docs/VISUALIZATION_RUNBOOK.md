# CIL run visualizations

`visualize_cil_run.py` creates five diagnostic images from a completed run. It
does not retrain or modify checkpoints.

## One command after a run

From the repository root:

```bash
python scripts/visualize_cil_run.py \
  --run-root outputs/p3_hybrid_full8_e30_mem4_seed0 \
  --outdir outputs/p3_hybrid_full8_e30_mem4_seed0/visualizations \
  --ensemble
```

Use `--split validation` to inspect validation predictions. Use `--member-id 1`
without `--ensemble` to inspect one member. If prediction artifacts for all
members are present, `--ensemble` averages their probabilities after checking
sample-ID and label alignment.

## Five generated images

1. `01_buffer_allocation_*.png`: allocated slots by class, observed class
   support, and the buffer/support ratio. This checks whether the fixed budget
   actually reaches rare classes and whether large classes dominate the buffer.
2. `02_replay_exposure_*.png`: replay draws by class. If no replay audit was
   exported by the trainer, the figure is explicitly labelled as a planned
   buffer-allocation proxy; it must not be interpreted as actual exposure.
3. `03_classwise_performance_*.png`: per-class F1 and recall/balanced recall.
   Marker size is the evaluation support and the dashed lines are macro means.
4. `04_forgetting_heatmap_*.png`: class F1 at every available task boundary.
   The companion CSV contains `best_previous_f1`, `final_f1`, and the
   class-wise forgetting drop.
5. `05_confusion_matrix_*.png`: row-normalized confusion matrix plus the top
   off-diagonal confusion pairs. This shows which classes are being confused,
   not only that macro-F1 is low.

The output directory also contains CSV files for the plotted data and
`visualization_manifest.json`, which records the source artifacts and any
missing/invalid-artifact warnings. A missing NPZ or replay audit does not make
the command fail; it creates a placeholder image and records the reason.

The visualizations should be generated after offline evaluation, when
`member_predictions/task_*/{validation,test}.npz` and `buffer_audit.json` are
available. They are diagnostics, not replacements for the official aggregate
metrics in the run report.
