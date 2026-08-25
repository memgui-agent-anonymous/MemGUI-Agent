---
license: apache-2.0
task_categories:
  - image-text-to-text
  - visual-question-answering
language:
  - en
tags:
  - gui-agent
  - mobile-gui
  - android
  - memory
  - context-management
  - conact
  - long-horizon
size_categories:
  - 1K<n<10K
pretty_name: MemGUI-3K
configs:
  - config_name: task_index
    data_files:
      - split: all
        path: split.json
---

# MemGUI-3K

MemGUI-3K is a memory-intensive mobile GUI agent trajectory dataset for
training and analyzing agents that proactively manage long-horizon context.
It contains teacher rollouts from MemGUI-Agent using the ConAct
Context-as-Action paradigm, where the agent emits both GUI actions and context
actions for history folding and UI memory management.

Code, data processing scripts, model training scripts, and evaluation tools are
available in the MemGUI-Agent repository:

https://github.com/memgui-agent-anonymous/MemGUI-Agent

## Dataset Summary

| Metric | Value |
|---|---:|
| Total trajectories | 2,956 |
| Train trajectories | 2,661 |
| Test trajectories | 295 |
| Total task steps | 82,103 |
| Train task steps | 73,807 |
| Test task steps | 8,296 |
| Reasonable steps | 64,430 |
| Train reasonable steps | 57,951 |
| Test reasonable steps | 6,479 |
| Android apps covered | 26 |

## Dataset Structure

```text
MemGUI-3K/
|-- README.md
|-- metadata.json
|-- system_prompt.txt
|-- split.json
|-- train_trajectories.jsonl
|-- test_trajectories.jsonl
`-- image_archives/
    |-- images.z01
    |-- images.z02
    |-- images.z03
    |-- images.z04
    `-- images.zip
```

`split.json` is a task-level index with one row per trajectory. It exposes
`instruction`, `n_steps`, `n_reasonable_steps`, `trajectory_id`, `split`,
`reasonable_steps`, `action_type_counts`, and `n_memory_actions`. This is the
only file configured for the hosted table view, so the Dataset Viewer can
render quickly.

`train_trajectories.jsonl` and `test_trajectories.jsonl` contain one full
trajectory per line. Each trajectory contains evaluation metadata, IRR, token
statistics, and a nested `steps` array. Each step includes the action, user
prompt, assistant response, reasonableness annotation, token details, and a
screenshot path.

Screenshots are stored as split zip archives under `image_archives/` to avoid
uploading 82,103 individual PNG files. The archive restores to an `images/`
directory whose relative paths match the screenshot paths stored in the
trajectory files.

For downloading, restoring screenshots, rebuilding training JSONL files, and
running evaluation, see:

https://github.com/memgui-agent-anonymous/MemGUI-Agent

## License

MemGUI-3K is released under the Apache License 2.0.

## Citation

Citation metadata is withheld during anonymous review and will be restored
after the review period.
