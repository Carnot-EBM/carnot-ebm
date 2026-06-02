def pick_next_task(completed_log):
    for task in RESEARCH_TASKS:
        excluded, reason = _task_is_excluded(task)
        if excluded:
            continue
        return task

def other_fn():
    pass
