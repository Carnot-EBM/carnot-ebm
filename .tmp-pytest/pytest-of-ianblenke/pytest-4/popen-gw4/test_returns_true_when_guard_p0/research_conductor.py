logger.info('RESEARCH STEP')
    task_id = task.get('id', '')
    if not validate_manifest_at_dequeue(task_id):
        return True
