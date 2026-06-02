from scripts.experiment_template import BatchedInferenceRunner
questions = load_questions(100)
runner = BatchedInferenceRunner(infer)
results = runner.run_batch(questions)
