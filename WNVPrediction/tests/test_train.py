import pytest
import train_pipeline

def test_train(roc_thresh):
	score = train_pipeline.run_training()
	assert score >= roc_thresh