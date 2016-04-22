package com.medallia.word2vec.neuralnetwork;

/**
 * Supported types for the neural network
 */
public enum NeuralNetworkType {
	/** Faster, slightly better accuracy for frequent words */
	CBOW {
		@Override public double getDefaultInitialLearningRate() {
			return 0.05;
		}
	},
	/** Slower, better for infrequent words */
	SKIP_GRAM {
		@Override public double getDefaultInitialLearningRate() {
			return 0.025;
		}
	},
	;
	
	/** @return Default initial learning rate */
	public abstract double getDefaultInitialLearningRate();
}