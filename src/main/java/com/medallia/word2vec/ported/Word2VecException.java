package com.medallia.word2vec.ported;

/**
 * Runtime exception thrown by {@link Word2VecTrainer} training process.
 */
public class Word2VecException extends Exception {
	/** Constructor with only message. */
	public Word2VecException(String message) {
		super(message);
	}

	/** Chained constructor with message. */
	public Word2VecException(String message, Throwable cause) {
		super(message, cause);
	}
}
