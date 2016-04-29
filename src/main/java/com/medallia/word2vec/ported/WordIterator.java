package com.medallia.word2vec.ported;

import com.medallia.word2vec.util.AC;

import java.util.Iterator;

/**
 * Iterator of words for {@link Word2VecTrainer}. Each invocation of {@link #next()} returns a new
 * word, separated by (\r)?\n, \s, or \t.
 *
 * This is not part of code ported from the original Word2Vec C version.
 */
interface WordIterator extends Iterator<String>, AC {
	String NEW_LINE_TOKEN = "</s>";
}
