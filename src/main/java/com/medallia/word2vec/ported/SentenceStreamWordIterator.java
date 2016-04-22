package com.medallia.word2vec.ported;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Iterator which produces words given an {@link Iterable} of sentences.
 *
 * This is not part of code ported from the original Word2Vec C version.
 */
class SentenceStreamWordIterator implements WordIterator {
	// Max word length
	private static final int MAX_STRING_LENGTH = Word2VecTrainer.MAX_STRING;
	private final Iterator<List<String>> sentences;
	private List<String> words;
	private int index;

	/** @param sentences sentences, each sentence is a {@link List} of {@link String}s (words). */
	SentenceStreamWordIterator(Iterable<List<String>> sentences) {
		this.sentences = sentences.iterator();
		words = null;
		index = 0;
	}

	@Override
	public void close() {
		// NO-OP for in memory {@link WordIterator}
	}

	@Override
	public boolean hasNext() {
		if (words != null && index <= words.size()) {
			return true;
		} else {
			if (!sentences.hasNext()) {
				return false;
			}
			List<String> nextLine = sentences.next();
			words = new ArrayList<>(nextLine.size());
			for (String word : nextLine) {
				if (!word.isEmpty()) {
					words.add(word);
				}
			}

			index = 0;
			return true;
		}
	}

	@Override
	public String next() {
		if (!hasNext()) {
			throw new NoSuchElementException("No more words");
		}
		if (index < words.size()) {
			String word = words.get(index);
			index++;
			return (word.length() > MAX_STRING_LENGTH) ? word.substring(0, MAX_STRING_LENGTH) : word;
		}
		index++;
		return WordIterator.NEW_LINE_TOKEN;
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException("Remove is not supported");
	}
}
