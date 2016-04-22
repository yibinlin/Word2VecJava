package com.medallia.word2vec;

import java.util.Comparator;
import java.util.List;
import java.util.Set;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;

/** Provides search functionality */
public interface Searcher {
	/** @return true if a word is inside the model's vocabulary. */
	boolean contains(String word);
	
	/** @return Raw word vector */
	ImmutableList<Float> getRawVector(String word);
	
	/** @return Top matches to the given word, not including the given word. */
	List<Match> getMatches(String word, int maxMatches);

	/**
	 * For testing only.
	 *
	 * @param ignored words to be ignored in the match result
	 *
	 * @return top maxMatches words that are most similar to the given vec, excluding the words
	 * in the ignored list.
	 */
	List<Match> getMatchesFromVector(float[] wordVector, int maxMatches, Set<String> ignored);
	
	/** Represents the similarity between two words */
	interface SemanticDifference {
		/** @return Top matches to the given word which share this semantic relationship */
		List<Match> getMatches(String word, int maxMatches);
	}
	
	/** @return {@link SemanticDifference} between the word vectors for the given */
	SemanticDifference similarity(String s1, String s2);

	/** @return cosine similarity between two words. */
	float cosineDistance(String s1, String s2);
	
	/** Represents a match to a search word */
	interface Match {
		/** @return Matching word */
		String match();
		/** @return Cosine distance of the match */
		float distance();
		/** {@link Ordering} which compares {@link Match#distance()} */
		Comparator<Match> ORDERING = Comparator.comparing(match -> match.distance());

		/** {@link Function} which forwards to {@link #match()} */
		Function<Match, String> TO_WORD = result -> result.match();
	}

}
