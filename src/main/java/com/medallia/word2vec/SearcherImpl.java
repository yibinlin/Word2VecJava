package com.medallia.word2vec;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Ordering;
import com.google.common.primitives.Floats;
import textmining.topicclustering.util.UnknownWordException;
import tiny.Pair;
import tiny.streams.MoreCollectors;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

/** Implementation of {@link Searcher} */
public class SearcherImpl implements Searcher {
	private final NormalizedWord2VecModel model;
	private final ImmutableMap<String, Integer> word2vectorOffset;

	SearcherImpl(NormalizedWord2VecModel model) {
		this.model = model;

		final ImmutableMap.Builder<String, Integer> result = ImmutableMap.builder();
		for (int i = 0; i < model.vocab.size(); i++) {
			result.put(model.vocab.get(i), i * model.layerSize);
		}

		word2vectorOffset = result.build();
	}

	private void normalize(float[] v) {
		double len = 0;
		for (double d : v)
			len += d * d;
		len = Math.sqrt(len);

		for (int i = 0; i < v.length; i++)
			v[i] /= len;
	}

	@Override public List<Match> getMatches(String s, int maxNumMatches) throws UnknownWordException {
		return getMatches(getVector(s), maxNumMatches, ImmutableSet.of(s));
	}

	@Override
	public List<Match> getMatchesFromVector(float[] wordVector, int maxMatches, Set<String> ignored) {
		return getMatches(wordVector, maxMatches, ignored);
	}


	@Override public float cosineDistance(String s1, String s2) throws UnknownWordException {
		return calculateDistance(getVector(s1), getVector(s2));
	}

	@Override public boolean contains(String word) {
		return normalized.containsKey(word);
	}

	private List<Match> getMatches(float[] vec, int maxNumMatches, Set<String> ignored) {
		return Ordering.from(Match.ORDERING).greatestOf(
				model.vocab.stream()
				.filter(word -> !ignored.contains(word))
				.map( other -> {
						float[] otherVec = normalized.get(other);
						float d = calculateDistance(otherVec, vec);
						return new MatchImpl(other, d);
				}).collect(MoreCollectors.toImmutableList()),
				maxNumMatches
		);
	}

	private float calculateDistance(float[] otherVec, float[] vec) {
		float d = 0.0f;
		for (int a = 0; a < model.layerSize; a++)
			d += vec[a] * otherVec[a];
		return d;
	}

	@Override public ImmutableList<Float> getRawVector(String word) throws UnknownWordException {
		return ImmutableList.copyOf(Floats.asList(getVector(word)));
	}

	/**
	 * @return Vector for the given word
	 * @throws UnknownWordException If word is not in the model's vocabulary
	 */
	private float[] getVector(String word) throws UnknownWordException {
		if (!normalized.containsKey(word))
			throw new UnknownWordException(word);
		return normalized.get(word);
	}

	/** @return Vector difference from v1 to v2 */
	private float[] getDifference(float[] v1, float[] v2) {
		float[] diff = new float[model.layerSize];
		for (int i = 0; i < model.layerSize; i++)
			diff[i] = v1[i] - v2[i];
		return diff;
	}

	@Override public SemanticDifference similarity(String s1, String s2) throws UnknownWordException {
		float[] v1 = getVector(s1);
		float[] v2 = getVector(s2);
		final float[] diff = getDifference(v1, v2);

		return (word, maxMatches) -> {
				float[] target = getDifference(getVector(word), diff);
				return SearcherImpl.this.getMatches(target, maxMatches, ImmutableSet.of(word));
		};
	}

	/** Implementation of {@link Match} */
	private static class MatchImpl extends Pair<String, Float> implements Match {
		private MatchImpl(String first, Float second) {
			super(first, second);
		}

		@Override public String match() {
			return first;
		}

		@Override public float distance() {
			return second;
		}

		@Override public String toString() {
			return String.format("%s [%s]", first, second);
		}
	}
}
