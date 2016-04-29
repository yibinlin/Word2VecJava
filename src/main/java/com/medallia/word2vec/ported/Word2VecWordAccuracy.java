package textmining.word2vec.ported;

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.math.DoubleMath;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.util.AutoLog;
import org.apache.commons.logging.Log;

import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>
 * Calculate word2vec word accuracy based on compute-accuracy.c in original c codebase.
 *
 * <p>
 * Word accuracy calculation is described in the paper "Efficient Estimation of Word Representations in
 * Vector Space". The calculation is to answer a list of questions regarding to word relationship.
 * For example the following question:
 * <p>
 * <font face="courier">athens greece paris france</font>
 *
 * <p>
 * So a correct answer from the model should be france, given "athens greece paris" as the input.
 *
 * <p>
 * For sample usage, see corresponding test.
 */
public class Word2VecWordAccuracy {
	private static final Log LOG = AutoLog.getLog();

	private final Word2VecModel model;
	private final Searcher searcher;

	/**
	 * Explanation of internal variables.
	 *
	 * <ul>
	 * <li>TCN: total questions count in this section (separated by :) </li>
	 * <li>CCN: total correct TOP1 question count in this section: TOP1 means that the best distance word is
	 * the same as the expected result. </li>
	 * <li>TACN: total questions count </li>
	 * <li>CACN: total correct TOP1 question count </li>
	 * <li>SECN: total syntactic questions count </li>
	 * <li>SYCN: total semantic questions count </li>
	 * <li>SEAC: total syntactic TOP1 correct questions count </li>
	 * <li>SYAC: total semantic TOP1 correct questions count </li>
	 * <li>QID: section id </li>
	 * <li>TQ: total questions (including unanswered ones) count </li>
	 * <li>TQS: total questions that the program is able to answer </li>
	 * </ul>
	 */
	private int TCN = 0, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;

	Word2VecWordAccuracy(Word2VecModel model) {
		this.model = model;
		this.searcher = model.forSearch();
	}

	/**
	 * Compute word accuracy given a test file.
	 *
	 * @param threshold maximum number of vocabulary size. If the model has more than this number,
	 * the remaining words in the vocabulary will be ignored.
	 * @param questions the question file defined by the original C project, line by line.
	 */
	public AccuracyResult computeWordAccuracy(int threshold, List<String> questions) {
		TCN = 0;
		CCN = 0;
		TACN = 0;
		CACN = 0;
		SECN = 0;
		SYCN = 0;
		SEAC = 0;
		SYAC = 0;
		QID = 0;
		TQ = 0;
		TQS = 0;

		for (String line : questions) {
			String[] words = line.trim().split(" ");
			Preconditions.checkState(words.length > 0);
			if (words[0].equals(":") || words[0].equals("EXIT")) {
				printStats();
				LOG.info(String.format("%s:\n", words[1]));
				continue;
			}
			Preconditions.checkState(words.length >= 4);

			for (int i = 0; i < 4; i++) {
				words[i] = words[i].toLowerCase();
			}

			boolean outOfVocab = false;
			for (int i = 0; i < 4; i++) {
				if (searcher.contains(words[i]) && model.getWordIndex(words[i]) < threshold) {
					continue;
				} else {
					outOfVocab = true;
					break;
				}
			}
			TQ++;

			if (outOfVocab) {
				continue;
			}

			// virtual target word vector.
			float[] targetWordVec = new float[model.getLayerSize()];
			for (int i = 0; i < model.getLayerSize(); i++) {
				targetWordVec[i] = (searcher.getRawVector(words[1]).get(i) - searcher.getRawVector(words[0]).get(i)) + searcher.getRawVector(words[2]).get(i);
			}
			TQS++;

			// find most similar words.
			List<String> matches = searcher.getMatchesFromVector(targetWordVec, 5, ImmutableSet.of(words[0], words[1], words[2]))
					.stream()
					.map(match -> match.match())
					.collect(Collectors.toList());

			String theMatch = matches.get(0);

			if (theMatch.equals(words[3])) {
				CCN++;
				CACN++;
				if (QID <= 5) {
					SEAC++;
				} else {
					SYAC++;
				}
			}

			if (QID <= 5) {
				SECN++;
			} else {
				SYCN++;
			}
			TCN++;
			TACN++;
		}
		printStats();
		LOG.info(String.format("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS / (float) TQ * 100));

		return new AccuracyResult(TQS, TQ, CACN / (double) TACN * 100, SEAC / (double) SECN * 100, SYAC / (double) SYCN * 100);
	}

	/** Prints the status to the screen. */
	private void printStats() {
		if (TCN == 0) TCN = 1;
		if (QID != 0) {
			LOG.info(String.format("ACCURACY TOP1: %.2f %%  (%d / %d)\n",
					CCN / (double) TCN * 100,
					CCN,
					TCN));
			LOG.info(String.format("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n",
					CACN / (double) TACN * 100,
					SEAC / (double) SECN * 100,
					SYAC / (double) SYCN * 100));
		}
		QID++;
		TCN = 0;
		CCN = 0;
	}

	/**
	 * Result of Accuracy benchmarking.
	 */
	public static final class AccuracyResult {
		public final int questionsSeen;
		public final int questionsTotal;
		public final double totalAccuracy;
		public final double semanticAccuracy;
		public final double syntacticAccuracy;

		/**
		 * @param questionsSeen Questions that are able to be answered by the model
		 * @param questionsTotal Total questions in the questions txt file
		 * @param totalAccuracy total correct answers / total answerable questions
		 * @param semanticAccuracy total correct answers in semantic section / total answerable
		 * questions in semantic section
		 * @param syntacticAccuracy otal correct answers in syntactic section / total answerable
		 * questions in syntactic section
		 */
		public AccuracyResult(int questionsSeen, int questionsTotal, double totalAccuracy, double semanticAccuracy, double syntacticAccuracy) {
			this.questionsSeen = questionsSeen;
			this.questionsTotal = questionsTotal;
			this.totalAccuracy = totalAccuracy;
			this.semanticAccuracy = semanticAccuracy;
			this.syntacticAccuracy = syntacticAccuracy;
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == null) {
				return false;
			}
			if (obj instanceof AccuracyResult) {
				AccuracyResult other = (AccuracyResult) obj;
				return (this.questionsSeen == other.questionsSeen)
						&& (this.questionsTotal == other.questionsTotal)
						&& DoubleMath.fuzzyEquals(this.totalAccuracy, other.totalAccuracy, 1e-4)
						&& DoubleMath.fuzzyEquals(this.semanticAccuracy, other.semanticAccuracy, 1e-4)
						&& DoubleMath.fuzzyEquals(this.syntacticAccuracy, other.syntacticAccuracy, 1e-4);
			}

			return false;
		}

		@Override
		public int hashCode() {
			return Objects.hashCode(questionsSeen, questionsTotal, totalAccuracy, semanticAccuracy, syntacticAccuracy);
		}

		@Override public String toString() {
			return MoreObjects.toStringHelper(this)
					.add("questionsSeen", questionsSeen)
					.add("questionsTotal", questionsTotal)
					.add("totalAccuracy", totalAccuracy)
					.add("semanticAccuracy", semanticAccuracy)
					.add("syntacticAccuracy", syntacticAccuracy)
					.toString();
		}
	}
}
