package com.medallia.word2vec;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.thrift.TException;
import org.junit.After;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.medallia.word2vec.Searcher.Match;
import com.medallia.word2vec.Searcher.UnknownWordException;
import com.medallia.word2vec.Word2VecTrainerBuilder.TrainingProgressListener;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
import com.medallia.word2vec.util.Common;
import com.medallia.word2vec.util.ThriftUtils;

/**
 * Tests for {@link Word2VecModel} and related classes.
 * <p>
 * Note that the implementation is expected to be deterministic if numThreads is
 * set to 1
 */
public class Word2VecTest {
	@Rule public ExpectedException expected = ExpectedException.none();

	/** Clean up after a test run */
	@After public void after() {
		// Unset the interrupted flag to avoid polluting other tests
		Thread.interrupted();
	}

	/** Test {@link textmining.word2vec.neuralnetwork.NeuralNetworkType#CBOW} */
	@Test public void testCBOW() throws IOException, TException, textmining.word2vec.ported.Word2VecException {
		assertModelMatches("cbowBasic.model",
				Word2VecModel.forRawWord2VecOnly(
						Word2VecModel.trainer(
								new Word2VecModelConfigThrift()
										.setMinFrequency(6)
										.setThreads(1)
										.setWindowSize(8)
										.setNeuralNetworkType(NeuralNetworkTypeThrift.CBOW)
										.setUseHierarchicalSoftmax(true)
										.setLayerSize(25)
										.setDownSampleRate(1e-3)
										.setIterations(1)
						)
								.train(testData()))
		);
	}

	/** Test {@link textmining.word2vec.neuralnetwork.NeuralNetworkType#CBOW} with 15 iterations */
	@Test public void testCBOWwith15Iterations() throws IOException, TException, textmining.word2vec.ported.Word2VecException {
		assertModelMatches("cbowIterations.model",
				Word2VecModel.forRawWord2VecOnly(
						Word2VecModel.trainer(
								new Word2VecModelConfigThrift()
										.setMinFrequency(5)
										.setThreads(1)
										.setWindowSize(8)
										.setNeuralNetworkType(NeuralNetworkTypeThrift.CBOW)
										.setUseHierarchicalSoftmax(true)
										.setLayerSize(25)
										.setNegativeSamples(5)
										.setDownSampleRate(1e-3)
										.setIterations(15)
						)
								.train(testData()))
		);
	}

	/** Test {@link textmining.word2vec.neuralnetwork.NeuralNetworkType#SKIP_GRAM} */
	@Test public void testSkipGram() throws IOException, TException, textmining.word2vec.ported.Word2VecException {
		assertModelMatches("skipGramBasic.model",
				Word2VecModel.forRawWord2VecOnly(
						Word2VecModel.trainer(
								new Word2VecModelConfigThrift()
										.setMinFrequency(6)
										.setThreads(1)
										.setWindowSize(8)
										.setNeuralNetworkType(NeuralNetworkTypeThrift.SKIP_GRAM)
										.setUseHierarchicalSoftmax(true)
										.setLayerSize(25)
										.setDownSampleRate(1e-3)
										.setIterations(1)
						)
								.train(testData()))
		);
	}

	/** Test {@link textmining.word2vec.neuralnetwork.NeuralNetworkType#SKIP_GRAM} with 15 iterations */
	@Test public void testSkipGramWith15Iterations() throws IOException, TException, textmining.word2vec.ported.Word2VecException {
		assertModelMatches("skipGramIterations.model",
				Word2VecModel.forRawWord2VecOnly(
						Word2VecModel.trainer(
								new Word2VecModelConfigThrift()
										.setMinFrequency(6)
										.setThreads(1)
										.setWindowSize(8)
										.setNeuralNetworkType(NeuralNetworkTypeThrift.SKIP_GRAM)
										.setUseHierarchicalSoftmax(true)
										.setLayerSize(25)
										.setDownSampleRate(1e-3)
										.setIterations(15)
						)
								.train(testData()))
		);
	}

	/**
	 * Long-running test with larger dataset. It uses skip gram with 15 iterations and
	 * hierarhical softmax. Minimum word count is 1.
	 *
	 * @see #largerData()
	 */
	@Test @Ignore
	public void testLargerDataset() throws IOException, textmining.word2vec.ported.Word2VecException, TException {
		assertModelMatches("text8-10M.model",
				Word2VecModel.forRawWord2VecOnly(
						Word2VecModel.trainer(
								new Word2VecModelConfigThrift()
										.setMinFrequency(100)
										.setThreads(1)
										.setWindowSize(5)
										.setNeuralNetworkType(NeuralNetworkTypeThrift.SKIP_GRAM)
										.setUseHierarchicalSoftmax(true)
										.setLayerSize(100)
										.setDownSampleRate(1e-3)
										.setNegativeSamples(0)
										.setIterations(15)
						)
								.train(largerData()))
		);
	}

	/**
	 * Test that we can interrupt the huffman encoding process
	 *
	 * TODO we need to decide if we want to rebuild interrupt functionality.
	 */
	@Test @Ignore
	public void testInterruptHuffman() throws IOException, textmining.word2vec.ported.Word2VecException {
		expected.expect(InterruptedException.class);
		trainer()
				.type(textmining.word2vec.neuralnetwork.NeuralNetworkType.SKIP_GRAM)
				.setNumIterations(15)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.CREATE_HUFFMAN_ENCODING)
							Thread.currentThread().interrupt();
						else if (stage == Stage.TRAIN_NEURAL_NETWORK)
							fail("Should not have reached this stage");
					}
				})
				.train(testData());
	}

	/**
	 * Test that we can interrupt the neural network training process
	 *
	 * TODO we need to decide if we want to rebuild interrupt functionality.
	 */
	@Test @Ignore
	public void testInterruptNeuralNetworkTraining() throws IOException, textmining.word2vec.ported.Word2VecException {
		expected.expect(InterruptedException.class);
		trainer()
				.type(textmining.word2vec.neuralnetwork.NeuralNetworkType.SKIP_GRAM)
				.setNumIterations(15)
				.setListener(new TrainingProgressListener() {
					@Override public void update(Stage stage, double progress) {
						if (stage == Stage.TRAIN_NEURAL_NETWORK)
							Thread.currentThread().interrupt();
					}
				})
				.train(testData());
	}

	/**
	 * Test the search results are deterministic
	 * Note the actual values may not make sense since the model we train isn't tuned
	 */
	@Test public void testSearch() throws IOException, UnknownWordException, textmining.word2vec.ported.Word2VecException {
		Word2VecModel model = Word2VecModel.forRawWord2VecOnly(trainer()
				.type(textmining.word2vec.neuralnetwork.NeuralNetworkType.SKIP_GRAM)
				.train(testData()));

		List<textmining.word2vec.Searcher.Match> matches = model.forSearch().getMatches("anarchism", 5);

		assertEquals(
				ImmutableList.of("anarcho", "specific", "as", "intellectual", "general"),
				Lists.transform(matches, textmining.word2vec.Searcher.Match.TO_WORD)
		);
	}

	/** @return {@link textmining.word2vec.ported.Word2VecTrainer} which by default uses all of the supported features */
	@VisibleForTesting public static Word2VecTrainerBuilder trainer() {
		return Word2VecModel.trainer()
						.setMinFrequency(6)
						.setThreads(1)
						.setWindowSize(8)
						.setNeuralNetworkType(NeuralNetworkTypeThrift.CBOW)
						.setUseHierarchicalSoftmax(true)
						.setLayerSize(25)
						.setDownSampleRate(1e-3)
						.setIterations(1);
	}

	/** @return raw test dataset. The tokens are separated by newlines. */
	@VisibleForTesting public static Iterable<List<String>> testData() throws IOException {
		List<String> lines = Common.readResource(Word2VecTest.class, "word2vec.short.txt");
		Iterable<List<String>> partitioned = Iterables.partition(lines, 1000);
		return partitioned;
	}

	/** @return raw dataset for larger dataset (10 million bytes of Wikipedia). */
	private static Iterable<List<String>> largerData() throws IOException {
		List<String> content = Common.readResource(Word2VecTest.class, "text8.10M");
		return ImmutableList.of(Strings.split(CollUtils.getOnlyElement(content), " "));
	}

	private void assertModelMatches(String expectedResource, Word2VecModel model) throws TException {
		final String thrift;
		try {
			thrift = Common.readResourceToStringChecked(getClass(), expectedResource);
		} catch (IOException ioe) {
			String filename = "/tmp/" + expectedResource;
			try {
				FileUtils.writeStringToFile(
						new File(filename),
						ThriftUtils.serializeJson(model.toThrift())
				);
			} catch (IOException e) {
				throw new AssertionError("Could not read resource " + expectedResource + " and could not write expected output to /tmp");
			}
			throw new AssertionError("Could not read resource " + expectedResource + " wrote to " + filename);
		}

		Word2VecModelThrift expected = ThriftUtils.deserializeJson(
				new Word2VecModelThrift(),
				thrift
		);

		assertEquals("Mismatched vocab", expected.getVocab().size(), Iterables.size(model.getVocab()));

		assertEquals(expected, model.toThrift());
	}
}
