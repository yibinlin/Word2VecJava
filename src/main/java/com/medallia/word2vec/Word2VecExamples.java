package com.medallia.word2vec;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.medallia.thrift.textmining.word2vec.NeuralNetworkTypeThrift;
import com.medallia.thrift.textmining.word2vec.Word2VecModelConfigThrift;
import com.medallia.thrift.textmining.word2vec.Word2VecModelThrift;
import common.AutoLog;
import common.Common;
import common.ProfilingTimer;
import common.ThriftUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.logging.Log;
import org.apache.thrift.TException;
import textmining.topicclustering.util.UnknownWordException;
import textmining.word2vec.Searcher.Match;
import textmining.word2vec.ported.Word2VecException;
import tiny.Strings;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

/** Example usages of {@link Word2Vec} */
public class Word2VecExamples {
	private static final Log LOG = AutoLog.getLog();

	/** Runs the example */
	public static void main(String[] args) throws IOException, TException, UnknownWordException, Word2VecException {
		demoWord();
	}

	/**
	 * Trains a model and allows user to find similar words
	 * demo-word.sh example from the open source C implementation
	 */
	public static void demoWord() throws IOException, TException, UnknownWordException, Word2VecException {
		List<String> read = Common.readToList(new File("text8"));
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override public List<String> apply(String input) {
				return Strings.split(input, " ");
			}
		});

		Word2VecModel model = Word2VecModel.forRawWord2VecOnly(
				Word2VecModel.trainer(
						new Word2VecModelConfigThrift()
								.setMinFrequency(5)
								.setThreads(20)
								.setWindowSize(8)
								.setNeuralNetworkType(NeuralNetworkTypeThrift.CBOW)
								.setUseHierarchicalSoftmax(false)
								.setLayerSize(5)
								.setNegativeSamples(25)
								.setDownSampleRate(1e-4)
								.setIterations(1)
				)
						.train(partitioned));

		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
			FileUtils.writeStringToFile(
					new File("text8.model"),
					ThriftUtils.serializeJson(model.toThrift())
			);
		}

		interact(model.forSearch());
	}

	/** Loads a model and allows user to find similar words */
	public static void loadModel() throws IOException, TException, UnknownWordException {
		final Word2VecModel model;
		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Loading model")) {
			String json = Common.readFileToString(new File("text8.model"));
			model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), json));
		}
		interact(model.forSearch());
	}

	/** Example using Skip-Gram model */
	public static void skipGram() throws IOException, TException, UnknownWordException, Word2VecException {
		List<String> read = Common.readToList(new File("sents.cleaned.word2vec.txt"));
		List<List<String>> partitioned = Lists.transform(read, new Function<String, List<String>>() {
			@Override public List<String> apply(String input) {
				return Strings.split(input, " ");
			}
		});

		Word2VecModel model = Word2VecModel.forRawWord2VecOnly(
				Word2VecModel.trainer(
						new Word2VecModelConfigThrift()
								.setMinFrequency(100)
								.setThreads(20)
								.setWindowSize(7)
								.setNeuralNetworkType(NeuralNetworkTypeThrift.SKIP_GRAM)
								.setUseHierarchicalSoftmax(true)
								.setLayerSize(300)
								.setNegativeSamples(0)
								.setDownSampleRate(1e-3)
								.setIterations(5)
				).train(partitioned));


		try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
			FileUtils.writeStringToFile(
					new File("300layer.20threads.5iter.model"),
					ThriftUtils.serializeJson(model.toThrift())
			);
		}

		interact(model.forSearch());
	}

	private static void interact(textmining.word2vec.Searcher searcher) throws IOException, UnknownWordException {
		try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
			while (true) {
				System.out.print("Enter word or sentence (EXIT to break): ");
				String word = br.readLine();
				if (word.equals("EXIT")) {
					break;
				}
				List<Match> matches = searcher.getMatches(word, 20);
				System.out.println(Strings.joinObjects("\n", matches));
			}
		}
	}
}
