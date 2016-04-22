package com.medallia.word2vec.ported;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Multiset;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkConfig;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.util.AutoLog;
import com.medallia.word2vec.util.Strings;
import org.apache.commons.logging.Log;
import org.joda.time.DateTime;
import org.joda.time.Seconds;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.EOFException;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * Class to perform training the word2vec model.
 * <p>
 * 
 * This class is a direct port from C word2vec (https://code.google.com/archive/p/word2vec/)
 * revision 41 and it is supposed to perform exactly the same training process (even including the
 * random number generator sequence).
 * <p>
 * 
 * It has been verified that the generated model file using {@link #TrainModel()} is exactly the
 * same as the generated model from C version, given the same sorted vocabulary using
 * {@link #ReadVocab()}. Therefore, we are certain of the correctness of this Java version compared
 * to the original C version.
 * <p>
 * 
 * The {@link #main(String[])} function is also a port from the Word2vec C command line main
 * function.
 */
public class Word2VecTrainer {
	private final Log log;

	static int MAX_STRING = 100;
	private static final int EXP_TABLE_SIZE = 1000;
	private static final int MAX_EXP = 6;
	private static final int MAX_SENTENCE_LENGTH = 1000;
	private static final int MAX_CODE_LENGTH = 40;

	static final int vocab_hash_size = 30000000; // Maximum 30 * 0.7 = 21M words in the vocabulary

	/** Class for a word in training dataset. */
	private static class vocab_word implements Comparable<vocab_word>
	{
		long cn;
		int[] point;
		String word;
		int[] code;
		int codeLen;

		/** Compare by count. */
		@Override
		public int compareTo(vocab_word o) {
			if (o == null) {
				return 1;
			}
			long compare = (o.cn - this.cn);
			if (compare > 0L) {
				return 1;
			} else if (compare < 0L) {
				return -1;
			} else {
				return 0;
			}
		}
	}

	/** sentences, each sentence is a list of Strings (words). */
	private final Iterable<List<String>> sentences;
	private final String output_file;
	private final String read_vocab_file;
	private final Optional<Multiset<String>> overwrite_vocab;
	private vocab_word[] vocab;
	private int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
	private final int[] vocab_hash;
	private int vocab_size = 0;
	private int vocab_max_size = 1000, layer1_size = 100;
	private long train_words = 0, word_count_actual = 0;
	private int classes = 0;
	private int iter = 5;
	private float alpha = 0.025f;
	private float starting_alpha, sample = 1e-3f;
	// TODO TA-771 may probably need to be volatile for multithreading env
	private float[] syn0, syn1, syn1neg, expTable;
	private DateTime start;

	// Hierachical softmax
	private boolean hs = false;
	private int negative = 5;
	private final int table_size = (int) 1e8;
	private int[] table;

	/**
	 * For automated training process.
	 *
	 * @param minFrequency minimum word frequency for the word to be included in the training
	 * process
	 * @param overwrite_vocab vocabulary for word2vec to use, instead of collecting it from the
	 * sentences parameter.
	 * @param debugLevel 0 for silent, 2 for everything/debugging (default 1).
	 */
	public Word2VecTrainer(
			Log log,
			int debugLevel,
			Integer minFrequency,
			Optional<Multiset<String>> overwrite_vocab,
			NeuralNetworkConfig neuralNetworkConfig,
			Iterable<List<String>> sentences) {
		this(
				log,
				neuralNetworkConfig.getLayerSize(),
				sentences,
				overwrite_vocab,
				"",
				debugLevel,
				neuralNetworkConfig.getType() == NeuralNetworkType.CBOW ? 1 : 0,
				(float) neuralNetworkConfig.getInitialLearningRate(),
				"",
				neuralNetworkConfig.getWindowSize(),
				(float) neuralNetworkConfig.getDownSampleRate(),
				neuralNetworkConfig.isUseHierarchicalSoftmax(),
				neuralNetworkConfig.getNegativeSamples(),
				neuralNetworkConfig.getNumThreads(),
				neuralNetworkConfig.getIterations(),
				minFrequency);
	}

	/**
	 * Direct use of this constructor is for debugging only.
	 *
	 * For explanations of each parameter please see {@link #main(String[])}.
	 */
	private Word2VecTrainer(
			Log log,
			int layer1_size,
			Iterable<List<String>> sentences,
			Optional<Multiset<String>> overwrite_vocab, // in-memory vocabulary overwrite
			String read_vocab_file,
			int debug_mode, // larger than or equal to 0, the larger it is, the more specific, largest is 2 for debugging
			int cbow,
			float alpha,
			String output_file,
			int window,
			float sample,
			boolean hs,
			int negative,
			int num_threads,
			int iter,
			int min_count) {
		this.log = log;
		this.layer1_size = layer1_size;
		this.sentences = sentences;
		this.overwrite_vocab = overwrite_vocab;
		this.read_vocab_file = Strings.hasContent(read_vocab_file) ? read_vocab_file : "";
		this.debug_mode = debug_mode;
		this.cbow = cbow;
		this.alpha = alpha;
		this.output_file = Strings.hasContent(output_file) ? output_file : "";
		this.window = window;
		this.sample = sample;
		this.hs = hs;
		this.negative = negative;
		this.num_threads = num_threads;
		this.iter = iter;
		this.min_count = min_count;

		this.vocab_hash = new int[vocab_hash_size];
		this.expTable = new float[EXP_TABLE_SIZE + 1];
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = (float) Math.exp((i / (float) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
			expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
		}

		this.vocab = new vocab_word[vocab_max_size];
		for (int j = 0; j < vocab_max_size; j++) {
			vocab[j] = new vocab_word();
		}
	}

	private void InitUnigramTable() {
		int a, i;
		long train_words_pow = 0;
		float d1, power = 0.75f;
		table = new int[table_size];
		for (a = 0; a < vocab_size; a++)
			train_words_pow += Math.pow(vocab[a].cn, power);
		i = 0;
		d1 = (float) Math.pow(vocab[i].cn, power) / train_words_pow;
		for (a = 0; a < table_size; a++) {
			table[a] = i;
			if (a / (float) table_size > d1) {
				i++;
				d1 += (float) Math.pow(vocab[i].cn, power) / train_words_pow;
			}
			if (i >= vocab_size) i = vocab_size - 1;
		}
	}

	/** Iterate a file by word, only for testing. */
	@VisibleForTesting
	public static class FileWordIterator implements WordIterator {
		private static final Log LOG = AutoLog.getLog();

		private static final String pattern = "\\s|\\t|(\\r)?\\n";
		private final RandomAccessFile file;
		private List<String> words;
		private int index;

		/** Only constructor taking a {@link RandomAccessFile} object. */
		public FileWordIterator(RandomAccessFile file) {
			this.file = file;
			words = null;
			index = 0;
		}

		@Override
		public boolean hasNext() {
			if (words != null && index <= words.size()) {
				return true;
			} else {
				String nextLine = null;
				try {
					nextLine = file.readLine();
				} catch (EOFException e) {
					return false;
				} catch (IOException e) {
					String message = "Error reading training file during word2vec training. ";
					LOG.error(message, e);
					throw new IllegalStateException(message, e);
				}

				if (nextLine == null) {
					return false;
				}

				String[] rawWords = nextLine.split(pattern);
				words = new ArrayList<>(rawWords.length);
				for (String word : rawWords) {
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
				throw new IllegalStateException("No more words");
			}
			if (index < words.size()) {
				String word = words.get(index);
				index++;
				if (word.length() > MAX_STRING) {
					return word.substring(0, MAX_STRING);
				}
				return word;
			} else {
				index++;
				return WordIterator.NEW_LINE_TOKEN;
			}
		}

		@Override
		public void close() {
			try {
				file.close();
			} catch (IOException e) {
				// Close silently.
			}
		}

		/**
		 * @return sentences, as defined in {@link Word2VecTrainer}, by iterating a
		 * {@link RandomAccessFile} using the logic of {@link FileWordIterator}.
		 */
		@VisibleForTesting
		public static Iterable<List<String>> getSentencesFromFile(RandomAccessFile file) {
			try (FileWordIterator wordIterator = new FileWordIterator(file)) {
				List<List<String>> sentences = new ArrayList<>();
				List<String> currentSentence = new ArrayList<>();
				while (wordIterator.hasNext()) {
					String word = wordIterator.next();
					if (word.equals(WordIterator.NEW_LINE_TOKEN)) {
						sentences.add(currentSentence);
						currentSentence = new ArrayList<>();
					} else {
						currentSentence.add(word);
					}
				}
				if (!currentSentence.isEmpty()) {
					sentences.add(currentSentence);
				}
				return sentences;
			}
		}
	}

	// Returns hash value of a word
	static int GetWordHash(String word) {
		long a, hash = 0;
		for (a = 0; a < word.length(); a++)
			hash = hash * 257 + word.charAt((int) a);
		hash = module(hash, vocab_hash_size);
		return (int) hash;
	}

	static int module(long number, int divide) {
		return (int) ((number % divide) + divide) % divide;
	}

	static int module(BigInteger number, int divide) {
		return number.mod(BigInteger.valueOf(divide)).intValue();
	}

	// Returns position of a word in the vocabulary; if the word is not found, returns -1
	private int SearchVocab(String word) {
		int hash = GetWordHash(word);
		while (true) {
			if (vocab_hash[hash] == -1) return -1;
			if (word.equals(vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
			hash = module((hash + 1), vocab_hash_size);
		}
	}

	// Reads a word and returns its index in the vocabulary
	private int ReadWordIndex(WordIterator iterator) {
		if (!iterator.hasNext()) return -1;
		return SearchVocab(iterator.next());
	}

	// Adds a word to the vocabulary
	private int AddWordToVocab(String word) {
		int hash, length = word.length() + 1;
		if (length > MAX_STRING) length = MAX_STRING;

		vocab[vocab_size].word = word;

		vocab[vocab_size].cn = 0;
		vocab_size++;
		// Reallocate memory if needed
		if (vocab_size + 2 >= vocab_max_size) {
			vocab_max_size += 1000;
			vocab_word[] newVocab = new vocab_word[vocab_max_size];
			for (int i = 0; i < vocab.length; i++) {
				newVocab[i] = vocab[i];
			}
			for (int i = vocab.length; i < vocab_max_size; i++) {
				newVocab[i] = new vocab_word();
			}

			vocab = newVocab;
		}
		hash = GetWordHash(word);
		while (vocab_hash[hash] != -1) {
			hash = module((hash + 1), vocab_hash_size);
		}
		vocab_hash[hash] = vocab_size - 1;
		return vocab_size - 1;
	}

	// Sorts the vocabulary by frequency using word counts
	private void SortVocab() {
		int a, size;
		int hash;
		// Sort the vocabulary and keep </s> at the first position
		// Note this uses a stable sorting algorithm to make debugging easier.
		Arrays.sort(vocab, 1, vocab_size);

		for (a = 0; a < vocab_hash_size; a++)
			vocab_hash[a] = -1;
		size = vocab_size;
		train_words = 0;
		if (debug_mode > 0) {
			log.info(String.format("original vocab size: %s.", vocab_size));
		}

		for (a = 0; a < size; a++) {
			// Words occuring less than min_count times will be discarded from the vocab
			if ((vocab[a].cn < min_count) && (a != 0)) {
				vocab_size--;
				vocab[a] = null;
			} else {
				// Hash will be re-computed, as after the sorting it is not actual
				hash = GetWordHash(vocab[a].word);
				while (vocab_hash[hash] != -1)
					hash = module((hash + 1), vocab_hash_size);
				vocab_hash[hash] = a;
				train_words += vocab[a].cn;
			}
		}

		vocab_word[] newVocab = new vocab_word[vocab_size + 1];
		for (int i = 0; i < vocab_size; i++) {
			newVocab[i] = vocab[i];
		}

		// Allocate memory for the binary tree construction
		for (a = 0; a < vocab_size; a++) {
			vocab[a].code = new int[MAX_CODE_LENGTH];
			vocab[a].point = new int[MAX_CODE_LENGTH];
		}
	}

	// Reduces the vocabulary by removing infrequent tokens
	private void ReduceVocab() {
		int a, b = 0;
		int hash;
		for (a = 0; a < vocab_size; a++)
			if (vocab[a].cn > min_reduce) {
				vocab[b].cn = vocab[a].cn;
				vocab[b].word = vocab[a].word;
				b++;
			} else {
				vocab[a] = null;
			}
		vocab_size = b;
		for (a = 0; a < vocab_hash_size; a++)
			vocab_hash[a] = -1;
		for (a = 0; a < vocab_size; a++) {
			// Hash will be re-computed, as it is not actual
			hash = GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1)
				hash = module((hash + 1), vocab_hash_size);
			vocab_hash[hash] = a;
		}
		min_reduce++;
	}

	// Create binary Huffman tree using the word counts
	// Frequent words will have short unique binary codes
	private void CreateBinaryTree() {
		int a, b, i, min1i, min2i, pos1, pos2;
		int[] point = new int[MAX_CODE_LENGTH];
		int[] code = new int[MAX_CODE_LENGTH];
		long[] count = new long[vocab_size * 2 + 1];
		long[] binary = new long[vocab_size * 2 + 1];
		int[] parent_node = new int[vocab_size * 2 + 1];
		for (a = 0; a < vocab_size; a++)
			count[a] = vocab[a].cn;
		for (a = vocab_size; a < vocab_size * 2; a++)
			count[a] = (long) 1e15;
		pos1 = vocab_size - 1;
		pos2 = vocab_size;
		// Following algorithm constructs the Huffman tree by adding one node at a time
		for (a = 0; a < vocab_size - 1; a++) {
			// First, find two smallest nodes 'min1, min2'
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min1i = pos1;
					pos1--;
				} else {
					min1i = pos2;
					pos2++;
				}
			} else {
				min1i = pos2;
				pos2++;
			}
			if (pos1 >= 0) {
				if (count[pos1] < count[pos2]) {
					min2i = pos1;
					pos1--;
				} else {
					min2i = pos2;
					pos2++;
				}
			} else {
				min2i = pos2;
				pos2++;
			}
			count[vocab_size + a] = count[min1i] + count[min2i];
			parent_node[min1i] = vocab_size + a;
			parent_node[min2i] = vocab_size + a;
			binary[min2i] = 1;
		}
		// Now assign binary code to each vocabulary word
		for (a = 0; a < vocab_size; a++) {
			b = a;
			i = 0;
			while (true) {
				code[i] = (int) binary[b];
				point[i] = b;
				i++;
				b = parent_node[b];
				if (b == vocab_size * 2 - 2) break;
			}
			vocab[a].codeLen = i;
			vocab[a].point[0] = vocab_size - 2;
			for (b = 0; b < i; b++) {
				vocab[a].code[i - b - 1] = code[b];
				vocab[a].point[i - b] = point[b] - vocab_size;
			}
		}
	}

	private void LearnVocabFromTrainFile() {
		String word;

		try (WordIterator wordIterator = new SentenceStreamWordIterator(sentences)) {
			int a, i;
			for (a = 0; a < vocab_hash_size; a++)
				vocab_hash[a] = -1;
			vocab_size = 0;
			AddWordToVocab("</s>");
			while (wordIterator.hasNext()) {
				word = wordIterator.next();
				train_words++;
				if ((debug_mode > 1) && (train_words % 100000 == 0)) {
					log.info(String.format("%s%c", train_words / 1000, 13));
				}
				i = SearchVocab(word);
				if (i == -1) {
					a = AddWordToVocab(word);
					vocab[a].cn = 1;
				} else
					vocab[i].cn++;
				if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
			}
			SortVocab();
			if (debug_mode > 0) {
				log.info(String.format("filtered Vocab size: %s (min count: %s)", vocab_size, min_count));
				log.info(String.format("Words in train file: %s", train_words));
			}
		}
	}

	private void constructVocabFromInMemoryOverride() {
		readInMemoryOverride(overwrite_vocab.get());
		SortVocab();
		if (debug_mode > 0) {
			log.info(String.format("Vocab size: %s", vocab_size));
			log.info(String.format("Words in train file: %s", train_words));
		}
	}

	/** Deal with in-memory override for word vocabulary. */
	private void readInMemoryOverride(Multiset<String> vocabOverwrite) {
		Preconditions.checkArgument(vocabOverwrite != null);
		for (int i = 0; i < vocab_hash_size; i++)
			vocab_hash[i] = -1;
		vocab_size = 0;
		for (Multiset.Entry<String> entry : vocabOverwrite.entrySet()) {
			int a = AddWordToVocab(entry.getElement());
			vocab[a].cn = entry.getCount();
		}
	}

	private void ReadVocab() throws Word2VecException {
		try (BufferedReader reader = new BufferedReader(new FileReader(new File(read_vocab_file)))) {
			String strLine;

			for (int i = 0; i < vocab_hash_size; i++)
				vocab_hash[i] = -1;
			vocab_size = 0;
			while ((strLine = reader.readLine()) != null)
			{
				strLine = strLine.trim();
				String[] wordCount = strLine.split(",");
				Preconditions.checkState(wordCount.length == 2);
				int a = AddWordToVocab(wordCount[0]);
				vocab[a].cn = Long.parseLong(wordCount[1]);
			}
		} catch (IOException e) {
			String message = "Error reading vocabulary file for word2vec training";
			log.warn(message, e);
			throw new Word2VecException(message, e);
		}
		SortVocab();
		if (debug_mode > 0) {
			log.info(String.format("Vocab size: %s", vocab_size));
			log.info(String.format("Words in train file: %s", train_words));
		}
	}

	private void InitNet() {
		int a, b;
		long next_random = 1;
		syn0 = new float[vocab_size * layer1_size];
		if (hs) {
			syn1 = new float[vocab_size * layer1_size];
			for (a = 0; a < vocab_size; a++) {
				for (b = 0; b < layer1_size; b++) {
					syn1[a * layer1_size + b] = 0;
				}
			}
		}
		if (negative > 0) {
			syn1neg = new float[vocab_size * layer1_size];
			for (a = 0; a < vocab_size; a++) {
				for (b = 0; b < layer1_size; b++) {
					syn1neg[a * layer1_size + b] = 0;
				}
			}
		}
		for (a = 0; a < vocab_size; a++)
			for (b = 0; b < layer1_size; b++) {
				next_random = next_random * 25214903917L + 11;
				syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float) 65536) - 0.5f) / layer1_size;
			}
		CreateBinaryTree();
	}

	/** Actual model training logic */
	private class TrainModelThread implements Runnable {

		public final int id;

		TrainModelThread(int id) {
			this.id = id;
		}

		@Override
		public void run() {
			try {
				int a, b, last_word, d;
				long cw;
				int word, sentence_length = 0, sentence_position = 0;
				long word_count = 0L, last_word_count = 0L;
				int[] sen = new int[MAX_SENTENCE_LENGTH + 1];
				int l1, l2, c;
				int target, label;
				int local_iter = iter;
				long next_random = id;
				float f, g;
				DateTime now;
				float[] neu1 = new float[layer1_size];
				float[] neu1e = new float[layer1_size];
				// TODO TA-771 write pseudo-seek functionality for multithreading environment
				//                RandomAccessFile file = new RandomAccessFile(new File(train_file), "r");
				//                file.seek(file_size / (long) num_threads * (long) id);

				WordIterator wordIterator = new SentenceStreamWordIterator(sentences);
				while (true) {
					if (word_count - last_word_count > 10000) {
						word_count_actual += word_count - last_word_count;
						last_word_count = word_count;
						alpha = starting_alpha * (1 - word_count_actual / (float) (iter * train_words + 1));
						if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001f;
						if ((debug_mode > 1)) {
							now = new DateTime();
							log.info(String.format("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
									word_count_actual / (float) (iter * train_words + 1) * 100,
									word_count_actual / ((float) (Seconds.secondsBetween(start, now).getSeconds() + 1) * 1000)));
						}
					}
					if (sentence_length == 0) {
						while (wordIterator.hasNext()) {
							word = ReadWordIndex(wordIterator);
							if (word == -1) continue;
							word_count++;
							if (word == 0) break;
							// The subsampling randomly discards frequent words while keeping the ranking same
							if (sample > 0) {
								float ran = ((float) Math.sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
								next_random = next_random * 25214903917L + 11L;
								if (ran < (next_random & 0xFFFF) / (float) 65536) continue;
							}
							sen[sentence_length] = word;
							sentence_length++;
							if (sentence_length >= MAX_SENTENCE_LENGTH) break;
						}
						sentence_position = 0;
					}
					if ((!wordIterator.hasNext() || (word_count > train_words / num_threads)) && sentence_length == 0) {
						word_count_actual += word_count - last_word_count;

						// Dump each iteration
						BufferedWriter fo = new BufferedWriter(new FileWriter(output_file + "_" + local_iter));
						writeWordModel(fo);
						fo.close();

						local_iter--;
						if (local_iter == 0) break;
						word_count = 0;
						last_word_count = 0;
						sentence_length = 0;
						wordIterator.close();
						wordIterator = new SentenceStreamWordIterator(sentences);
						continue;
					}
					word = sen[sentence_position];
					if (word == -1) continue;
					for (c = 0; c < layer1_size; c++)
						neu1[c] = 0;
					for (c = 0; c < layer1_size; c++)
						neu1e[c] = 0;

					next_random = next_random * 25214903917L + 11;
					BigInteger big_next_random = parseBigIntegerPositive(next_random);
					b = (module(big_next_random, window));
					if (cbow == 1) { //train the cbow architecture
						// in -> hidden
						cw = 0;
						for (a = b; a < window * 2 + 1 - b; a++)
							if (a != window) {
								c = sentence_position - window + a;
								if (c < 0) continue;
								if (c >= sentence_length) continue;
								last_word = sen[c];
								if (last_word == -1) continue;
								for (c = 0; c < layer1_size; c++)
									neu1[c] += syn0[(c + last_word * layer1_size)];
								cw++;
							}
						if (cw > 0L) {
							for (c = 0; c < layer1_size; c++)
								neu1[c] /= cw;
							if (hs) for (d = 0; d < vocab[word].codeLen; d++) {
								f = 0;
								l2 = vocab[word].point[d] * layer1_size;
								// Propagate hidden -> output
								for (c = 0; c < layer1_size; c++)
									f += neu1[c] * syn1[c + l2];
								if (f <= -MAX_EXP)
									continue;
								else if (f >= MAX_EXP)
									continue;
								else
									f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
								// 'g' is the gradient multiplied by the learning rate
								g = (1 - vocab[word].code[d] - f) * alpha;
								// Propagate errors output -> hidden
								for (c = 0; c < layer1_size; c++)
									neu1e[c] += g * syn1[c + l2];
								// Learn weights hidden -> output
								for (c = 0; c < layer1_size; c++)
									syn1[c + l2] += g * neu1[c];
							}
							// NEGATIVE SAMPLING
							if (negative > 0) for (d = 0; d < negative + 1; d++) {
								if (d == 0) {
									target = word;
									label = 1;
								} else {
									next_random = next_random * 25214903917L + 11;
									big_next_random = parseBigIntegerPositive(next_random);
									target = table[module((big_next_random.shiftRight(16)), table_size)];
									if (target == 0) target = module(next_random, (vocab_size - 1) + 1);
									if (target == word) continue;
									label = 0;
								}
								l2 = target * layer1_size;
								f = 0;
								for (c = 0; c < layer1_size; c++)
									f += neu1[c] * syn1neg[c + l2];
								if (f > MAX_EXP)
									g = (label - 1) * alpha;
								else if (f < -MAX_EXP)
									g = (label - 0) * alpha;
								else
									g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
								for (c = 0; c < layer1_size; c++)
									neu1e[c] += g * syn1neg[c + l2];
								for (c = 0; c < layer1_size; c++)
									syn1neg[c + l2] += g * neu1[c];
							}
							// hidden -> in
							for (a = b; a < window * 2 + 1 - b; a++)
								if (a != window) {
									c = sentence_position - window + a;
									if (c < 0) continue;
									if (c >= sentence_length) continue;
									last_word = sen[c];
									if (last_word == -1) continue;
									for (c = 0; c < layer1_size; c++)
										syn0[c + last_word * layer1_size] += neu1e[c];
								}
						}
					} else { //train skip-gram
						for (a = b; a < window * 2 + 1 - b; a++) {
							if (a != window) {
								c = sentence_position - window + a;
								if (c < 0) continue;
								if (c >= sentence_length) continue;
								last_word = sen[c];

								if (last_word == -1) continue;
								l1 = last_word * layer1_size;
								for (c = 0; c < layer1_size; c++)
									neu1e[c] = 0;
								// HIERARCHICAL SOFTMAX
								if (hs) for (d = 0; d < vocab[word].codeLen; d++) {
									f = 0;
									l2 = vocab[word].point[d] * layer1_size;
									// Propagate hidden -> output
									for (c = 0; c < layer1_size; c++)
										f += syn0[c + l1] * syn1[c + l2];
									if (f <= -MAX_EXP)
										continue;
									else if (f >= MAX_EXP)
										continue;
									else
										f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
									// 'g' is the gradient multiplied by the learning rate
									g = (1 - vocab[word].code[d] - f) * alpha;
									// Propagate errors output -> hidden
									for (c = 0; c < layer1_size; c++)
										neu1e[c] += g * syn1[c + l2];
									// Learn weights hidden -> output
									for (c = 0; c < layer1_size; c++)
										syn1[c + l2] += g * syn0[c + l1];
								}
								// NEGATIVE SAMPLING
								if (negative > 0) for (d = 0; d < negative + 1; d++) {
									if (d == 0) {
										target = word;
										label = 1;
									} else {
										next_random = next_random * 25214903917L + 11L;
										big_next_random = parseBigIntegerPositive(next_random);
										target = table[module((big_next_random.shiftRight(16)), table_size)];
										if (target == 0) target = module(next_random, (vocab_size - 1) + 1);
										if (target == word) continue;
										label = 0;
									}
									l2 = target * layer1_size;
									f = 0;
									for (c = 0; c < layer1_size; c++)
										f += syn0[c + l1] * syn1neg[c + l2];
									if (f > MAX_EXP)
										g = (label - 1) * alpha;
									else if (f < -MAX_EXP)
										g = (label - 0) * alpha;
									else
										g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
									for (c = 0; c < layer1_size; c++)
										neu1e[c] += g * syn1neg[c + l2];
									for (c = 0; c < layer1_size; c++)
										syn1neg[c + l2] += g * syn0[c + l1];
								}
								// Learn weights input -> hidden
								for (c = 0; c < layer1_size; c++)
									syn0[c + l1] += neu1e[c];
							}
						}
					}
					sentence_position++;
					if (sentence_position >= sentence_length) {
						sentence_length = 0;
						continue;
					}
				}
				wordIterator.close();
			} catch (IOException e) {
				String message = "Unexpected IOException occured during training the neurons.";
				log.warn(message, e);
				// Throws an (unchecked) runtime exception and the main thread should convert the exception to a {@link Word2VecException}..
				throw new RuntimeException(message, e);
			}
		}
	}

	private static final BigInteger TWO_COMPL_REF = BigInteger.ONE.shiftLeft(64);

	private static BigInteger parseBigIntegerPositive(long num) {
		BigInteger b = BigInteger.valueOf(num);
		if (b.compareTo(BigInteger.ZERO) < 0)
			b = b.add(TWO_COMPL_REF);
		return b;
	}

	/**
	 * Run recurrent neural network training of specific {@link NeuralNetworkType}.
	 *
	 * @return word2vec model after the training process.
	 */
	public RawWord2VecModel TrainModel() throws Word2VecException {
		try {
			int a;

			starting_alpha = alpha;
			if (overwrite_vocab.isPresent()) {
				constructVocabFromInMemoryOverride();
			} else if (!read_vocab_file.isEmpty()) {
				ReadVocab();
			} else {
				LearnVocabFromTrainFile();
			}

			InitNet();
			if (negative > 0) InitUnigramTable();
			start = new DateTime();

			ExecutorService executor = Executors.newFixedThreadPool(Math.min(num_threads, Runtime.getRuntime().availableProcessors()));

			List<Future<?>> threads = new ArrayList<>();
			for (a = 0; a < num_threads; a++) {
				threads.add(executor.submit(new TrainModelThread(a)));
			}
			for (a = 0; a < num_threads; a++)
				threads.get(a).get();
			if (classes == 0) {
				if (!output_file.isEmpty() && binary == 0) {
					try (BufferedWriter fo = new BufferedWriter(new FileWriter(output_file))) {
						writeWordModel(fo);
					}
				}
			} else {
				// Run K-means on the word vectors
				// TODO port it from C later.
			}

			executor.shutdown();
			return new RawWord2VecModel(
					Arrays.stream(vocab)
							.filter(vocabWord -> (vocabWord != null) && vocabWord.word != null)
							.map(vocabWord -> vocabWord.word)
							.collect(Collectors.toList()),
					layer1_size,
					syn0);
		} catch (IOException e) {
			throw new Word2VecException("IOException occured while writing output model file", e);
		} catch (ExecutionException e) {
			throw new Word2VecException("Model Training thread encountered exception", e.getCause());
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new RuntimeException("Word2VecTrainer interrupted", e);
		}
	}

	private void writeWordModel(BufferedWriter fo) throws IOException {
		int a;
		int b;
		// Save the word vectors
		fo.write(String.format("%d %d\n", vocab_size, layer1_size));
		for (a = 0; a < vocab_size; a++) {
			fo.write(String.format("%s ", vocab[a].word));
			// we only support non-binary for now
			for (b = 0; b < layer1_size; b++)
				fo.write(String.format("%f ", syn0[a * layer1_size + b]));
			fo.write("\n");
		}
	}

	private static int ArgPos(String str, String[] argv) {
		int a;
		for (a = 0; a < argv.length; a++)
			if (str.equals(argv[a])) {
				if (a == argv.length - 1) {
					System.out.printf("Argument missing for %s\n", str);
					System.exit(1);
				}
				return a;
			}
		return -1;
	}

	/** For testing word2vec training with command line and training file. */
	public static void main(String[] args) throws IOException, Word2VecException {
		int i;
		if (args.length == 0) {
			System.out.printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
			System.out.printf("Options:\n");
			System.out.printf("Parameters for training:\n");
			System.out.printf("\t-train <file>\n");
			System.out.printf("\t\tUse text data from <file> to train the model\n");
			System.out.printf("\t-output <file>\n");
			System.out.printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
			System.out.printf("\t-size <int>\n");
			System.out.printf("\t\tSet size of word vectors; default is 100\n");
			System.out.printf("\t-window <int>\n");
			System.out.printf("\t\tSet max skip length between words; default is 5\n");
			System.out.printf("\t-sample <float>\n");
			System.out.printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
			System.out.printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
			System.out.printf("\t-hs <int>\n");
			System.out.printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
			System.out.printf("\t-negative <int>\n");
			System.out.printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
			System.out.printf("\t-threads <int>\n");
			System.out.printf("\t\tUse <int> threads (default 12)\n");
			System.out.printf("\t-iter <int>\n");
			System.out.printf("\t\tRun more training iterations (default 5)\n");
			System.out.printf("\t-min-count <int>\n");
			System.out.printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
			System.out.printf("\t-alpha <float>\n");
			System.out.printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
			System.out.printf("\t-classes <int>\n");
			System.out.printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
			System.out.printf("\t-debug <int>\n");
			System.out.printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
			System.out.printf("\t-binary <int>\n");
			System.out.printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
			System.out.printf("\t-read-vocab <file>\n");
			System.out.printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
			System.out.printf("\t-cbow <int>\n");
			System.out.printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
			System.out.printf("\nExamples:\n");
			System.out.printf("java NewWord2vecTrainer -train data.txt -output vec.txt -cbow 0 -size 100 -window 5 -negative 0 -min-count 1 -hs 1 -sample 1e-3 -threads 1 -binary 0 -iter 15\n\n");
			return;
		}
		String output_file = "";
		String read_vocab_file = "";

		String train_file = "";
		vocab_word[] vocab;
		int cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12;
		int vocab_max_size = 1000, layer1_size = 100;
		int iter = 5;
		float alpha = 0.025f;
		float sample = 1e-3f;

		// Hierachical softmax
		boolean hs = false;
		int negative = 5;
		if ((i = ArgPos("-size", args)) >= 0) layer1_size = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-train", args)) >= 0) train_file = args[i + 1];
		if ((i = ArgPos("-read-vocab", args)) >= 0) read_vocab_file = args[i + 1];
		if ((i = ArgPos("-debug", args)) >= 0) debug_mode = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-cbow", args)) >= 0) cbow = Integer.parseInt(args[i + 1]);
		if (cbow != 0) alpha = 0.05f;
		if ((i = ArgPos("-alpha", args)) >= 0) alpha = Float.parseFloat(args[i + 1]);
		if ((i = ArgPos("-output", args)) >= 0) output_file = args[i + 1];
		if ((i = ArgPos("-window", args)) >= 0) window = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-sample", args)) >= 0) sample = Float.parseFloat(args[i + 1]);
		if ((i = ArgPos("-hs", args)) >= 0) hs = Integer.parseInt(args[i + 1]) != 0;
		if ((i = ArgPos("-negative", args)) >= 0) negative = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-threads", args)) >= 0) num_threads = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-iter", args)) >= 0) iter = Integer.parseInt(args[i + 1]);
		if ((i = ArgPos("-min-count", args)) >= 0) min_count = Integer.parseInt(args[i + 1]);
		// TODO port word classifications later
		//if ((i = ArgPos("-classes", args))>=0)classes = Integer.parseInt(args[i + 1]);
		vocab = new vocab_word[vocab_max_size];
		for (int j = 0; j < vocab_max_size; j++) {
			vocab[j] = new vocab_word();
		}
		try (RandomAccessFile trainingFile = new RandomAccessFile(new File(train_file), "r")) {
			new Word2VecTrainer(
					AutoLog.getLog(),
					layer1_size,
					FileWordIterator.getSentencesFromFile(trainingFile),
					Optional.absent(),
					read_vocab_file,
					debug_mode,
					cbow,
					alpha,
					output_file,
					window,
					sample,
					hs,
					negative,
					num_threads,
					iter,
					min_count).TrainModel();
		}
	}

}
