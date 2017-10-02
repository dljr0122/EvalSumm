from __future__ import print_function
from __future__ import division
import sys
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cplex
from cplex.exceptions import CplexError

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

counter = 0
debug = False


class Algorithms:
    """Several algorithms for obtaining word cover of a document."""

    def __init__(self):
        """set some class level variables."""
        self.mcdonald_answer = 'nothing'
        self.dynamic_table = {}
        self.dynamic_ans = []
        self.remove_table = {}
        self.subset_table = {}
        self.sorted_table = {}
        self.vectorizer = TfidfVectorizer(norm=None, min_df=1)

    @staticmethod
    def greedy(sent_set, word_set, dictionary,
               update=False, penalty=False,
               word_threshold=False,
               word_count=100):
        global debug
        scores = {}
        greedy_answer = []
        words_in_answer = 0
        for sent in sent_set:
            scores[sent] = Counter(sent)
        full = set(word_set)

        def update_scores(words_of_max):
            for sentence, sent_counter in scores.items():
                for word, freq in sent_counter.items():
                    if word in words_of_max:
                        if penalty:
                            scores[sentence][word] *= -1
                        else:
                            scores[sentence][word] = 0

        while full and (not word_threshold or words_in_answer < word_count):
            max_sent = max(scores, key=lambda key: sum(scores[key].values()))
            if debug:
                print()
                print('sent:', str(max_sent))
                print('words:', str(full), '\nmax:', str(set(max_sent)))

            if update:
                update_scores(full.intersection(set(max_sent)))
            if debug:
                for k, v in scores.items():
                    print('key:', str(k), 'value:', str(v))
            if max_sent not in greedy_answer:
                greedy_answer.append(max_sent)
                del scores[max_sent]
                words_in_answer += dictionary[max_sent][1]
            full = full.difference(set(max_sent))

        return greedy_answer

    @staticmethod
    def greedy2(sent_set, word_set, original_size, dictionary,
                rval=0,
                update=False, penalty=False,
                word_threshold=False,
                word_count=100):
        """greedy selection based on ratio of unique word count to original word count."""
        global debug
        scores = {}
        greedy_answer = []
        words_in_answer = 0
        for sent in sent_set:
            scores[sent] = Counter(sent)
        full = set(word_set)

        def update_scores(words_of_max):
            for sentence, sent_counter in scores.items():
                for word, freq in sent_counter.items():
                    if word in words_of_max:
                        if penalty:
                            scores[sentence][word] *= -1
                        else:
                            scores[sentence][word] = 0

        while full and (not word_threshold or words_in_answer < word_count):
            # as_list.sort(key=lambda sent: len(sent) / len(max(original_size[sent], key=len)), reverse=True)
            max_sent = max(scores, key=lambda key: (
                sum(scores[key].values()) / pow(dictionary[key][1], rval)
                if pow(dictionary[key][1], rval) != 0
                else 0, dictionary[key][1], len(dictionary) - dictionary[key][2]))
            if debug:
                print()
                print('r:', str(sum(scores[max_sent].values()) / dictionary[max_sent][1]))
                # print(len(original_size[max_sent]))
                print('n:', sum(scores[max_sent].values()))
                print('d:', dictionary[max_sent][1])
                print('sent:', str(max_sent))
                print('words:', str(full), '\nmax:', str(set(max_sent)))

            if update:
                update_scores(full.intersection(set(max_sent)))
            if debug:
                for k, v in scores.items():
                    print('key:', str(k), 'value:', str(v))
            if max_sent not in greedy_answer:
                greedy_answer.append(max_sent)
                del scores[max_sent]
                words_in_answer += dictionary[max_sent][1]
            full = full.difference(set(max_sent))
        return greedy_answer

    @staticmethod
    def tfidf(sent_set, word_set, dictionary, rval=0,
              ratio=False, update=False, penalty=False,
              use_threshold=False, word_count=100):
        full = set(word_set)

        class Score:
            def inc_tf(self):
                self.tf += 1

            def inc_df(self):
                self.df += 1

            def update_score(self):
                self.idf = math.log((self.size / self.df))
                self.tf_idf = self.tf * self.idf

            def set_tf(self, count):
                self.tf = count
                self.update_score()

            def penalize_tf(self):
                self.tf *= -1
                self.update_score()

            def get_tfidf(self):
                return self.tf_idf

            def __repr__(self):
                return repr(self.tf_idf)

            def __init__(self, s):
                self.tf = 1
                self.df = 1
                self.size = s
                self.idf = math.log((self.size / self.df))
                self.tf_idf = self.tf * self.idf

        size = len(sent_set)
        scores = {}
        for sentence in sent_set:
            seen_words = []
            for word in sentence:
                if word not in seen_words:
                    seen_words.append(word)
                    if word not in scores:
                        scores[word] = Score(size)
                    else:
                        scores[word].inc_tf()
                        scores[word].inc_df()
                        scores[word].update_score()
                else:
                    scores[word].inc_tf()
                    scores[word].update_score()
        # print(scores)

        sentence_values = []
        for sentence in sent_set:
            value = 0
            for word in sentence:
                value += scores[word].get_tfidf()
            if ratio:
                if len(sentence) != 0:
                    value = value / pow(dictionary[sentence][1], rval)
                    sentence_values.append([sentence, value])
                else:
                    continue
            else:
                sentence_values.append([sentence, value])
                # print([sentence, value])

        # debugging for sentence values
        for val in sentence_values:
            print('s: {}\nv: {}\n'.format(val[0], val[1]))
            
        def update_sentence_values(s, u_penalty=False):
            for w in s:
                if u_penalty:
                    scores[w].penalize_tf()
                else:
                    scores[w].set_tf(0)
            for s, v in sentence_values:
                if debug:
                    print('s:', s, 'v:', v)
                    print(sentence_values[sentence_values.index([s,v])])
                val = 0
                for t in s:
                    val += scores[t].get_tfidf()
                sentence_values[sentence_values.index([s, v])] = [s, val / dictionary[s][1]] if ratio else [s, val]

        progress = 0
        words_in_answer = 0
        greedy_answer = []
        while full and (not use_threshold or words_in_answer < word_count):
            progress += 1
            if debug: print(progress)
            best = max(sentence_values, key=lambda v: v[1])
            if debug: print('max:', best[0])
            sentence_values.remove(best)
            full = full.difference(set(best[0]))
            if update:
                update_sentence_values(set(best[0]), u_penalty=penalty)
            for item in sentence_values:
                if item[1] == 0:
                    sentence_values.remove(item)
            greedy_answer.append(best[0])
            words_in_answer += dictionary[best[0]][1]

        return greedy_answer

    @staticmethod
    def stfidf(sent_set, word_set, dictionary,
               rval=0,
               ratio=False, update=False, penalty=False,
               use_threshold=False, word_count=100):
        class Score:
            def inc_df(self):
                self.df += 1
                self.idf = math.log(self.size / self.df)

            def inc_tf(self):
                self.tf += 1

            def set_tf(self, new_count):
                self.tf = new_count

            def penalize_tf(self):
                self.tf *= -1

            def get_tfidf(self):
                return self.idf * self.tf

            def __repr__(self):
                return repr(self.idf)

            def __init__(self, s):
                self.df = 1
                self.tf = 0
                self.size = s
                self.idf = math.log((self.size / self.df))

        size = len(sent_set)
        scores = {}

        # calculate idf scores
        for sentence in sent_set:
            # a list that tracks words seen in sentence
            # this is to ensure that df is updated only
            # once per sentence
            seen_words = []
            for word in sentence:
                if word not in scores:
                    scores[word] = Score(size)
                    seen_words.append(word)
                else:
                    if word not in seen_words:
                        scores[word].inc_df()

        def update_sentence_values():
            ret_val = []
            for sent in sent_set:
                sent_tfidf_sum = sum(scores[w].get_tfidf() for w in sent)

                # store correct value based on ratio or not
                if ratio:
                    if len(sent) != 0:
                        sent_tfidf_sum /= pow(dictionary[sent][1], rval)
                        ret_val.append([sent, sent_tfidf_sum])
                else:
                    ret_val.append([sent, sent_tfidf_sum])

            return ret_val

        for sentence in sent_set:
            for word in sentence:
                scores[word].inc_tf()
        sentence_values = update_sentence_values()

        progress = 0
        words_in_answer = 0
        greedy_answer = []
        full = set(word_set)
        while full and (not use_threshold or words_in_answer < word_count):
            progress += 1
            # print(progress)

            # get highest scoring sentence
            best = max(sentence_values, key=lambda v: v[1])
            sentence_values.remove(best)

            # remove every word in best from full list of words
            if update:
                for word in full.intersection(set(best[0])):
                    if debug: print(progress, word)
                    if penalty:
                        scores[word].penalize_tf()
                    else:
                        scores[word].set_tf(0)
                sentence_values = update_sentence_values()
            full = full.difference(set(best[0]))

            greedy_answer.append(best[0])
            words_in_answer += dictionary[best[0]][1]

        return greedy_answer

    # @staticmethod
    def is_subset(self, set1, set2):
        if (set1, set2) not in self.subset_table:
            for s1 in set1:
                found = False
                for s2 in set2:
                    if s1 == s2:
                        found = True
                        break
                if not found:
                    self.subset_table[(set1, set2)] = False
                    return False
            self.subset_table[(set1, set2)] = True
            return True
        else:
            return self.subset_table[(set1, set2)]

    @staticmethod
    def remove_tuples(set1, set2):
        # if (set1, set2) not in self.remove_table:
        #     set1_as_list = list(set1)
        #     for item in set2:
        #         if item in set1:
        #             set1_as_list.remove(item)
        #     self.remove_table[(set1, set2)] = tuple(set1_as_list)
        #     return tuple(set1_as_list)
        # else:
        #     return self.remove_table[(set1, set2)]
        set1_as_list = list(set1)
        for item in set2:
            if item in set1:
                if item in set1_as_list:
                    set1_as_list.remove(item)
        return tuple(set1_as_list)

    @staticmethod
    def add_tuple(set1, set2):
        set1_as_list = list(set1)
        set1_as_list.append(set2)
        return tuple(set1_as_list)

    @staticmethod
    def get_sorted_tuple(unsorted_tuple):
        as_list = list(unsorted_tuple)
        as_list.sort(key=len)
        return tuple(as_list)

    def dynamic_lookup(self, sent_set, word_set):
        if len(sent_set) == 0:
            return
        sent_set = self.get_sorted_tuple(sent_set)
        yes_no, max_used, count = self.dynamic_table[(sent_set, word_set)]
        sent_set_no_max = self.remove_tuples(sent_set, (sent_set[-1],))
        word_set_no_max = self.remove_tuples(word_set, sent_set[-1])
        if yes_no:
            if max_used:
                sys.stdout.write(str(sent_set[-1]))
                self.dynamic_lookup(sent_set_no_max, word_set_no_max)
            else:
                self.dynamic_lookup(sent_set_no_max, word_set)
        else:
            return

    def dynamic_calc_answer(self, sent_set, word_set):
        if len(sent_set) == 0:
            return
        sent_set = self.get_sorted_tuple(sent_set)
        yes_no, max_used, count = self.dynamic_table[(sent_set, word_set)]
        sent_set_no_max = self.remove_tuples(sent_set, (sent_set[-1],))
        word_set_no_max = self.remove_tuples(word_set, sent_set[-1])
        if yes_no:
            if max_used:
                self.dynamic_ans.append(sent_set[-1])
                self.dynamic_calc_answer(sent_set_no_max, word_set_no_max)
            else:
                self.dynamic_calc_answer(sent_set_no_max, word_set)
        else:
            if debug: print('')
            return

    def dynamic(self, sent_set_tuple, word_set_tuple):
        sent_set_tuple = self.get_sorted_tuple(sent_set_tuple)
        yes_no, max_used, count = self.dynamic_tuple(sent_set_tuple, word_set_tuple)
        self.dynamic_table[(sent_set_tuple, word_set_tuple)] = yes_no, max_used, count

    def dynamic_tuple(self, sent_set_tuple, word_set_tuple):
        """ returns boolean1, boolean2, int
            boolean1: does answer exist
            boolean2: was the max sentence used
            int     : current number of sentences used
        """
        global counter
        counter += 1
        # if counter % 100000 == 0: print counter
        if len(sent_set_tuple) == 1:
            # if there is a word set, and all the words are subset of the sentence
            if word_set_tuple and self.is_subset(word_set_tuple, sent_set_tuple[0]):
                # return True, sent_set_tuple.__hash__()
                return True, True, 1
            elif not word_set_tuple:
                # return True, ().__hash__()
                return True, False, 0
            else:
                # return False, ().__hash__()
                return False, False, 0
        max_sent = sent_set_tuple[-1]
        sent_set_tuple_no_max_sent = self.remove_tuples(sent_set_tuple, (max_sent,))
        word_set_tuple_no_max_sent = self.remove_tuples(word_set_tuple, max_sent)
        if (sent_set_tuple_no_max_sent, word_set_tuple) not in self.dynamic_table:
            bool_wo, ans_wo, cnt_wo = self.dynamic_tuple(sent_set_tuple_no_max_sent, word_set_tuple)
            self.dynamic_table[(sent_set_tuple_no_max_sent, word_set_tuple)] = bool_wo, ans_wo, cnt_wo
        else:
            bool_wo, ans_wo, cnt_wo = self.dynamic_table[(sent_set_tuple_no_max_sent, word_set_tuple)]
        if (sent_set_tuple_no_max_sent, word_set_tuple_no_max_sent) not in self.dynamic_table:
            bool_w, ans_w, cnt_w = self.dynamic_tuple(sent_set_tuple_no_max_sent, word_set_tuple_no_max_sent)
            self.dynamic_table[(sent_set_tuple_no_max_sent, word_set_tuple_no_max_sent)] = bool_w, ans_w, cnt_w
        else:
            bool_w, ans_w, cnt_w = self.dynamic_table[(sent_set_tuple_no_max_sent, word_set_tuple_no_max_sent)]

        # ans_w = self.add_tuple(ans_w, max_sent)
        cnt_w += 1
        # increase cnt_w because max_sent is taken/needed/used
        if bool_wo and bool_w:
            if cnt_w < cnt_wo:
                return True, True, cnt_w
            else:
                return True, False, cnt_wo
        elif bool_wo:
            return True, False, cnt_wo
        elif bool_w:
            return True, True, cnt_w
        else:
            return False, False, 0

    def ilp(self, sentences):
        """
        This will find optimal set using integer linear programming.
        :param sentences: tuple of lists for all sentences in document.
        :return: list containing indexes of solution set.
        """

        # Use sklearn vectorizer to produce TDM (term-document matrix)
        list_sentences = [list(s) for s in sentences]  # sklearn uses lists not tuples
        # override the tokenizer and preprocessor, because data already preprocessed
        vectorizer = CountVectorizer(min_df=0,
                                     tokenizer=lambda x: x,
                                     preprocessor=lambda x: x)
        # need transpose because sklearn produces DT
        TD = vectorizer.fit_transform(list_sentences).transpose()
        print(TD.toarray())
        print(vectorizer.vocabulary_)

        # this is for pre-processed data. duplicate sentences removed
        sentence_count = len(sentences)
        my_obj = np.ones((sentence_count,))
        my_ub = np.ones((sentence_count,))
        my_lb = np.zeros((sentence_count,))
        my_ctype = "I" * sentence_count
        my_colnames = [str(x) for x in range(sentence_count)]

        # number of terms is the number of rows in TD
        vocab_size = TD.shape[0]
        my_rhs = np.ones((vocab_size,))
        my_rownames = sorted(vectorizer.vocabulary_)
        my_sense = "G" * vocab_size

        problem = cplex.Cplex()  # initialize problem
        problem.objective.set_sense(problem.objective.sense.minimize)  # make it a minimization problem

        problem.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype, names=my_colnames)

        # each row represents the LHS of constraint equations.
        rows = []

        # build the equations for each term of TDM
        for r in range(vocab_size):
            if debug: print("r_i: ", r, my_rownames[r])
            # for the r-th row, print all nonzero entries.
            if debug: print("nonzero():", TD.getrow(r).nonzero())
            colnames = []
            # get the r-th row. look at second part of tuple.
            col_coefs = TD.getrow(r).nonzero()[1]
            for coef in col_coefs:
                # print('coef:', coef)
                # colnames represent the original sentence.
                colnames.append(my_colnames[coef])
            if debug: print('sentences that include the term', colnames)
            # add constraint to rows[]
            rows.append([colnames, [1] * len(col_coefs)])
            if debug: print()

        if debug: print('rows', rows)

        # add constraints to the problem
        problem.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs, names=my_rownames)

        try:
            problem.solve()
        except CplexError as exc:
            print(exc)
            exit()

        if debug: print()
        # solution.get_status() returns an integer code
        if debug: print("Solution status = ", problem.solution.get_status(), ":", end=' ')
        # the following line prints the corresponding string
        if debug: print(problem.solution.status[problem.solution.get_status()])
        # how many sentences needed for minimal set cover?
        if debug: print("Solution value  = ", problem.solution.get_objective_value())

        numcols = problem.variables.get_num()
        numrows = problem.linear_constraints.get_num()

        slack = problem.solution.get_linear_slacks()
        # STORE SOLUTION HERE!
        x = problem.solution.get_values()
        print(x)

        for j in range(numrows):
            if debug: print("Row %d:  Slack = %10f" % (j, slack[j]))

        # PRINT SOLUTION HERE!
        for j in range(numcols):
            if debug: print("Column %d:  Value = %10f" % (j, x[j]))

        sentences_used = 0
        for j in range(numcols):
            if x[j] == 1:
                sentences_used += 1
                if debug: print(sentences[j])

        if debug: print('used:', sentences_used)
        if debug: print('sentences:', numcols)
        if debug: print('ratio:', sentences_used / numcols)

        answer = []
        for idx in range(len(x)):
            if x[idx]:
                answer.append(sentences[idx])

        self.dynamic_ans = answer

        return x
        pass

    def bottom_up(self, sentences):
        # initialize necessary variables
        list_sentences = list(sentences)
        word_sets = []  # set of word sets for dynamic table
        used = []  # set of sentences already processed
        seen_words = []  # set of words reviewed

        """
        This section builds the list of word_sets (i.e. columns of table). It does not create all 
        permutations of possible word_sets. Instead it only includes permutations that will be used 
        by the bottum-up algorithm. This is possible when the original sentences are sorted by size 
        (use original position for ties).
        """
        # sort list by length and then position
        if debug: print('sentences', sentences)
        list_sentences.sort(key=lambda x: (len(x), sentences.index(x)))
        if debug: print('list_sentences', list_sentences, end='\n\n')
        while list_sentences:
            new_word_sets = []  # will hold new word sets to include based on included minimum sentence

            # get min item, based on length and position
            min_item = min(list_sentences,
                           key=lambda x: (len(x), sentences.index(x)))  # probably replace with list_sentences[0]
            list_sentences.remove(min_item)  # remove from list

            # add the min item itself as a word set
            new_word_sets.append(min_item)

            # column with all previously seen words plus min item
            word_set = self.tuple_union(tuple(seen_words), min_item)
            new_word_sets.append(word_set)
            if debug:
                print('{:>10s} : {}'.format('min', min_item))
                print('{:>10s} : {}'.format('list', list_sentences))
                print('{:>10s} : {}'.format('word set', word_set))
                print('{:>10s} : {}'.format('new sets', new_word_sets))

            # create sub word sets by deleting words of previous sentences (in order)
            used.append(min_item)  # add current sentence to used sentences list
            temp = list(used)  # copy list of previously used sentences
            seen_words = list(word_set)  # copy word list
            while temp:
                # find max item in used sentences (traversing rows backwards)
                max_item = max(temp, key=lambda x: (len(x), sentences.index(x)))
                if debug: print('  {:>>12s}: {}'.format(' max', max_item))
                # from the current word set, remove max_item words
                for word in max_item:
                    if seen_words:
                        if word in seen_words:
                            seen_words.remove(word)
                    if debug: print('    {:>>12s}: {}'.format(' word', word))
                    if debug: print('    {:>>12s}: {}'.format(' seen list', seen_words))
                new_word_sets.append(tuple(seen_words))  # once word set has a sentence removed add it as word_set
                if debug: print('  {:>>12s}: {}'.format(' added', seen_words))
                temp.remove(max_item)

            # add all new word sets to dynamic table columns
            word_sets.extend(new_word_sets)

            # update relevant variables for next iteration
            # used.append(min_item)
            seen_words = list(word_set)
            if debug:
                print('{:>10s} : {}'.format('end used', used))
                print('{:>10s} : {}'.format('end seen', seen_words))
                print('{:>10s} : {}'.format('end cols', word_sets))
                print('*' * 50)

        # remove duplicates and then sort
        # [set(item) for item in set(frozenset(item) for item in L)]
        word_sets = [set(item) for item in set(frozenset(item) for item in word_sets)]
        word_sets.sort(key=lambda x: (len(tuple(x)), tuple(x)))
        if debug: print('{:>10s} : {}'.format('final', word_sets));print('*' * 50)

        # self.bottom_up()
        # get sorted list of sentences based on set of sentence tuples
        list_sentences = sorted(list(sentences), key=lambda x: (len(x), sentences.index(x)))

        """
        build table until all sentences are used.
        empty table to be filled in by dynamic programming
        each entry has 3 values:
            no. of sentences used for answer,
            sum of sentence lengths of solution sentences,
            ptr to next table entry
        """
        rows = len(sentences) + 1  # add one to account for row that has no sentences
        cols = len(word_sets)  # assumes empty word set is included
        # initialize empty table
        table = [[[None, None, None] for x in range(cols)] for x in range(rows)]
        for i in range(rows):
            if i is not 0:
                sentence = list_sentences[i - 1]
            else:
                sentence = ()
            for j, word_set in enumerate(word_sets):
                if debug: print('\ni ={:>2d}, j ={:>2d}\n {:>4s}:{}\n {:>4s}:{}'
                                .format(i, j, 'sent', sentence, 'set', word_set))
                # check for non-base case, because majority will be non-base
                if word_set and sentence:
                    # word set that has curr sentence words removed
                    set_wo_curr = self.tuple_remove_intersection(word_set, sentence)
                    # if no entry in table, assume no answer possible that uses current sentence
                    if set_wo_curr not in word_sets:
                        use_new = [None, None, None]
                    else:
                        # table lookup for sub-problem
                        j_wo_curr = word_sets.index(set_wo_curr)
                        use_new = list(table[i - 1][j_wo_curr])
                    # get previous entry that does NOT use current sentence
                    # doing with deep copies, to prevent changes to table
                    existing = list(table[i - 1][j])
                    if debug:
                        print('{:>12s}:{}'.format('/w new', use_new))
                        print('{:>12s}:{}'.format('w/o new', existing))
                    # check that valid solution in both
                    if use_new[0] is not None and existing[0] is not None:
                        if existing[0] < (use_new[0] + 1):
                            table[i][j] = [existing[0], existing[1], (i - 1, j)]
                        elif existing[0] > (use_new[0] + 1):
                            table[i][j] = [use_new[0] + 1, use_new[1] + len(sentence), (i - 1, j_wo_curr)]
                        else:  # means both solutions use equal no. of sentences
                            if existing[1] <= (use_new[1] + len(sentence)):
                                table[i][j] = [existing[0], existing[1], (i - 1, j)]
                            else:
                                table[i][j] = [use_new[0] + 1, use_new[1] + len(sentence), (i - 1, j_wo_curr)]
                    # only the existing solution is valid
                    elif existing[0] is not None:
                        table[i][j] = [existing[0], existing[1], (i - 1, j)]
                    # only the new solution is valid
                    elif use_new[0] is not None:
                        table[i][j] = table[i][j] = [use_new[0] + 1, use_new[1] + len(sentence), (i - 1, j_wo_curr)]
                    else:
                        if debug: print('No Valid Solutions')
                        table[i][j] = [None, None, None]
                # base case
                elif not word_set:  # word_set is empty
                    table[i][j] = [0, 0, None]
                elif not sentence:  # sentence is empty
                    table[i][j] = [None, None, None]

                if debug: print('entry:{}'.format(table[i][j]))
        """
        for creating the solution, if table traverses up a column (e.g. no row change) do not add current sentence
        but if table moves diagonally (e.g. next table entry changes both i and j),
        then include current sentence to summary.
        """
        has_next = (len(list_sentences), len(word_sets) - 1)
        answer = []
        while has_next:
            j = has_next[1]
            has_next = table[has_next[0]][has_next[1]][2]
            if has_next is None:
                continue
            elif has_next[1] != j:
                answer.append(list_sentences[has_next[0]])
        self.dynamic_ans = answer
        if debug: print(answer)

    def tuple_union(self, tuple1, tuple2):
        """ Creates new tuple that is set union of tuple1 and tuple2"""
        list1 = list(tuple1)
        list2 = list(tuple2)
        union = list(set(list1 + list2))
        union.sort()
        return set(union)

    def tuple_remove_intersection(self, tuple1, tuple2):
        """ returns tuple1 where intersection with tuple2 is removed."""
        list1 = list(tuple1)
        for entry in tuple2:
            if entry in list1:
                list1.remove(entry)
        return set(list1)

    def mcdonald(self, mod_sentences, mod_words, dictionary, use_threshold=True, word_count=100):
        """
        mcdonald algorithm is for multi-setting document.
        In their algorithm it treats the basic thought unit as sentences.
        and uses the following for their relevance and redundance formuals
        rel(t_i) = Position(t_i, d_k)^-1 + CosineSimilarity(t_i, D)
        red(t_i, t_j) = CosineSimilarity(t_i, t_j)
        we simply use document bag-or-words vector for both the position
        and cosine similarity in relevance formula
        """
        corpus = [" ".join(sentence) for sentence in mod_sentences]

        # append a row for whole document
        # corpus.append(" ".join(corpus))
        if debug:
            print("sentences", str(mod_sentences))
            print("corpus", str(corpus))

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        if debug: print(tfidf_matrix.toarray())

        if debug:
            print()
            print(tfidf_matrix[0:1].toarray())
            print(tfidf_matrix[0].toarray())

        # produce bag-of-words tfidf vector for whole document
        doc_vector = self.vectorizer.transform([" ".join(corpus)])
        if debug: print('doc_vector:', doc_vector.toarray())

        # inner method that will be used by dynamic algorithm (relevance score)
        def relevance(idx):
            pos_score = 1.0 / (idx + 1)
            similarity = cosine_similarity(tfidf_matrix[idx], doc_vector)
            # because cosine_similarity returns matrix, 0th row and 0th position is just the value (i.e. [[.5]])
            return pos_score + similarity[0][0]

        # inner method that will be used by dynamic algorithm (redundancy score)
        def redundancy(thought1, thought2):
            return cosine_similarity(thought1, thought2)[0][0]

        """
        initialize for McDonald
        1) create empty table for mcdonald algorithm
        2) initialize initial column to empty set
        3) bottom up algorithm
        """
        # empty array
        # word_count=10
        # create a table with empty values
        # table = {}
        table_list_of_lists = [[[] for x in range(word_count + 1)] for y in range(len(mod_sentences))]
        # if debug: print(table_list_of_lists)

        # for all summaries of length 0 the ith thought unit is not used, and empty set solution (i.e. score)
        for i in range(tfidf_matrix.shape[0]):
            if debug: print(i)
            # table[i, 0] = [False, 0]
            table_list_of_lists[i][0] = [False, 0]

        # create dictionary of solution sets (later change this solution to a space efficient trie or BST)
        # key= tuple (i, k) where i is the ith table row, and k is the k number of words needed
        # value=tuple (s, r) where s is a set of used thought units, and r is the redundancy value of that set
        redundancy_values = {}

        def redundancy_sum(idx, kth):
            if (idx - 1, k - len(mod_sentences[idx])) in redundancy_values:
                prev = redundancy_values[idx - 1, k - len(mod_sentences[idx])]
                score = prev[1]
                for thought_idx in prev[0]:
                    score += redundancy(tfidf_matrix[idx], tfidf_matrix[thought_idx])
                redundancy_values[idx, kth] = (prev[0].union([idx]), score)
                return score
            else:
                redundancy_values[idx, kth] = ({idx}, 0)
                return 0

        # table records if ith thought unit is included or not, and the resulting score of solution, -1 for no solution
        for i in range(tfidf_matrix.shape[0]):
            for k in range(1, word_count + 1):
                # get previous solution without current thought unit
                # check that solution exists
                s1 = table_list_of_lists[i - 1][k] if i != 0 else [False, -1]
                # s1 = table[i - 1, k] if i != 0 else [False, -1]
                # produce solution with current thought unit.
                if k - len(mod_sentences[i]) >= 0:  # check not over current limit of words
                    prev_entry = table_list_of_lists[i - 1][k - len(mod_sentences[i])] if i != 0 else [False, -1]
                    # prev_entry = table[i - 1, k - len(mod_sentences[i])] if i != 0 else [False, -1]

                    redundancy_sum(i, k)
                    if prev_entry[1] != -1:
                        s2 = [True, prev_entry[1] + relevance(i) - redundancy_values[i, k][1]]
                    else:
                        s2 = [False, -1]
                else:
                    s2 = [False, -1]

                if s1[1] >= s2[1]:
                    table_list_of_lists[i][k] = [False, s1[1]]
                    # table[i, k] = [False, s1[1]]
                else:
                    table_list_of_lists[i][k] = [True, s2[1]]
                    # table[i, k] = [True, s2[1]]
            if debug: print(i, k, table_list_of_lists[i][k])
            # print(i, k, table[i, k])

        if debug:
            print('after dynamic:')
            print(mod_sentences)
            for row in table_list_of_lists:
                print(str(row))

        # print answer by finding largest K with solution
        answer = []
        for j in reversed(range(word_count + 1)):
            is_found = True if table_list_of_lists[len(mod_sentences) - 1][j][1] >= 0 else False
            if is_found:
                for i in reversed(range(len(mod_sentences))):
                    if table_list_of_lists[i][j][0]:
                        answer.append(mod_sentences[i])
                        j -= len(mod_sentences[i])
                    if j == 0: break
                break
        self.dynamic_ans = answer
        return answer
        if debug: print(answer)


if __name__ == "__main__":
    sentences = (('hi', 'i', 'is', 'daniel'), ('hi', 'you'), ('you',))
    words = ('i', 'is', 'daniel', 'hi', 'you')
    sentence_map = {('i', 'is', 'daniel'): 'I am daniel', ('hi', 'you'): 'Hello you.'}
    Algorithms.greedy(sentences, words, sentence_map)
