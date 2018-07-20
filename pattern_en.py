from pattern.en import referenced, pluralize, singularize, comparative, superlative, conjugate, number, numerals, lemma, lexeme, tenses,\
    PAST, PL, verbs, conjugate, PARTICIPLE, quantify, suggest, ngrams, parse, tag, tokenize, pprint, parsetree, tree, Text, WORD, POS, CHUNK, PNP, REL, LEMMA, sentiment, \
    Sentence, Word, Chunk, PNPChunk, modality, wordnet, ADJECTIVE


#indefinite article
print referenced('university')
print referenced('hour')
# pluralization and singularization
print pluralize('child')
print singularize('wolves')
# comparative and superlative
print comparative('bad')
print superlative('bad')
# verb conjugation
print lexeme('purr')
print lemma('purring')
print conjugate('purred', '3sg') # he / she / it
print 'p' in tenses('purred') # By alias.
print PAST in tenses('purred')
print (PAST, 1, PL) in tenses('purred')
# rule-based conjugation
print 'google'  in verbs.infinitives
print 'googled' in verbs.inflections
print conjugate('googled', tense=PARTICIPLE, parse=False)
print conjugate('googled', tense=PARTICIPLE, parse=True)
# quantification
print number("seventy-five point two")  # "seventy-five point two" => 75.2
print numerals(2.245, round=2)  # 2.245 => "two point twenty-five"
print quantify(['goose', 'goose', 'duck', 'chicken', 'chicken', 'chicken'])
print quantify({'carrot': 100, 'parrot': 20})
print quantify('carrot', amount=1000)
# spelling
print suggest("parot")
# n-grams
print ngrams("I am eating pizza.", n=2)  # bigrams
print ngrams("I am eating pizza.", n=3, punctuation=".,;:!?()[]{}`''\"@#$^&*+-|=~_", continuous=False)
# parser
print parse('I eat pizza with a fork.',
            tokenize = True,  # Split punctuation marks from words?
            tags = True,  # Parse part-of-speech tags? (NN, JJ, ...)
            chunks = True,  # Parse chunks? (NP, VP, PNP, ...)
            relations = False,  # Parse chunk relations? (-SBJ, -OBJ, ...)
            lemmata = False,  # Parse lemmata? (ate => eat)
            encoding = 'utf-8',  # Input string encoding.
            tagset = None)  # Penn Treebank II (default) or UNIVERSAL.
# parser tagger and tokenizer
for word, pos in tag('I feel *happy*!', tokenize=True, encoding='utf-8'):
    if pos == "JJ": # Retrieve all adjectives.
        print word
print tokenize('I feel *happy*!', punctuation=".,;:!?()[]{}`''\"@#$^&*+-|=~_", replace={})
# parser output
pprint(parse('I ate pizza.', relations=True, lemmata=True))
# parse trees
s = parsetree('The cat sat on the mat.',
              tokenize = True,  # Split punctuation marks from words?
              tags = True,  # Parse part-of-speech tags? (NN, JJ, ...)
              chunks = True,  # Parse chunks? (NP, VP, PNP, ...)
              relations = False,  # Parse chunk relations? (-SBJ, -OBJ, ...)
              lemmata = False,  # Parse lemmata? (ate => eat)
              encoding = 'utf-8',  # Input string encoding.
              tagset = None)  # Penn Treebank II (default) or UNIVERSAL.
print repr(s)
for sentence in s:
    for chunk in sentence.chunks:
        print chunk.type, [(w.string, w.type) for w in chunk.words]
for sentence in tree(open('data/input/tagged.txt'), token=[WORD, POS, CHUNK]):  # CHECK FOR ERROR
    print sentence
# text
text = Text(open('data/input/corpus.txt'), token=[WORD, POS, CHUNK, PNP, REL, LEMMA])
# text = Text.from_xml('data/input/multilingual-all-words.en.xml')  # Reads an XML string generated with Text.xml.
print text.string  # 'The cat sat on the mat .'
print text.sentences  # [Sentence('The cat sat on the mat .')]
print text.copy()
print text.xml
# sentence
# sentence = Sentence(open('data/input/corpus.txt'), token=[WORD, POS, CHUNK, PNP, REL, LEMMA])
# sentence = Sentence.from_xml(xml)
print sentence.parent  # Sentence parent, or None.
print sentence.id  # Unique id for each sentence.
print sentence.start  # 0
print sentence.stop  # len(Sentence).
print sentence.string  # Tokenized string, without tags.
print sentence.words  # List of Word objects.
print sentence.lemmata  # List of word lemmata.
print sentence.chunks  # List of Chunk objects.
print sentence.subjects  # List of NP-SBJ chunks.
print sentence.objects  # List of NP-OBJ chunks.
print sentence.verbs  # List of VP chunks.
print sentence.relations  # {'SBJ': {1: Chunk('the cat/NP-SBJ-1')}, 'VP': {1: Chunk('sat/VP-1')}, 'OBJ': {}}
print sentence.pnp  # List of PNPChunks: [Chunk('on the mat/PNP')]
print sentence.constituents(pnp=False)
# print sentence.slice([0], [1])
print sentence.copy()
print sentence.xml
# sentence words
word = Word('The cat sat on the mat.', 'The cat sat on the mat.', lemma=None, type=None, index=0)
print word.sentence              # Sentence parent.
print word.index                 # Sentence index of word.
print word.string                # String (Unicode).
print word.lemma                 # String lemma, e.g. 'sat' => 'sit',
print word.type                  # Part-of-speech tag (NN, JJ, VBD, ...)
print word.chunk                 # Chunk parent, or None.
print word.pnp                   # PNPChunk parent, or None.
# sentence chunk
chunk = Chunk('The cat sat on the mat.', words=[], type=None, role=None, relation=None)
print chunk.sentence             # Sentence parent.
# print chunk.start                # Sentence index of first word.
# print chunk.stop                 # Sentence index of last word + 1.
print chunk.string               # String of words (Unicode).
print chunk.words                # List of Word objects.
print chunk.lemmata              # List of word lemmata.
# print chunk.head                 # Primary Word in the chunk.
print chunk.type                 # Chunk tag (NP, VP, PP, ...)
print chunk.role                 # Role tag (SBJ, OBJ, ...)
print chunk.relation             # Relation id, e.g. NP-SBJ-1 => 1.
print chunk.relations            # List of (id, role)-tuples.
# print chunk.related              # List of Chunks with same relation id.
# print chunk.subject              # NP-SBJ chunk with same id.
# print chunk.object               # NP-OBJ chunk with same id.
# print chunk.verb                 # VP chunk with same id.
# print chunk.modifiers            # []
print chunk.conjunctions         # []
print chunk.pnp                  # PNPChunk parent, or None.
# print chunk.previous(type=None)
# print chunk.next(type=None)
# print chunk.nearest(type='VP')
# propositional noun phrases
pnp = PNPChunk('The cat sat on the mat.', words=[], type=None, role=None, relation=None)
print pnp.string                 # String of words (Unicode).
print pnp.chunks                 # List of Chunk objects.
# print pnp.preposition            # First PP chunk in the PNP.
# sentiment
print sentiment("The movie attempts to be surreal by incorporating various time paradoxes,"
                "but it's presented in such a ridiculous way it's seriously boring.")
print sentiment('Wonderfully awful! :-)').assessments
# mode and modality
s = "Some amino acids tend to be acidic while others may be basic." # weaseling
s = parse(s, lemmata=True)
s = Sentence(s)
print modality(s)
# wordnet
s = wordnet.synsets('bird')[0]
print 'Definition:', s.gloss  # Definition string.
print '  Synonyms:', s.synonyms  # List of word forms (i.e., synonyms)
print ' Hypernyms:', s.hypernyms()  # returns a list of  parent synsets (i.e., more general). Synset (semantic parent).
print ' Hypernyms:', s.hypernyms(recursive=False, depth=None)
print '  Hyponyms:', s.hyponyms()  # returns a list child synsets (i.e., more specific).
print '  Hyponyms:', s.hyponyms(recursive=False, depth=None)
print '  Holonyms:', s.holonyms()  # List of synsets (of which this is a member).
print '  Meronyms:', s.meronyms()  # List of synsets (members/parts).
print '       POS:', s.pos  # Part-of-speech: NOUN | VERB | ADJECTIVE | ADVERB.
print '  Category:', s.lexname  # Category string, or None.
print 'Info Cont.:', s.ic  # Information Content (float).
print '   Antonym:', s.antonym  # Synset (semantic opposite).
print '   Synsets:', s.similar()  # List of synsets (similar adjectives/verbs).
# sense similarity
a = wordnet.synsets('cat')[0]
b = wordnet.synsets('dog')[0]
c = wordnet.synsets('box')[0]
print wordnet.ancestor(a, b)
print wordnet.similarity(a, a)
print wordnet.similarity(a, b)
print wordnet.similarity(a, c)
# synset sentiment
print wordnet.synsets('happy', ADJECTIVE)[0].weight
print wordnet.synsets('sad', ADJECTIVE)[0].weight
