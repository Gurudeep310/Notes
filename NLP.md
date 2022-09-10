```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold) @@

```
<mark>Marked text</mark>
<mark style="background-color: lightblue">Marked text</mark>
<span style="color: green"> Some green text </span>

----
We can use NLP to create systems like speech recognition, document summarization, machine translation, spam detection, named entity recognition, question answering, autocomplete, predictive typing and so on.

**<$\rightarrow$>** What is Corpus ?
A <mark>Corpus is defined as a collection of text documents</mark> for example a data set containing news is a corpus or the tweets containing Twitter data is a corpus.

Corpus $\rightarrow$ Documents $\rightarrow$ Paragraphs $\rightarrow$ Sentences $\rightarrow$ Tokens

<mark>Tokens can be words, phrases, or Engrams, and Engrams are defined as the group of n words together.</mark>

For example, consider this given sentence-
“I love my phone.”
uni-grams(n=1) are: I, love, my, phone
Di-grams(n=2) are: I love, love my, my phone
And tri-grams(n=3) are: I love my, love my phone

----

**<$\rightarrow$>** What is Tokenization?
<mark>Tokenization is a process of splitting a text object into smaller units called tokens</mark>.
The most commonly used tokenization process is **White-space Tokenization**.

----

**<$\rightarrow$>** What is White Space Tokenization?
Also known as **unigram tokenization**. <mark>Entire text is split into words by splitting them from white spaces.</mark>

For example, in a sentence- “I went to New-York to play football.”
This will be splitted into following tokens: “I”, “went”, “to”, “New-York”, “to”, “play”, “football.”

**<$\rightarrow$>** What is Regular Expression Tokenization ?
<mark>A regular expression pattern is used to get the tokens.</mark>. We can split the text by passing a splitting pattern.

```python
Sentence= “Football, Cricket; Golf Tennis"
re.split(r’[;,\s]’, Sentence)
```
Tokens= “Football”, ”Cricket”, “Golf”, “Tennis”

**Tokenization can be performed at the sentence level or at the world level or even at the character level**

----

**<$\rightarrow$>** What is Normalization?
A Morpheme is defined as the base form of a word.A token is generally made up of two components, <mark>Morphemes, which are the base form of the word, and Inflectional forms, which are essentially the suffixes and prefixes added to morphemes.</mark>
Eg: Antinationalist: Anti + national + ist $\iff$ prefix + morpheme + suffix

<mark>Normalization is the process of converting a token into its base form.</mark> Helps in removing redundant information
Few normalization techniques are **Stemming and Lemmatization.**

----

**<$\rightarrow$>** What is Stemming?
<mark>Rule-based process to remove inflection part from a token.</mark> **Output is a stem**. They are easier to implement and usually run faster. Stemmer operates without knowledge of the context. Crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time.

Eg: "laughing","laughed","laughs","laugh" >>> "laugh"

**Not Good** as can produce words that are not in the dictionary
Eg:  “His teams are not winning”
After stemming the tokens that we will get are- “hi”, “team”, “are”, “not”,  “winn”
Notice that the keyword “winn” is not a regular word and “hi” changed the context of the entire sentence.

----

**<$\rightarrow$>** What is Lemmatization?
<mark>Lemmatization is a systematic step-by-step process for removing inflection forms of a word.</mark> **Output is a lemma**. Uses vocabulary and morphological analysis of words.
Eg: Running, Ran, Run >> Run

-----------
**<$\rightarrow$>** Parts of Speech Tags in Natural Language Processing

<mark>PoS tags is the properties of words that define their main context, their function, and the usage in a sentence.</mark>
Some of the commonly used parts of speech tags are- Nouns, which define any object or entity; Verbs, which define some action; and Adjectives or Adverbs, which act as the modifiers, quantifiers, or intensifiers in any sentence.
They are used in a variety of tasks such as text cleaning, feature engineering tasks, and word sense disambiguation.

-----

**<$\rightarrow$>** Grammar in NLP

- Constituency Grammar
- Dependency Grammar

What is Constituency Grammar?
<mark>Defines the structural pieces of a sentence, phrase, or clause driven by driven by their part of speech tags, noun or verb phrase identification.</mark>
Subject + Context + Object
“The dogs are barking in the park.”
“They are eating happily.”
“The cats are running since morning.”

What is Dependency Grammar?
<mark>Words of a sentence are dependent upon other words of the sentence.</mark>
<mark style="background-color: #e35f12"> I Didnt understand this</mark>
Used for Named Entity Recognition, Question Answering System, Coreference Resolution, Text summarization and Text classification 

----

What are Stop Words?
<mark>Stop words are filtered out before or after processing of text as they can add lot of noise.</mark> Eg: "and”, “the”, “a”