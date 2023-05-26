# Natural language processing course 2022/23: `Constructing a Co-Occurrence Graph from a Short Story Database: A Pipeline Approach`

Team members:
 * `Jer Pelhan`, `jp4861@student.uni-lj.si`
 * `Nina Velikajne`, `nv6920@student.uni-lj.si`

Group public acronym/name: `Mataviuc`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

____________________________________
## Description
This is repository for course NLP. We have chosen the first project, namely literacy situation models knowledge base creation.

Literature is a diverse field with unique characters and relationships that interact in complex ways. NLP may struggle to understand these elements due to the ambiguity and unclear references in natural language. In this assignment, we focus on short story NLP analysis. We build our corpus from Gutenberg short stories. On the collected corpus, we apply coreference resolution and then test several methods. First, we test AllanNLP and Stanza models for named entity recognition (NER). We implement and test out own BERT model for NER. An important aspect of literature is also single-character and character-to-character sentiment. We test Stanza and Vader models. Based on the captured information we build and analyse the co-occurrence graph and report the accuracy of each tested method.

## Project structure
* Folder [data](data/) contains our short stories corpus. We used two corpora. The first corpus consists of 44 short stories. The data was taken from a project of a group from last year. They annotated 55 stories. We decided to enlarge the corpus so we added 73 additional stories, but annotated only the characters, not the sentiment as well.
* [Notebook](corpus_analysis.ipynb) contains basic corpus analysis.
* Folder [src](src/) contains main code. Scripts can be run using the basic python command.
