# yw_text-mining
This is my independent study, using YesWorkflow to do text mining annotation, and do queries, using the provenance and retrospective provenance to answer queries that cannot be answered manually. 
# Introduction 
This is a use case with YesWorkflow, where combing the Latent Dirichlet Allocation models with provenance and retrospective provenance. There are three parts for this project: text mining part, YW annotation part, queries part. 
In the text mining part, it includes data cleaning,removing the stopwords, using regular expression to do filtering,lemmatize the words and add bigrams to the documents. Then use Gensim LDA model to do topic modeling, where firstly set the hyper parameters as 1.0/(number of topics) , and then use log files to do tracing,outputing log files when it converges. 
In the YW annotation part, I use the Yesworkflow annotation to annotate the python file, @IN and @OUT are used to show the input and output data, and @param is to input the parameter, @log is to trace the log file, @BEGIN and @END is to start and end the workflow. And use yw-recon to generate reconfacts file, and then use reconfacts file to do queries.
In the queries part, use yw-queries which includes simple generic queries and advanced generic queries. It will output the create a Graphviz-generated layout of the complete workflow, answer the downstream of the input and upstream of the output. 
*  [YesWorkflow](https://github.com/yesworkflow-org/yw-prototypes)
*  [Gensim](https://radimrehurek.com/gensim/models/ldamodel.html)
*  [yw-queries](https://github.com/yesworkflow-org/yw-idcc-17/tree/master/queries)
*  [yw-recon](https://github.com/yesworkflow-org/yw-tapp-15-recon) 
*  [NLTK](http://www.nltk.org/book/ch01.html)
*  [LOG](https://docs.python.org/3/library/logging.html)
