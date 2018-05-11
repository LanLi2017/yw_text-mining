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
*  [log-queries](https://github.com/kurator-org/kurator-SPNHC17-YW)
*  [yw-recon](https://github.com/yesworkflow-org/yw-tapp-15-recon) 
*  [NLTK](http://www.nltk.org/book/ch01.html)
*  [LOG](https://docs.python.org/3/library/logging.html)
*  [yw-case](https://github.com/yesworkflow-org/yw-idcc-17)

# Demo Queries

Please read [Query README](https://github.com/idaks/dataone-ahm-2016-poster/blob/master/queries/README.md) in the demo repo.

# Layouts of Repository

| Directory | Description                                                          |
|-----------| :--------------------------------------------------------------------|
|queries/ | it stores the scripts to the nine demo queries we asked.|
|rules/| it contains a set of Prolog rules for generating prospective yesworkflow views rules (`yw_rules.P` and `yw_views.P`), retrospective reconstructed rules (`recon_rules.P`), graph rendering rules (`gv_rules.P`), and populating graph rules (`yw_graph_rules.P`).|
|text_mining/| A python workflow project `YW_text_mining` that is a real-life use case and consists of multiple scripts.|
|yw_jar/| Contains two version YesWorkflow Java library.|

The project text_mining subfolders also have a typical folder structure:
|nipstxt/| Dataset used for text mining|
|topic_modeling/| A directory that contains the yesworkflow and topic modeling |
|settings.sh/| Dataset used for text mining|
`yw_text-minging/text_mining/topic_modeling/` 

Subfolders that all `topic_modeling` folders have:


| Directory | Description                                                          |
|-----------| :--------------------------------------------------------------------|
| script/ | the example script or scripts that make up  \<my_example\> |
| facts/ | the YW facts for \<my_example\>, generated by running YW on the example script(s)|
| views/ | materialized views for \<my_example\>|
| recon/ | reconstructed provenance used for \<my_example\>|
| results/ | all artifacts generated by make.sh|
|supplementary/ | a folder with supplementary files and information about the example|
| clean.sh | removes generated demo artifacts for \<my_example\> |
| make.sh | creates demo artifacts for \<my_example\> |
Please 
Note: after running `clean.sh` and `make.sh`, you can use git status to see what demo artifacts have just been created.

```
