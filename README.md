# football
WIP

This is a football (soccer) chatbot. To begin with, it will just contain match scores, but it may expand to other features, like stats and predictions.

It uses RAG to be able to answer questions about the latest matches.

The program uses tokens stored in the environment variables: `FOOTBALL_DATA_TOKEN` (for the https://api.football-data.org token)

To keep the system self-contained and architecturally simple, the documents, embeddings, etc. are stored in memory. In a production system, you would likely want them to be persisted in a database and vector database.

To create a production system you would want to create an evaluation, among other things. The main reason that there is no evaluation yet is that the LLM is very slow on my measly laptop with no GPU (even having an evaluation with 10 queries would take hours).

It uses the [Football Data API](https://www.football-data.org/) to retrieve match data.