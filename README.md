# football

This is a football (soccer) chatbot. To begin with, it will just contain match scores, but it may expand to other features, like stats and predictions.

It uses RAG to be able to answer questions about the latest matches.

The program uses tokens stored in the environment variables: `FOOTBALL_DATA_TOKEN` (for the https://api.football-data.org token) and `HF_WRITE_TOKEN` (for the Hugging Face write token)

To keep the system self-contained and architecturally simple, the documents, embeddings, etc. are stored in memory. In a production system, you would likely want them to be persisted in a database and vector database.


It uses the [Football Data API](https://www.football-data.org/) to retrieve match data.