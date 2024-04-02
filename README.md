# RAG-based tool to query the ML Engineeering Open Book

The [ML Engineering Open Book](https://github.com/stas00/ml-engineering) by [Stas Bekman](https://twitter.com/StasBekman) is quite informative and I often tend to refer to it for a quick lookup. I decided to build a quick RAG-based tool to query the repository to find answers to my questions.

Refer steps below to try it out -

1. Clone the ML Engineering Open Book repository
    ```
    git clone https://github.com/stas00/ml-engineering
    ```

2. Keep the cloned repository in the root folder of this repository or change the path in [app.py](app.py) script.

3. Install dependencies
    ```
    pip install requirements.txt
    ```

4. Run the app
    ```
    python app.py
    ```

5. Use the [query.py](query.py) script to search with your query.

Optionally, convert the query script to a Gradio app if needed. 

I'm sure there can be improvements made in the app, however this version is also proving to be quite useful to me, especially with the sources.